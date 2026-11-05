"""Streaming Conformer-CTC cho tiếng Việt, hướng tới chạy realtime trên CPU / edge.

Thiết kế cho streaming (giống ý tưởng WeNet U2 "dynamic chunk training"):
- Subsampling conv2d *không padding* (valid) x4: 10 ms -> 40 ms / frame. Chia audio thành
  các cửa sổ chồng 3 frame thì kết quả từng khúc trùng khớp với chạy cả câu.
- Self-attention bị giới hạn bởi *chunk mask*: frame trong chunk c chỉ nhìn chunk c và
  `left_chunks` chunk trước. Khi train, chunk size được lấy ngẫu nhiên mỗi batch, nên
  một model dùng được cả chế độ offline (full context) lẫn streaming với độ trễ tùy chọn.
- Depthwise conv *causal* + LayerNorm (không BatchNorm) để không phụ thuộc tương lai.
- RoPE (rotary position) thay cho relative position bias: chỉ phụ thuộc khoảng cách,
  nên KV-cache dùng lại được khi stream dài vô hạn.
- `forward_chunk` nhận/trả cache kích thước cố định -> export ONNX tĩnh, chạy bằng
  onnxruntime trên CPU / ARM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConformerConfig:
    n_mels: int = 80
    vocab_size: int = 96
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 12
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    subsampling_channels: int = 256


# ---------------------------------------------------------------- building blocks


def subsampled_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Số frame sau 2 lớp conv k=3, s=2, không padding."""
    return ((lengths - 1) // 2 - 1) // 2


class Conv2dSubsampling4(nn.Module):
    def __init__(self, n_mels: int, channels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2),
            nn.SiLU(),
        )
        f = ((n_mels - 1) // 2 - 1) // 2
        self.out = nn.Linear(channels * f, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, T', D)
        x = self.conv(x.unsqueeze(1))
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f))


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary embedding. x: (B, H, T, Dh), pos: (T,) vị trí tuyệt đối (float)."""
    half = x.shape[-1] // 2
    inv = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / half))
    ang = pos.float()[:, None] * inv[None, :]  # (T, half)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d * mult, d), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class RoPEAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def _split(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v  # (B, H, T, Dh)

    def _attend(self, q, k, v, mask):
        # mask: bool, True = được phép nhìn. Viết tay (không dùng SDPA) để export ONNX ổn định.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(~mask, -1e4 if att.dtype == torch.float16 else -1e9)
        att = F.dropout(att.softmax(-1), self.p, self.training)
        b, h, t, dh = q.shape
        return self.drop(self.proj((att @ v).transpose(1, 2).reshape(b, t, h * dh)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(x)
        pos = torch.arange(x.shape[1], device=x.device)
        return self._attend(rope(q, pos), rope(k, pos), v, mask)

    def forward_chunk(self, x, k_cache, v_cache, offset):
        """x: (B, C, D); cache: (B, H, Lc, Dh) giữ Lc frame trước (đã RoPE). offset: số frame đã xử lý."""
        q, k, v = self._split(x)
        c, lc = x.shape[1], k_cache.shape[2]
        pos = offset + torch.arange(c, device=x.device)
        q, k = rope(q, pos), rope(k, pos)
        k_all = torch.cat([k_cache, k], dim=2)
        v_all = torch.cat([v_cache, v], dim=2)
        # Vị trí cache chưa được lấp (đầu stream) thì che đi.
        cache_pos = offset - lc + torch.arange(lc, device=x.device)
        valid = torch.cat([cache_pos >= 0, torch.ones(c, dtype=torch.bool, device=x.device)])
        out = self._attend(q, k_all, v_all, valid.view(1, 1, 1, -1))
        return out, k_all[:, :, c:], v_all[:, :, c:]


class CausalConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.k = kernel
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.norm2 = nn.LayerNorm(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def _body(self, x_padded):  # (B, D, k-1+T) -> (B, T, D)
        y = self.dw(x_padded)
        y = F.silu(self.norm2(y.transpose(1, 2))).transpose(1, 2)
        return self.drop(self.pw2(y).transpose(1, 2))

    def _pre(self, x):
        return F.glu(self.pw1(self.norm(x).transpose(1, 2)), dim=1)  # (B, D, T)

    def forward(self, x, pad_mask=None):
        y = self._pre(x)
        if pad_mask is not None:
            y = y.masked_fill(~pad_mask[:, None, :], 0.0)
        return self._body(F.pad(y, (self.k - 1, 0)))

    def forward_chunk(self, x, cache):  # cache: (B, D, k-1)
        y = torch.cat([cache, self._pre(x)], dim=2)
        return self._body(y), y[:, :, -(self.k - 1) :]


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ConformerConfig):
        super().__init__()
        d = cfg.d_model
        self.ff1 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.att = RoPEAttention(d, cfg.n_heads, cfg.dropout)
        self.conv = CausalConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, att_mask, pad_mask):
        x = x + 0.5 * self.ff1(x)
        x = x + self.att(x, att_mask)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)

    def forward_chunk(self, x, k_cache, v_cache, conv_cache, offset):
        x = x + 0.5 * self.ff1(x)
        a, k_cache, v_cache = self.att.forward_chunk(x, k_cache, v_cache, offset)
        x = x + a
        c, conv_cache = self.conv.forward_chunk(x, conv_cache)
        x = x + c
        x = x + 0.5 * self.ff2(x)
        return self.norm(x), k_cache, v_cache, conv_cache


def chunk_mask(t: int, chunk: int, left_chunks: int, device=None) -> torch.Tensor:
    """(T, T) bool. chunk<=0: full context. left_chunks<0: nhìn toàn bộ quá khứ."""
    if chunk <= 0:
        return torch.ones(t, t, dtype=torch.bool, device=device)
    i = torch.arange(t, device=device)
    q_chunk = i[:, None] // chunk
    k_chunk = i[None, :] // chunk
    m = k_chunk <= q_chunk
    if left_chunks >= 0:
        m &= k_chunk >= q_chunk - left_chunks
    return m


# ---------------------------------------------------------------- model


class ConformerCTC(nn.Module):
    def __init__(self, cfg: ConformerConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.zeros(cfg.n_mels))
        self.register_buffer("std", torch.ones(cfg.n_mels))
        self.subsample = Conv2dSubsampling4(cfg.n_mels, cfg.subsampling_channels, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def set_cmvn(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-5))

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor, chunk: int = 0, left_chunks: int = -1):
        """feats (B, T, n_mels) log-mel -> logits (B, T', V), lengths (B,)."""
        x = self.drop(self.subsample((feats - self.mean) / self.std))
        lengths = subsampled_lengths(lengths).clamp(min=0, max=x.shape[1])
        t = x.shape[1]
        pad_mask = torch.arange(t, device=x.device)[None, :] < lengths[:, None]  # (B, T)
        att_mask = chunk_mask(t, chunk, left_chunks, x.device)[None, None] & pad_mask[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, att_mask, pad_mask)
        return self.head(x), lengths

    # ------------------------------------------------------------ streaming

    def init_cache(self, cache_frames: int, batch: int = 1, device=None):
        c = self.cfg
        dh = c.d_model // c.n_heads
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        return (
            z(c.n_layers, batch, c.n_heads, cache_frames, dh),
            z(c.n_layers, batch, c.n_heads, cache_frames, dh),
            z(c.n_layers, batch, c.d_model, c.conv_kernel - 1),
        )

    @staticmethod
    def window_frames(chunk: int) -> tuple[int, int]:
        """(số mel frame cần cho mỗi bước, bước nhảy). Cửa sổ chồng nhau 3 frame."""
        return 4 * chunk + 3, 4 * chunk

    def forward_chunk(self, feats, k_cache, v_cache, conv_cache, offset):
        """feats: (B, 4*C+3, n_mels) -> logits (B, C, V) + cache mới. offset: tensor int (frame đã xử lý)."""
        x = self.subsample((feats - self.mean) / self.std)
        ks, vs, cs = [], [], []
        for i, blk in enumerate(self.blocks):
            x, k, v, cc = blk.forward_chunk(x, k_cache[i], v_cache[i], conv_cache[i], offset)
            ks.append(k)
            vs.append(v)
            cs.append(cc)
        return self.head(x), torch.stack(ks), torch.stack(vs), torch.stack(cs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
