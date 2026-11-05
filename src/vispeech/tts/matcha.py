"""Matcha-TTS (Mehta et al., ICASSP 2024) viết lại gọn cho tiếng Việt.

    văn bản ─► text encoder (Transformer + RoPE) ─► μ (mel "trung bình" cho từng ký tự)
                     └─► duration predictor ─► số frame mỗi ký tự
    μ kéo giãn theo duration ─► decoder flow matching (U-Net 1D) ─► mel ─► Vocos ─► sóng âm

Khi train, căn chỉnh ký tự–frame tìm bằng Monotonic Alignment Search (MAS, từ Glow-TTS):
không cần forced aligner bên ngoài. Decoder học *conditional flow matching* (OT-CFM):
đi thẳng từ nhiễu Gaussian tới mel, nên khi suy luận chỉ cần 2–10 bước Euler,
nhanh hơn nhiều so với diffusion (50–1000 bước): quan trọng cho CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..stt.model import rope


@dataclass
class MatchaConfig:
    vocab_size: int = 100
    n_feats: int = 100  # mel bin (Vocos 24 kHz)
    n_speakers: int = 1
    spk_dim: int = 64
    # text encoder
    enc_dim: int = 192
    enc_heads: int = 2
    enc_layers: int = 6
    enc_ff: int = 768
    enc_kernel: int = 3
    dropout: float = 0.1
    # duration predictor
    dp_dim: int = 256
    # decoder U-Net
    dec_channels: tuple = (256, 256)
    dec_mid_blocks: int = 2
    dec_heads: int = 4
    sigma_min: float = 1e-4
    segment_frames: int = 192  # train decoder trên đoạn ngẫu nhiên ~2 s (như Matcha) để tiết kiệm bộ nhớ
    # Điều khiển ngữ điệu (prosody): vector z-score [cao độ, biên độ cao độ, năng lượng, tốc độ] mỗi câu.
    # 0 = tắt. Khi train, với xác suất cond_drop thay bằng 0 (= ngữ điệu trung bình của người nói),
    # để lúc suy luận không truyền gì cũng ra giọng "bình thường".
    n_prosody: int = 0
    pros_dim: int = 32
    cond_drop: float = 0.2

    @property
    def cond_dim(self) -> int:
        return (self.spk_dim if self.n_speakers > 1 else 0) + (self.pros_dim if self.n_prosody > 0 else 0)


def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = max_len or int(lengths.max())
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def intersperse(ids: list[int], blank: int) -> list[int]:
    """Chèn token trống giữa các ký tự (như VITS/Matcha): giúp MAS căn chỉnh ổn định hơn."""
    out = [blank] * (2 * len(ids) + 1)
    out[1::2] = ids
    return out


# ------------------------------------------------------------------ text encoder


class LayerNormC(nn.Module):
    """LayerNorm theo trục kênh cho tensor (B, C, T)."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ff, kernel, dropout):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Conv1d(d, 3 * d, 1)
        self.o = nn.Conv1d(d, d, 1)
        self.n1, self.n2 = LayerNormC(d), LayerNormC(d)
        self.ff1 = nn.Conv1d(d, ff, kernel, padding=kernel // 2)
        self.ff2 = nn.Conv1d(ff, d, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):  # x (B, C, T), mask (B, 1, T)
        b, c, t = x.shape
        q, k, v = self.qkv(x).view(b, 3, self.h, self.dh, t).permute(1, 0, 2, 4, 3)  # (B, H, T, Dh)
        pos = torch.arange(t, device=x.device)
        q, k = rope(q, pos), rope(k, pos)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.masked_fill(~mask[:, :, None, :], -1e4).softmax(-1)
        y = (self.drop(att) @ v).transpose(2, 3).reshape(b, c, t)
        x = self.n1(x + self.drop(self.o(y)))
        y = self.ff2(self.drop(F.relu(self.ff1(x * mask)))) * mask
        return self.n2(x + self.drop(y)) * mask


class TextEncoder(nn.Module):
    def __init__(self, cfg: MatchaConfig):
        super().__init__()
        d = cfg.enc_dim
        self.emb = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.emb.weight, 0.0, d**-0.5)
        self.prenet = nn.Sequential(*[m for _ in range(3) for m in (nn.Conv1d(d, d, 5, padding=2), LayerNormC(d), nn.ReLU(), nn.Dropout(0.5))])
        self.cond_dim = cfg.cond_dim
        d_in = d + self.cond_dim
        self.layers = nn.ModuleList(EncoderLayer(d_in, cfg.enc_heads, cfg.enc_ff, cfg.enc_kernel, cfg.dropout) for _ in range(cfg.enc_layers))
        self.proj_mu = nn.Conv1d(d_in, cfg.n_feats, 1)
        # Duration predictor (2 conv) trên đầu ra encoder đã detach, như Glow-TTS/Matcha
        self.dp = nn.Sequential(
            nn.Conv1d(d_in, cfg.dp_dim, 3, padding=1), nn.ReLU(), LayerNormC(cfg.dp_dim), nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.dp_dim, cfg.dp_dim, 3, padding=1), nn.ReLU(), LayerNormC(cfg.dp_dim), nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.dp_dim, 1, 1),
        )

    def forward(self, x, x_lens, cond=None):
        """cond: (B, cond_dim) = [speaker embedding, prosody embedding] hoặc None."""
        mask = sequence_mask(x_lens, x.shape[1])[:, None, :]  # (B, 1, T)
        h = self.emb(x).transpose(1, 2) * mask
        h = h + self.prenet(h) * mask  # residual prenet
        if self.cond_dim:
            h = torch.cat([h, cond[:, :, None].expand(-1, -1, h.shape[-1])], 1)
        for layer in self.layers:
            h = layer(h, mask)
        mu = self.proj_mu(h) * mask
        logw = self.dp(h.detach() * mask) * mask
        return mu, logw, mask


# ------------------------------------------------------------------ alignment


@torch.no_grad()
def maximum_path(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MAS: đường đơn điệu có tổng log-likelihood lớn nhất. logp, mask: (B, Tx, Ty) -> 0/1 (B, Tx, Ty).

    Quy hoạch động vector hoá theo Tx (lặp theo Ty), chạy được trên GPU.
    """
    b, tx, ty = logp.shape
    neg = -1e9
    logp = logp.float().masked_fill(~mask, neg)
    value = torch.full((b, tx), neg, device=logp.device)
    value[:, 0] = logp[:, 0, 0]
    direction = torch.zeros(b, tx, ty, dtype=torch.bool, device=logp.device)  # True = đi từ ký tự trước
    idx = torch.arange(tx, device=logp.device)
    for j in range(1, ty):
        prev = F.pad(value[:, :-1], (1, 0), value=neg)  # value[i-1]
        from_prev = prev > value
        direction[:, :, j] = from_prev
        value = torch.where(from_prev, prev, value) + logp[:, :, j]
        value = value.masked_fill(idx[None, :] > j, neg)  # không thể ở ký tự i > j
    # truy vết ngược
    path = torch.zeros(b, tx, ty, device=logp.device)
    x_lens = mask[:, :, 0].sum(1)
    y_lens = mask[:, 0, :].sum(1)
    cur = (x_lens - 1).clone()
    ar = torch.arange(b, device=logp.device)
    for j in range(ty - 1, -1, -1):
        active = j < y_lens
        path[ar[active], cur[active], j] = 1
        step = direction[ar, cur.clamp(min=0), j] & active & (cur > 0)
        cur = cur - step.long()
    return path * mask


def generate_path(durations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """durations (B, Tx) số nguyên -> ma trận căn chỉnh (B, Tx, Ty)."""
    cum = torch.cumsum(durations, 1)
    ty = mask.shape[-1]
    pos = torch.arange(ty, device=durations.device)[None, None, :]
    path = (pos < cum[:, :, None]) & (pos >= (cum - durations)[:, :, None])
    return path.float() * mask


# ------------------------------------------------------------------ decoder (U-Net 1D + flow matching)


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    a = 1000.0 * t[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.Mish())
        self.t = nn.Sequential(nn.Mish(), nn.Linear(tdim, cout))
        self.b2 = nn.Sequential(nn.Conv1d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.Mish())
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, mask, temb):
        h = self.b1(x * mask) + self.t(temb)[:, :, None]
        h = self.b2(h * mask)
        return (h + self.skip(x * mask)) * mask


class TransformerBlock(nn.Module):
    # Attention tự viết (không dùng nn.MultiheadAttention): bản kia khi export ONNX bị
    # đóng cứng độ dài chuỗi trong Reshape -> không chạy được câu dài/ngắn khác.
    def __init__(self, c, heads):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.o = nn.Linear(c, c)
        self.n2 = nn.LayerNorm(c)
        self.ff = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))

    def forward(self, x, mask):  # x (B, C, T)
        h = x.transpose(1, 2)
        b, t, c = h.shape
        q, k, v = self.qkv(self.n1(h)).reshape(b, t, 3, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) * (c // self.h) ** -0.5
        att = att.masked_fill(~mask[:, :, None, :], -1e4).softmax(-1)
        h = h + self.o((att @ v).transpose(1, 2).reshape(b, t, c))
        h = h + self.ff(self.n2(h))
        return h.transpose(1, 2) * mask


class UNetDecoder(nn.Module):
    """Ước lượng trường vận tốc v(x_t, μ, t) cho flow matching."""

    def __init__(self, cfg: MatchaConfig):
        super().__init__()
        ch = list(cfg.dec_channels)
        cin = 2 * cfg.n_feats + cfg.cond_dim
        tdim = ch[0] * 4
        self.tdim0 = ch[0]
        self.tmlp = nn.Sequential(nn.Linear(ch[0], tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.down = nn.ModuleList()
        c_prev = cin
        for i, c in enumerate(ch):
            last = i == len(ch) - 1
            self.down.append(nn.ModuleList([ResBlock(c_prev, c, tdim), TransformerBlock(c, cfg.dec_heads),
                                            nn.Identity() if last else nn.Conv1d(c, c, 3, stride=2, padding=1)]))
            c_prev = c
        self.mid = nn.ModuleList(nn.ModuleList([ResBlock(ch[-1], ch[-1], tdim), TransformerBlock(ch[-1], cfg.dec_heads)])
                                 for _ in range(cfg.dec_mid_blocks))
        self.up = nn.ModuleList()
        rev = ch[::-1]
        for i, c in enumerate(rev):
            last = i == len(rev) - 1
            c_out = rev[i + 1] if not last else rev[-1]
            self.up.append(nn.ModuleList([ResBlock(2 * c, c_out, tdim), TransformerBlock(c_out, cfg.dec_heads),
                                          nn.Identity() if last else nn.ConvTranspose1d(c_out, c_out, 4, stride=2, padding=1)]))
        self.final = nn.Sequential(nn.Conv1d(ch[0], ch[0], 3, padding=1), nn.GroupNorm(8, ch[0]), nn.Mish(), nn.Conv1d(ch[0], cfg.n_feats, 1))
        self.n_down = len(ch) - 1

    def forward(self, x, mask, mu, t, cond=None):
        temb = self.tmlp(time_embedding(t, self.tdim0))
        h = torch.cat([x, mu] + ([cond[:, :, None].expand(-1, -1, x.shape[-1])] if cond is not None else []), 1)
        masks, skips = [mask], []
        for res, tr, down in self.down:
            m = masks[-1]
            h = tr(res(h, m, temb), m)
            skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h * m)
                masks.append(m[:, :, ::2])
        for res, tr in self.mid:
            h = tr(res(h, masks[-1], temb), masks[-1])
        # Khối up thứ k làm việc ở cùng độ phân giải với khối down thứ (L-1-k).
        for k, (res, tr, up) in enumerate(self.up):
            m = masks[len(self.down) - 1 - k]
            h = tr(res(torch.cat([h, skips.pop()], 1), m, temb), m)
            if not isinstance(up, nn.Identity):
                h = up(h * m)
        return self.final(h * mask) * mask


# ------------------------------------------------------------------ full model


class MatchaTTS(nn.Module):
    def __init__(self, cfg: MatchaConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TextEncoder(cfg)
        self.decoder = UNetDecoder(cfg)
        self.spk_emb = nn.Embedding(cfg.n_speakers, cfg.spk_dim) if cfg.n_speakers > 1 else None
        self.pros_proj = (nn.Sequential(nn.Linear(cfg.n_prosody, cfg.pros_dim), nn.SiLU(), nn.Linear(cfg.pros_dim, cfg.pros_dim))
                          if cfg.n_prosody > 0 else None)
        self.register_buffer("mel_mean", torch.zeros(1))
        self.register_buffer("mel_std", torch.ones(1))
        self.align = 2 ** (len(cfg.dec_channels) - 1)

    def set_mel_stats(self, mean: float, std: float):
        self.mel_mean.fill_(mean)
        self.mel_std.fill_(std)

    def make_cond(self, batch: int, device, spk=None, prosody=None):
        """Ghép vector điều kiện. prosody=None -> 0 (ngữ điệu trung bình)."""
        parts = []
        if self.spk_emb is not None:
            parts.append(self.spk_emb(spk if spk is not None else torch.zeros(batch, dtype=torch.long, device=device)))
        if self.pros_proj is not None:
            if prosody is None:
                prosody = torch.zeros(batch, self.cfg.n_prosody, device=device)
            parts.append(self.pros_proj(prosody))
        return torch.cat(parts, -1) if parts else None

    def _fix_len(self, n: int) -> int:
        return int(math.ceil(n / self.align) * self.align)

    def forward(self, x, x_lens, y, y_lens, spk=None, prosody=None):
        """x (B, Tx) id ký tự; y (B, n_feats, Ty) log-mel; prosody (B, n_prosody) z-score. Trả dict các loss."""
        cfg = self.cfg
        if prosody is not None and self.training and cfg.cond_drop > 0:
            keep = (torch.rand(prosody.shape[0], 1, device=prosody.device) >= cfg.cond_drop).float()
            prosody = prosody * keep
        s = self.make_cond(x.shape[0], x.device, spk, prosody)
        y = (y - self.mel_mean) / self.mel_std
        mu_x, logw, x_mask = self.encoder(x, x_lens, s)
        y_mask = sequence_mask(y_lens, y.shape[-1])[:, None, :]
        attn_mask = x_mask.transpose(1, 2) & y_mask  # (B, Tx, Ty)

        # MAS: log N(y | μ_x, I) cho từng cặp (ký tự i, frame j), bỏ hằng số
        with torch.no_grad():
            mu32, y32 = mu_x.float(), y.float()
            logp = -0.5 * ((y32**2).sum(1, keepdim=True) - 2 * mu32.transpose(1, 2) @ y32 + (mu32**2).sum(1)[:, :, None])
            attn = maximum_path(logp, attn_mask)

        # Duration loss: log số frame thực (theo MAS) của mỗi ký tự
        logw_ = torch.log(1e-8 + attn.sum(-1))[:, None, :] * x_mask
        dur_loss = ((logw - logw_) ** 2).sum() / x_lens.sum()

        mu_y = (attn.transpose(1, 2) @ mu_x.transpose(1, 2)).transpose(1, 2)  # (B, F, Ty)
        prior_loss = (0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask).sum() / (y_mask.sum() * cfg.n_feats)

        # Flow matching trên đoạn ngẫu nhiên
        seg = self._fix_len(min(cfg.segment_frames, int(y_lens.max())))
        starts = [torch.randint(0, max(1, int(l) - seg + 1), (1,)).item() for l in y_lens]
        y_seg = torch.stack([F.pad(y[i, :, st : st + seg], (0, max(0, seg - y[i, :, st : st + seg].shape[-1]))) for i, st in enumerate(starts)])
        mu_seg = torch.stack([F.pad(mu_y[i, :, st : st + seg], (0, max(0, seg - mu_y[i, :, st : st + seg].shape[-1]))) for i, st in enumerate(starts)])
        m_seg = sequence_mask((y_lens - torch.tensor(starts, device=y.device)).clamp(max=seg), seg)[:, None, :]
        t = torch.rand(y.shape[0], device=y.device)
        x0 = torch.randn_like(y_seg)
        tt = t[:, None, None]
        xt = (1 - (1 - cfg.sigma_min) * tt) * x0 + tt * y_seg
        target = y_seg - (1 - cfg.sigma_min) * x0
        v = self.decoder(xt, m_seg, mu_seg, t, s)
        fm_loss = (((v - target) ** 2) * m_seg).sum() / (m_seg.sum() * cfg.n_feats)
        return {"dur": dur_loss, "prior": prior_loss, "fm": fm_loss, "attn": attn}

    @torch.no_grad()
    def synthesize(self, x, x_lens, n_steps: int = 4, temperature: float = 0.667, length_scale: float = 1.0, spk=None,
                   prosody=None):
        """Trả log-mel (B, n_feats, Ty) và số frame. prosody: (B, n_prosody) z-score, None = trung bình."""
        s = self.make_cond(x.shape[0], x.device, spk, prosody)
        mu_x, logw, x_mask = self.encoder(x, x_lens, s)
        # chặn trên 60 frame (~0,64 s) mỗi token: model chưa hội tụ có thể đoán độ dài vô lý -> tràn bộ nhớ
        w = torch.ceil(torch.exp(logw) * x_mask * length_scale).clamp(max=60)[:, 0]  # (B, Tx)
        y_lens = w.sum(1).clamp(min=1).long()
        ty = self._fix_len(int(y_lens.max()))
        y_mask = sequence_mask(y_lens, ty)[:, None, :]
        attn = generate_path(w.long(), x_mask.transpose(1, 2) & y_mask)
        mu_y = (attn.transpose(1, 2) @ mu_x.transpose(1, 2)).transpose(1, 2)
        z = torch.randn_like(mu_y) * temperature
        dt = 1.0 / n_steps
        for i in range(n_steps):  # Euler ODE từ t=0 (nhiễu) tới t=1 (mel)
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            z = z + dt * self.decoder(z, y_mask, mu_y, t, s)
        mel = z * self.mel_std + self.mel_mean
        return mel * y_mask, y_lens
