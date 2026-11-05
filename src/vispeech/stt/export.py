"""Export checkpoint -> ONNX (offline + streaming) và lượng tử hoá int8 cho CPU/edge.

    python -m vispeech.stt.export --ckpt runs/stt_small/best.pt --out exported/stt_small \
        --chunk 16 --left-chunks 4

Sinh ra:
    offline.onnx / offline.int8.onnx   cả câu một lần (batch 1, độ dài động)
    stream.onnx  / stream.int8.onnx    một bước streaming: chunk cố định + cache cố định
    tokens.txt, config.json            bảng ký tự và tham số runtime
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn

from ..audio import LogMel
from ..text.vocab import Vocab
from .model import ConformerConfig, ConformerCTC


class _Offline(nn.Module):
    def __init__(self, m: ConformerCTC):
        super().__init__()
        self.m = m

    def forward(self, feats):  # (1, T, F) -> log-prob (1, T', V)
        lengths = torch.full((feats.shape[0],), feats.shape[1], dtype=torch.long)
        logits, _ = self.m(feats, lengths)
        return logits.log_softmax(-1)


class _Stream(nn.Module):
    def __init__(self, m: ConformerCTC):
        super().__init__()
        self.m = m

    def forward(self, feats, k_cache, v_cache, conv_cache, offset):
        logits, k, v, c = self.m.forward_chunk(feats, k_cache, v_cache, conv_cache, offset)
        return logits.log_softmax(-1), k, v, c


def load_model(ckpt_path: str, use_ema: bool = True) -> ConformerCTC:
    """Nạp checkpoint; mặc định dùng trọng số EMA nếu có (thường tốt hơn trọng số bước cuối)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ConformerCTC(ConformerConfig(**ck["model_config"]))
    model.load_state_dict(ck["model_ema"] if use_ema and "model_ema" in ck else ck["model"])
    return model.eval()


def _export(module, args, path, input_names, output_names, dynamic_axes=None):
    torch.onnx.export(
        module, args, path, input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
    )


def quantize(src: str, dst: str):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Dynamic int8 cho MatMul/Gemm: trọng số int8, activation lượng tử hoá lúc chạy.
    # Không cần dữ liệu calibration; thường giảm ~3-4x kích thước, tăng tốc 1.5-3x trên CPU.
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=16, help="frame sau subsampling (x40 ms)")
    ap.add_argument("--left-chunks", type=int, default=4)
    ap.add_argument("--no-quant", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.ckpt)
    cfg = model.cfg
    fe = LogMel()

    _export(_Offline(model), (torch.randn(1, 400, cfg.n_mels),), os.path.join(args.out, "offline.onnx"),
            ["feats"], ["log_probs"], {"feats": {1: "T"}, "log_probs": {1: "T_out"}})

    win, hop = model.window_frames(args.chunk)
    cache_frames = args.chunk * args.left_chunks
    k, v, c = model.init_cache(cache_frames)
    _export(_Stream(model), (torch.randn(1, win, cfg.n_mels), k, v, c, torch.tensor(0, dtype=torch.long)),
            os.path.join(args.out, "stream.onnx"),
            ["feats", "k_cache", "v_cache", "conv_cache", "offset"], ["log_probs", "k_cache_out", "v_cache_out", "conv_cache_out"])

    if not args.no_quant:
        for name in ("offline", "stream"):
            quantize(os.path.join(args.out, f"{name}.onnx"), os.path.join(args.out, f"{name}.int8.onnx"))

    vocab = Vocab()
    with open(os.path.join(args.out, "tokens.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab.itos))
    runtime = {
        "features": {k_: getattr(fe, k_) for k_ in ("sample_rate", "n_fft", "win_length", "hop_length", "n_mels")},
        "chunk": args.chunk, "left_chunks": args.left_chunks, "window_frames": win, "hop_frames": hop,
        "cache_frames": cache_frames, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
        "head_dim": cfg.d_model // cfg.n_heads, "d_model": cfg.d_model, "conv_kernel": cfg.conv_kernel,
        "blank_id": vocab.blank_id, "chunk_ms": args.chunk * 40,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(runtime, f, indent=1)
    for fn in sorted(os.listdir(args.out)):
        print(f"{fn:22s} {os.path.getsize(os.path.join(args.out, fn)) / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
