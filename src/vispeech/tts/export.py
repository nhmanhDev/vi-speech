"""Export Matcha-TTS + Vocos sang ONNX cho CPU/edge.

    python -m vispeech.tts.export --ckpt runs/tts_matcha/last.pt --out exported/tts_matcha

Tạo 3 graph, ghép bằng numpy trong onnx_tts.py:
    encoder.onnx   ids (1, Tx) -> mu_x (1, F, Tx), logw (1, 1, Tx)
    decoder.onnx   z, mask, mu_y, t -> vận tốc v (1 bước Euler)
    vocos.onnx     mel (1, 100, T) -> mag, phase (1, 513, T)   [ISTFT làm bằng numpy]
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn

from ..text.vocab import Vocab
from .matcha import MatchaConfig, MatchaTTS


class _Enc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids):
        lens = torch.full((ids.shape[0],), ids.shape[1], dtype=torch.long)
        mu, logw, _ = self.m.encoder(ids, lens, None)
        return mu, logw


class _Dec(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, z, mask, mu, t):
        return self.m.decoder(z, mask.bool(), mu, t, None)


class _Vocos(nn.Module):
    def __init__(self, voc):
        super().__init__()
        self.backbone, self.out = voc.backbone, voc.head.out

    def forward(self, mel):
        x = self.out(self.backbone(mel)).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        return torch.exp(mag).clamp(max=1e2), p  # giống vocos.heads.ISTFTHead


def load(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = MatchaTTS(MatchaConfig(**ck["model_config"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocos", default="charactr/vocos-mel-24khz")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    m = load(args.ckpt)
    if m.cfg.n_speakers > 1:
        raise SystemExit("export đa giọng: chưa hỗ trợ")
    F_ = m.cfg.n_feats
    kw = dict(opset_version=17, dynamo=False)
    torch.onnx.export(_Enc(m), (torch.randint(1, 50, (1, 41)),), os.path.join(args.out, "encoder.onnx"),
                      input_names=["ids"], output_names=["mu_x", "logw"], dynamic_axes={"ids": {1: "Tx"}, "mu_x": {2: "Tx"}, "logw": {2: "Tx"}}, **kw)
    T = 128
    torch.onnx.export(_Dec(m), (torch.randn(1, F_, T), torch.ones(1, 1, T), torch.randn(1, F_, T), torch.tensor([0.5])),
                      os.path.join(args.out, "decoder.onnx"), input_names=["z", "mask", "mu", "t"], output_names=["v"],
                      dynamic_axes={"z": {2: "T"}, "mask": {2: "T"}, "mu": {2: "T"}, "v": {2: "T"}}, **kw)
    from vocos import Vocos

    voc = Vocos.from_pretrained(args.vocos).eval()
    torch.onnx.export(_Vocos(voc), (torch.randn(1, 100, T),), os.path.join(args.out, "vocos.onnx"),
                      input_names=["mel"], output_names=["mag", "phase"],
                      dynamic_axes={"mel": {2: "T"}, "mag": {2: "T"}, "phase": {2: "T"}}, **kw)
    vocab = Vocab(with_punct=True)
    cfg = {"vocab": vocab.itos, "blank_id": len(vocab), "mel_mean": float(m.mel_mean), "mel_std": float(m.mel_std),
           "n_feats": F_, "align": m.align, "sample_rate": 24000, "n_fft": 1024, "hop_length": 256}
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    for fn in sorted(os.listdir(args.out)):
        print(f"{fn:16s} {os.path.getsize(os.path.join(args.out, fn)) / 1e6:7.2f} MB")


if __name__ == "__main__":
    main()
