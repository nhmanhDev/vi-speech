"""Runtime TTS chỉ cần numpy + onnxruntime.

    tts = OnnxTTS("exported/tts_matcha")
    wav = tts("Xin chào, hôm nay là ngày 5/11/2026.")   # float32, 24 kHz
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import onnxruntime as ort

from ..text.normalize import normalize


def istft(mag: np.ndarray, phase: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """ISTFT 'center' + cửa sổ Hann, giống torch.istft dùng trong Vocos."""
    spec = mag * np.exp(1j * phase)  # (513, T)
    frames = np.fft.irfft(spec, n=n_fft, axis=0).T  # (T, n_fft)
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    frames = frames * win
    n = n_fft + hop * (len(frames) - 1)
    out = np.zeros(n, dtype=np.float32)
    norm = np.zeros(n, dtype=np.float32)
    for i, f in enumerate(frames):
        out[i * hop : i * hop + n_fft] += f
        norm[i * hop : i * hop + n_fft] += win**2
    pad = n_fft // 2
    out, norm = out[pad : n - pad], norm[pad : n - pad]
    return out / np.maximum(norm, 1e-11)


class OnnxTTS:
    def __init__(self, model_dir: str, threads: int = 2):
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        mk = lambda n: ort.InferenceSession(os.path.join(model_dir, n), so, providers=["CPUExecutionProvider"])  # noqa: E731
        self.enc, self.dec, self.voc = mk("encoder.onnx"), mk("decoder.onnx"), mk("vocos.onnx")
        self.stoi = {s: i for i, s in enumerate(self.cfg["vocab"])}

    def ids(self, text: str) -> np.ndarray:
        norm = normalize(text, keep_punct=True)
        ids = [self.stoi.get(c, 1) for c in norm]
        out = [self.cfg["blank_id"]] * (2 * len(ids) + 1)
        out[1::2] = ids
        return np.array([out], dtype=np.int64)

    def mel(self, text: str, n_steps: int = 4, temperature: float = 0.667, length_scale: float = 1.0, seed: int = 0):
        mu_x, logw = self.enc.run(None, {"ids": self.ids(text)})
        w = np.ceil(np.exp(logw[0, 0]) * length_scale).astype(np.int64)
        ty = int(w.sum())
        T = int(math.ceil(ty / self.cfg["align"]) * self.cfg["align"])
        mu_y = np.repeat(mu_x[0], w, axis=1)  # length regulator: lặp μ theo số frame
        mu_y = np.pad(mu_y, ((0, 0), (0, T - ty)))[None].astype(np.float32)
        mask = np.zeros((1, 1, T), dtype=np.float32)
        mask[..., :ty] = 1
        z = np.random.default_rng(seed).standard_normal(mu_y.shape).astype(np.float32) * temperature
        for i in range(n_steps):
            v = self.dec.run(None, {"z": z, "mask": mask, "mu": mu_y, "t": np.array([i / n_steps], dtype=np.float32)})[0]
            z = z + v / n_steps
        return (z * self.cfg["mel_std"] + self.cfg["mel_mean"])[:, :, :ty]

    def __call__(self, text: str, **kw) -> np.ndarray:
        mag, phase = self.voc.run(None, {"mel": self.mel(text, **kw).astype(np.float32)})
        return istft(mag[0], phase[0], self.cfg["n_fft"], self.cfg["hop_length"])
