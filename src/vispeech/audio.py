"""Đọc audio và trích đặc trưng log-mel.

Viết bằng NumPy thuần (không phụ thuộc torch) để cùng một code chạy được ở 3 nơi:
tiền xử lý dữ liệu, train, và inference trên edge (chỉ cần numpy + onnxruntime).
Đặc trưng lệch nhau giữa train và inference là lỗi rất hay gặp khi deploy ASR.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf


def load_audio(data: bytes | str, target_sr: int) -> np.ndarray:
    """Đọc wav/flac (bytes hoặc đường dẫn) -> float32 mono ở `target_sr`."""
    src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
    return wav


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import soxr  # nhanh và chất lượng tốt nếu có

        return soxr.resample(wav, orig_sr, target_sr).astype(np.float32)
    except ImportError:
        import torch
        import torchaudio.functional as AF

        return AF.resample(torch.from_numpy(wav), orig_sr, target_sr).numpy()


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Ma trận (n_mels, n_fft//2+1), tam giác trên thang mel (HTK), giống torchaudio mặc định."""
    fmax = fmax or sr / 2
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, center, hi = mel_pts[i : i + 3]
        up = (fft_freqs - lo) / (center - lo)
        down = (hi - fft_freqs) / (hi - center)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


@dataclass
class LogMel:
    """Log-mel 80 chiều, cửa sổ 25 ms, bước 10 ms (chuẩn ASR).

    Không padding hai đầu (center=False) để trích đặc trưng theo luồng (streaming)
    cho kết quả y hệt trích một lần cả file: frame thứ i luôn = samples[i*hop : i*hop+win].
    """

    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80

    def __post_init__(self):
        self.window = np.hanning(self.win_length + 1)[:-1].astype(np.float32)  # periodic Hann
        self.fb = mel_filterbank(self.sample_rate, self.n_fft, self.n_mels)

    def num_frames(self, num_samples: int) -> int:
        return max(0, 1 + (num_samples - self.win_length) // self.hop_length)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """(num_samples,) float32 -> (num_frames, n_mels) float32."""
        n = self.num_frames(len(wav))
        if n == 0:
            return np.zeros((0, self.n_mels), dtype=np.float32)
        idx = np.arange(self.win_length)[None, :] + self.hop_length * np.arange(n)[:, None]
        frames = wav[idx] * self.window
        spec = np.abs(np.fft.rfft(frames, n=self.n_fft, axis=1)) ** 2
        mel = spec.astype(np.float32) @ self.fb.T
        return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


class StreamingLogMel:
    """Trích log-mel theo từng khối audio nhỏ (vd 20 ms từ microphone / WebSocket)."""

    def __init__(self, fe: LogMel | None = None):
        self.fe = fe or LogMel()
        self.buf = np.zeros(0, dtype=np.float32)

    def reset(self):
        self.buf = np.zeros(0, dtype=np.float32)

    def accept(self, samples: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, samples.astype(np.float32)])
        n = self.fe.num_frames(len(self.buf))
        if n == 0:
            return np.zeros((0, self.fe.n_mels), dtype=np.float32)
        feats = self.fe(self.buf[: (n - 1) * self.fe.hop_length + self.fe.win_length])
        self.buf = self.buf[n * self.fe.hop_length :]
        return feats
