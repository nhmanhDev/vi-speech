"""Runtime ASR chỉ cần numpy + onnxruntime (không torch) — dùng trên edge device.

    asr = StreamingASR("exported/stt_small", int8=True)
    for pcm in mic_chunks():          # float32 16 kHz, độ dài bất kỳ (vd 20 ms)
        partial = asr.accept(pcm)     # text tạm thời (partial transcript)
    final = asr.finish()
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort

from ..audio import LogMel, StreamingLogMel


def _session(path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class _Tokens:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "tokens.txt"), encoding="utf-8") as f:
            self.itos = f.read().split("\n")

    def text(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i > 1)


class OfflineASR:
    def __init__(self, model_dir: str, int8: bool = True, threads: int = 2):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.fe = LogMel(**self.cfg["features"])
        self.sess = _session(os.path.join(model_dir, "offline.int8.onnx" if int8 else "offline.onnx"), threads)
        self.tokens = _Tokens(model_dir)

    def transcribe(self, wav: np.ndarray) -> str:
        feats = self.fe(wav)[None]
        lp = self.sess.run(None, {"feats": feats})[0][0]
        ids = lp.argmax(-1)
        keep = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
        return self.tokens.text(ids[keep].tolist()).strip()


class StreamingASR:
    def __init__(self, model_dir: str, int8: bool = True, threads: int = 1):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        c = self.cfg
        self.fe = StreamingLogMel(LogMel(**c["features"]))
        self.sess = _session(os.path.join(model_dir, "stream.int8.onnx" if int8 else "stream.onnx"), threads)
        self.tokens = _Tokens(model_dir)
        self.win, self.hop = c["window_frames"], c["hop_frames"]
        self.reset()

    def reset(self):
        c = self.cfg
        self.fe.reset()
        self.feats = np.zeros((0, c["features"]["n_mels"]), dtype=np.float32)
        self.k = np.zeros((c["n_layers"], 1, c["n_heads"], c["cache_frames"], c["head_dim"]), dtype=np.float32)
        self.v = np.zeros_like(self.k)
        self.conv = np.zeros((c["n_layers"], 1, c["d_model"], c["conv_kernel"] - 1), dtype=np.float32)
        self.offset = 0
        self.ids: list[int] = []
        self.prev = -1

    def _step(self, window: np.ndarray):
        lp, self.k, self.v, self.conv = self.sess.run(None, {
            "feats": window[None], "k_cache": self.k, "v_cache": self.v,
            "conv_cache": self.conv, "offset": np.array(self.offset, dtype=np.int64),
        })
        self.offset += self.cfg["chunk"]
        for i in lp[0].argmax(-1).tolist():  # greedy CTC giữ trạng thái qua các chunk
            if i != self.prev and i != 0:
                self.ids.append(i)
            self.prev = i

    @property
    def text(self) -> str:
        return self.tokens.text(self.ids).strip()

    def accept(self, samples: np.ndarray) -> str:
        self.feats = np.concatenate([self.feats, self.fe.accept(samples)])
        while len(self.feats) >= self.win:
            self._step(self.feats[: self.win])
            self.feats = self.feats[self.hop :]
        return self.text

    def finish(self) -> str:
        """Đệm 'im lặng' (log-mel rất nhỏ) cho chunk cuối rồi trả kết quả cuối."""
        if len(self.feats) > 3:
            pad = np.full((self.win - len(self.feats), self.feats.shape[1]), np.log(1e-10), dtype=np.float32)
            self._step(np.concatenate([self.feats, pad]))
        text = self.text
        self.reset()
        return text
