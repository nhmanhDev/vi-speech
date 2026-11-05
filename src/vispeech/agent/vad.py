"""VAD streaming + endpointing (phát hiện hết lượt nói).

Silero VAD (ONNX, ~2 MB) chạy trên từng khung 32 ms ở 16 kHz, chỉ cần numpy + onnxruntime.
Trên xác suất thô, ta thêm máy trạng thái có trễ (hysteresis):
- bắt đầu nói: prob >= `on` liên tục `start_ms`  (lọc tiếng gõ phím, tiếng "ừ" ngắn)
- hết lượt:     prob <  `off` liên tục `end_ms`   (người Việt hay ngừng ngắn giữa câu, 500–800 ms là hợp lý)

Endpointing chỉ bằng im lặng là cách đơn giản nhất. Bản tốt hơn kết hợp thêm mô hình
"turn detector" dựa trên ngữ nghĩa (câu đã trọn ý chưa), xem docs/AGENT.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

FRAME = 512  # 32 ms @ 16 kHz
CONTEXT = 64


def _default_model() -> str:
    import silero_vad

    return os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.onnx")


class SileroVAD:
    def __init__(self, path: str | None = None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path or _default_model(), so, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame.reshape(1, FRAME).astype(np.float32)], axis=1)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": np.array(16000, dtype=np.int64)})
        self.context = x[:, -CONTEXT:]
        return float(out[0, 0])


@dataclass
class VADEvent:
    kind: str  # "start" | "end"
    t: float  # giây tính từ đầu stream


class Endpointer:
    def __init__(self, vad=None, on: float = 0.5, off: float = 0.35, start_ms: int = 96, end_ms: int = 640):
        self.vad = vad or SileroVAD()
        self.on, self.off = on, off
        self.start_frames = max(1, start_ms // 32)
        self.end_frames = max(1, end_ms // 32)
        self.reset()

    def reset(self):
        self.vad.reset()
        self.buf = np.zeros(0, dtype=np.float32)
        self.speaking = False
        self.run = 0
        self.frames = 0
        self.last_prob = 0.0

    def accept(self, pcm: np.ndarray) -> list[VADEvent]:
        """pcm float32 16 kHz độ dài bất kỳ -> danh sách sự kiện start/end."""
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        events = []
        while len(self.buf) >= FRAME:
            p = self.vad(self.buf[:FRAME])
            self.buf = self.buf[FRAME:]
            self.frames += 1
            self.last_prob = p
            if not self.speaking:
                self.run = self.run + 1 if p >= self.on else 0
                if self.run >= self.start_frames:
                    self.speaking, self.run = True, 0
                    events.append(VADEvent("start", (self.frames - self.start_frames) * FRAME / 16000))
            else:
                self.run = self.run + 1 if p < self.off else 0
                if self.run >= self.end_frames:
                    self.speaking, self.run = False, 0
                    events.append(VADEvent("end", (self.frames - self.end_frames) * FRAME / 16000))
        return events
