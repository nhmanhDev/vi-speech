"""WER / CER bằng khoảng cách Levenshtein."""

from __future__ import annotations

from typing import Sequence


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


class ErrorRate:
    """Cộng dồn lỗi trên cả tập rồi mới chia (không lấy trung bình WER từng câu)."""

    def __init__(self):
        self.word_err = self.words = self.char_err = self.chars = 0

    def update(self, ref: str, hyp: str):
        self.word_err += edit_distance(ref.split(), hyp.split())
        self.words += len(ref.split())
        r, h = ref.replace(" ", ""), hyp.replace(" ", "")
        self.char_err += edit_distance(r, h)
        self.chars += len(r)

    @property
    def wer(self) -> float:
        return self.word_err / max(self.words, 1)

    @property
    def cer(self) -> float:
        return self.char_err / max(self.chars, 1)
