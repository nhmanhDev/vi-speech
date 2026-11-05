"""Bảng ký tự tiếng Việt cho CTC (STT) và TTS.

Tiếng Việt có 12 nguyên âm cơ sở (a ă â e ê i o ô ơ u ư y) × 6 thanh = 72 ký tự nguyên âm,
cộng phụ âm. Mô hình char-level với ~100 token là đủ, không cần BPE: mỗi âm tiết
tiếng Việt ngắn (1–7 ký tự) và chính tả gần như "viết sao đọc vậy".
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

BASE_VOWELS = "aăâeêioôơuưy"
TONE_MARKS = ["", "̀", "́", "̉", "̃", "̣"]  # ngang huyền sắc hỏi ngã nặng
CONSONANTS = "bcdđghklmnpqrstvx" + "fjwz"  # fjwz: từ mượn / tên riêng

VOWELS = [unicodedata.normalize("NFC", v + t) for v in BASE_VOWELS for t in TONE_MARKS]
LETTERS = sorted(set(VOWELS) | set(CONSONANTS))

BLANK = "<blank>"
PAD = "<pad>"
UNK = "<unk>"
SPACE = " "
PUNCT = list(",.?!")


class Vocab:
    """Ánh xạ ký tự <-> id.

    STT (CTC): id 0 = blank.  TTS: id 0 = pad. Hai vai trò dùng chung id 0.
    """

    def __init__(self, with_punct: bool = False):
        specials = [BLANK, UNK]
        chars = [SPACE] + LETTERS + (PUNCT if with_punct else [])
        self.itos: list[str] = specials + chars
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.blank_id = 0
        self.pad_id = 0
        self.unk_id = 1

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids if i > self.unk_id)

    def ctc_collapse(self, ids: Iterable[int]) -> str:
        """Greedy CTC: gộp ký tự lặp liên tiếp rồi bỏ blank."""
        out: list[int] = []
        prev = -1
        for i in ids:
            if i != prev and i != self.blank_id:
                out.append(i)
            prev = i
        return self.decode(out)


def split_tone(syllable: str) -> tuple[str, int]:
    """Tách thanh điệu: 'người' -> ('ngươi', 1).

    Mã thanh: 0 ngang, 1 huyền, 2 sắc, 3 hỏi, 4 ngã, 5 nặng.
    Dùng cho TTS tone-aware embedding.
    """
    decomposed = unicodedata.normalize("NFD", syllable)
    tone = 0
    kept = []
    for ch in decomposed:
        if ch in TONE_MARKS[1:]:
            tone = TONE_MARKS.index(ch)
        else:
            kept.append(ch)
    return unicodedata.normalize("NFC", "".join(kept)), tone
