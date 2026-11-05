"""Đọc số tiếng Việt thành chữ.

Quy tắc chính:
- 1 ở hàng đơn vị sau hàng chục >= 2 đọc là "mốt" (21 -> hai mươi mốt).
- 4 ở hàng đơn vị sau hàng chục >= 2 đọc là "tư" (24 -> hai mươi tư).
- 5 ở hàng đơn vị sau hàng chục >= 1 đọc là "lăm" (15 -> mười lăm, 25 -> hai mươi lăm).
- Hàng chục bằng 0 nhưng có hàng trăm thì đọc "linh" (105 -> một trăm linh năm).
- Nhóm 3 chữ số đứng sau nhóm khác mà < 100 thì đọc thêm "không trăm"
  (1005 -> một nghìn không trăm linh năm).
"""

from __future__ import annotations

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
SCALES = ["", "nghìn", "triệu", "tỷ"]


def _read_triple(n: int, full: bool) -> list[str]:
    """Đọc một nhóm 0..999. `full` = nhóm không đứng đầu (cần "không trăm", "linh")."""
    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    words: list[str] = []

    if hundreds > 0 or full:
        words += [DIGITS[hundreds], "trăm"]

    if tens == 0:
        if units == 0:
            return words
        if words:
            words.append("linh")
        words.append(DIGITS[units])
        return words

    if tens == 1:
        words.append("mười")
    else:
        words += [DIGITS[tens], "mươi"]

    if units == 1:
        words.append("mốt" if tens >= 2 else "một")
    elif units == 4:
        words.append("tư" if tens >= 2 else "bốn")
    elif units == 5:
        words.append("lăm")
    elif units > 0:
        words.append(DIGITS[units])
    return words


def number_to_words(n: int) -> str:
    """Chuyển số nguyên (có thể âm) thành chữ tiếng Việt."""
    if n < 0:
        return "âm " + number_to_words(-n)
    if n == 0:
        return DIGITS[0]

    # Số >= 10^12: tách theo "tỷ" đệ quy (1.000 tỷ = một nghìn tỷ).
    if n >= 10**12:
        high, low = divmod(n, 10**9)
        head = number_to_words(high) + " tỷ"
        return head if low == 0 else head + " " + _read_below_billion_tail(low)

    return " ".join(_read_groups(n, leading=True))


def _read_groups(n: int, leading: bool) -> list[str]:
    groups: list[int] = []
    while n > 0:
        n, g = divmod(n, 1000)
        groups.append(g)
    words: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if g == 0:
            continue
        is_first = not words and leading
        words += _read_triple(g, full=not is_first)
        if SCALES[idx]:
            words.append(SCALES[idx])
    return words


def _read_below_billion_tail(n: int) -> str:
    return " ".join(_read_groups(n, leading=False))


def digits_to_words(s: str) -> str:
    """Đọc từng chữ số (số điện thoại, mã số): '0912' -> 'không chín một hai'."""
    return " ".join(DIGITS[int(c)] for c in s if c.isdigit())
