"""Chuẩn hoá văn bản tiếng Việt (text normalization) cho STT và TTS.

Pipeline: Unicode NFC -> thống nhất kiểu bỏ dấu -> viết tắt -> ngày/giờ -> số/đơn vị
-> chữ thường -> bỏ ký tự ngoài bảng chữ cái.

Nhãn STT (VLSP, InfoRe) vốn đã ở dạng "spoken form": chữ thường, không dấu câu,
số viết bằng chữ. Hàm này đưa văn bản viết (written form) về cùng dạng đó, dùng cho:
- TTS: đầu vào người dùng gõ có số, ngày, viết tắt.
- STT: làm sạch nhãn của các dataset khác nhau cho đồng nhất.
"""

from __future__ import annotations

import re
import unicodedata

from .numbers import digits_to_words, number_to_words

# Cùng một chữ có hai kiểu bỏ dấu: "hoà"/"hòa", "khoẻ"/"khỏe", "thuỷ"/"thủy".
# Dữ liệu (kể cả InfoRe) trộn cả hai -> mô hình phải học 2 cách viết cho 1 âm.
# Ta quy về một kiểu: dấu đặt trên nguyên âm đầu khi vần mở (hòa, khỏe, thủy).
# Chỉ áp dụng khi cặp nguyên âm đứng cuối âm tiết ("hoàng" giữ nguyên)
# và không đứng sau "q" ("quý" giữ nguyên vì "qu" là phụ âm đầu).
_TONE_STYLE = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}
_TONE_RE = re.compile(r"(?<!q)(" + "|".join(map(re.escape, _TONE_STYLE)) + r")(?!\w)")

ABBREVIATIONS = {
    "tp.hcm": "thành phố hồ chí minh",
    "tp hcm": "thành phố hồ chí minh",
    "tphcm": "thành phố hồ chí minh",
    "hcm": "hồ chí minh",
    "tp.": "thành phố",
    "hn": "hà nội",
    "ubnd": "ủy ban nhân dân",
    "hđnd": "hội đồng nhân dân",
    "thpt": "trung học phổ thông",
    "thcs": "trung học cơ sở",
    "đh": "đại học",
    "gs.": "giáo sư",
    "ts.": "tiến sĩ",
    "pgs.": "phó giáo sư",
    "ths.": "thạc sĩ",
    "bs.": "bác sĩ",
    "vn": "việt nam",
    "vnd": "đồng",
    "vnđ": "đồng",
    "usd": "đô la mỹ",
    "v.v.": "vân vân",
    "v.v": "vân vân",
}
_ABBR_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)

UNITS = {
    "km": "ki lô mét", "m": "mét", "cm": "xen ti mét", "mm": "mi li mét",
    "kg": "ki lô gam", "g": "gam", "mg": "mi li gam",
    "l": "lít", "ml": "mi li lít",
    "h": "giờ", "km/h": "ki lô mét trên giờ",
    "%": "phần trăm", "đ": "đồng", "₫": "đồng", "$": "đô la",
    "gb": "gi ga bai", "mb": "mê ga bai", "kw": "ki lô oát", "w": "oát",
}

_DATE_RE = re.compile(r"(?:\bngày\s+)?(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(r"(?:\btháng\s+)?(?<!\d)(\d{1,2})/(\d{4})(?!\d)", re.IGNORECASE)
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2})(?:h|:)(\d{2})(?:'|p)?(?!\d)")
_HOUR_RE = re.compile(r"(?<!\d)(\d{1,2})h(?![\w])")
_PHONE_RE = re.compile(r"(?<!\d)(0\d{9,10})(?!\d)")
# Số kiểu VN: 1.000.000 (dấu chấm ngăn nghìn), 3,5 (dấu phẩy thập phân).
_NUMBER_RE = re.compile(r"(?<![\d,.])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?")
_UNIT_RE = re.compile(
    r"(?<![\w])(\d[\d.,]*)\s*(" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True)) + r")(?![\w])",
    re.IGNORECASE,
)
_SYMBOLS = {"&": " và ", "+": " cộng ", "=": " bằng ", "@": " a còng "}


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_tone_style(text: str) -> str:
    return _TONE_RE.sub(lambda m: _TONE_STYLE[m.group(0)], text)


def _read_number_token(int_part: str, frac: str | None) -> str:
    words = number_to_words(int(int_part.replace(".", "")))
    if frac:
        # 3,05 -> ba phẩy không năm ; 3,5 -> ba phẩy năm ; 3,25 -> ba phẩy hai mươi lăm
        frac_words = digits_to_words(frac) if frac.startswith("0") else number_to_words(int(frac))
        words += " phẩy " + frac_words
    return words


def _expand_numbers(text: str) -> str:
    def date(m: re.Match) -> str:
        d, mo, y = (int(x) for x in m.groups())
        return f"ngày {number_to_words(d)} tháng {number_to_words(mo)} năm {number_to_words(y)}"

    def month_year(m: re.Match) -> str:
        mo, y = (int(x) for x in m.groups())
        return f"tháng {number_to_words(mo)} năm {number_to_words(y)}"

    def time(m: re.Match) -> str:
        h, mi = (int(x) for x in m.groups())
        return f"{number_to_words(h)} giờ" + ("" if mi == 0 else f" {number_to_words(mi)} phút")

    def unit(m: re.Match) -> str:
        num, u = m.groups()
        return f"{num} {UNITS[u.lower()]}"

    text = _DATE_RE.sub(date, text)
    text = _MONTH_YEAR_RE.sub(month_year, text)
    text = _TIME_RE.sub(time, text)
    text = _HOUR_RE.sub(lambda m: f"{number_to_words(int(m.group(1)))} giờ", text)
    text = _PHONE_RE.sub(lambda m: digits_to_words(m.group(1)), text)
    text = _UNIT_RE.sub(unit, text)
    text = _NUMBER_RE.sub(lambda m: _read_number_token(m.group(1), m.group(2)), text)
    return text


def normalize(text: str, keep_punct: bool = False) -> str:
    """Chuẩn hoá về spoken form.

    keep_punct=True giữ lại `, . ? !` (TTS dùng làm điểm ngắt nghỉ);
    False cho nhãn STT.
    """
    text = nfc(text)
    text = _ABBR_RE.sub(lambda m: " " + ABBREVIATIONS[m.group(1).lower()] + " ", text)
    text = _expand_numbers(text)
    for sym, word in _SYMBOLS.items():
        text = text.replace(sym, word)
    text = normalize_tone_style(text.lower())

    if keep_punct:
        text = re.sub(r"[;:]", ",", text)
        text = re.sub(r"[^\w\s,.?!]", " ", text)
        text = re.sub(r"\s*([,.?!])\s*", r" \1 ", text)
    else:
        text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()
