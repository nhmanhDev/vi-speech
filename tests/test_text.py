import pytest

from vispeech.text import Vocab, normalize, number_to_words, split_tone


@pytest.mark.parametrize(
    "n, words",
    [
        (0, "không"),
        (10, "mười"),
        (15, "mười lăm"),
        (21, "hai mươi mốt"),
        (24, "hai mươi tư"),
        (25, "hai mươi lăm"),
        (14, "mười bốn"),
        (11, "mười một"),
        (105, "một trăm linh năm"),
        (110, "một trăm mười"),
        (1005, "một nghìn không trăm linh năm"),
        (2024, "hai nghìn không trăm hai mươi tư"),
        (1_000_000, "một triệu"),
        (1_500_000, "một triệu năm trăm nghìn"),
        (2_000_000_000, "hai tỷ"),
        (-3, "âm ba"),
    ],
)
def test_numbers(n, words):
    assert number_to_words(n) == words


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ngày 12/5/2024", "ngày mười hai tháng năm năm hai nghìn không trăm hai mươi tư"),
        ("lúc 10h30", "lúc mười giờ ba mươi phút"),
        ("giá 1.500.000đ", "giá một triệu năm trăm nghìn đồng"),
        ("tăng 3,5%", "tăng ba phẩy năm phần trăm"),
        ("UBND TP.HCM", "ủy ban nhân dân thành phố hồ chí minh"),
        ("gọi 0912345678", "gọi không chín một hai ba bốn năm sáu bảy tám"),
        ("60km/h", "sáu mươi ki lô mét trên giờ"),
        # chuẩn hoá kiểu bỏ dấu, nhưng không phá "hoàng", "quý"
        ("khoẻ hoà thuỷ hoàng quý", "khỏe hòa thủy hoàng quý"),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_normalize_keep_punct():
    assert normalize("Xin chào, bạn khoẻ không?", keep_punct=True) == "xin chào , bạn khỏe không ?"


def test_normalize_nfd_input():
    import unicodedata

    assert normalize(unicodedata.normalize("NFD", "Tiếng Việt")) == "tiếng việt"


def test_vocab_roundtrip_and_ctc():
    v = Vocab()
    text = "người việt nam đẹp quá"
    ids = v.encode(text)
    assert v.unk_id not in ids
    assert v.decode(ids) == text
    a = v.stoi["a"]
    assert v.ctc_collapse([0, a, a, 0, a]) == "aa"


def test_split_tone():
    assert split_tone("người") == ("ngươi", 1)
    assert split_tone("việt") == ("viêt", 5)
    assert split_tone("ma") == ("ma", 0)
