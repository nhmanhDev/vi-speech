# Nhật ký train

Ghi lại **mọi lượt train trên GPU**: cấu hình, kết quả, vì sao dừng, rút ra gì. Nguyên tắc của dự án:
**train trên GPU (Kaggle 2× T4), model sau train phải chạy tốt trên CPU** (ONNX, đo RTF/độ trễ trên CPU).
CPU chỉ dùng cho: test tự động, chạy thử vài bước để bắt lỗi trước khi tốn quota GPU, đo tốc độ suy luận.

## Kaggle: các giới hạn đã gặp

| Giới hạn | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Tối đa **2 phiên GPU** chạy cùng lúc | Phiên thứ 3 bị từ chối ("Maximum batch GPU session count of 2 reached") | Script tự thử lại khi có suất trống |
| Phiên tối đa 12 giờ, quota ~30 giờ GPU/tuần | | Dừng an toàn ở `--max-hours`, lưu `last.pt`, phiên sau resume |
| **Output chỉ có khi phiên kết thúc**; phiên bị huỷ thì **mất output** | Không nghe được mẫu giữa chừng; huỷ là mất công | Chia phiên ngắn (TTS 4 giờ), lưu mẫu giọng định kỳ |
| `/kaggle/input` là ổ mạng, đọc ngẫu nhiên chậm | Train STT 5,9 s/bước thay vì 0,42 | Copy dữ liệu sang `/tmp` (ổ local) trước khi train |
| API không huỷ được phiên (403) | Phải huỷ trên web | Trang notebook → View versions → ⋮ → Cancel, hoặc Active Events → Stop Session |

Lệnh điều khiển: `scripts/kaggle_job.py` (`push-code`, `prep`, `tts-prep`, `train`, `tts-train`, `stt-exp`, `status`, `output`).

## STT

### Lượt 1: `stt-small` (configs/stt_small.yaml), 2026-11-05 04:07 → huỷ ~05:20 UTC

- Model: Conformer-CTC 20,1M (12 lớp, d=256), char-level, dynamic chunk; dữ liệu VLSP 92,2 h + InfoRe 24,9 h; DDP 2× T4.
- LR 1,5e-3, warmup 5.000; SpecAugment 2 mặt nạ tần + **10** mặt nạ thời gian (mỗi cái ≤ 5%);
  chunk ngẫu nhiên 1–25 frame, left_chunks 0–8 (có cả trường hợp chunk 1 frame, không ngữ cảnh trái).
- Tốc độ: 5,9 s/bước lúc đầu (nghẽn I/O, xem trên) → 0,42 s/bước khi dữ liệu đã nằm trong RAM.

| Bước | CER dev | loss/ký tự |
|---|---|---|
| 2.000 | 0,83 | 2,94 |
| 4.000 | 0,90 | 3,18 |
| 6.000 | 0,98 | 3,59 |

**Phân kỳ**: loss tăng đúng lúc LR tăng theo warmup. Nghi nguyên nhân: LR cao + SpecAugment quá mạnh cho ~100 giờ
(cấu hình "10 mặt nạ" là của dữ liệu hàng nghìn giờ) + chunk 1 frame không ngữ cảnh làm khó tối ưu. Đã huỷ để thử cấu hình sửa.

### Thí nghiệm `exp-plateau` (2 giờ, mỗi GPU 1 cấu hình, cùng dữ liệu)

Thay đổi chung so với lượt 1: LR 1e-3; SpecAugment 2+2 mặt nạ (mặt nạ thời gian ≤ 40 frame); 4.000 bước đầu chỉ
full context rồi mới bật chunk (curriculum); left_chunks ≥ 1. Thêm log `blank_ratio`, `clipped_frac`.

| Cấu hình | Khác biệt | CER dev @1.000 |
|---|---|---|
| B (`configs/exp/stt_b.yaml`) | + InterCTC lớp 6 (trọng số 0,3) | 0,743 |
| C (`configs/exp/stt_c.yaml`) | không InterCTC | 0,753 |

Cả hai ở bước 1.000 đã tốt hơn lượt 1 ở bước 2.000 (0,83). Kết quả đầy đủ cập nhật khi xong.

Lưu ý: thí nghiệm này lỡ chạy **trùng 2 bản** (bản tự động + bản đẩy tay), chiếm cả 2 suất GPU trong 2 giờ.

## TTS

### Lượt 1: `tts-matcha` (configs/tts_matcha.yaml), 04:59 → bị huỷ ở bước 2.550

- Matcha-TTS 18,5M + Vocos 24 kHz (pretrained, không train), InfoRe 13.044 câu sau lọc; DDP 2× T4, 0,52 s/bước.
- Loss giảm đều (bước 2.550: dur 0,56; prior 1,08; flow matching 0,49). Bị huỷ → mất output (Kaggle không lưu khi huỷ).

### Lượt 2: `tts-matcha` với điều khiển ngữ điệu (configs/tts_matcha_prosody.yaml)

- Thêm vector ngữ điệu 3 chiều (cao độ, biên độ cao độ, tốc độ), dropout điều kiện 0,2.
- Phiên 4 giờ, resume nối tiếp; mỗi 3.000 bước lưu mẫu giọng 3 câu × 3 kiểu (bình thường / hào hứng / trầm).
- Trạng thái: chờ suất GPU (tự chạy khi thí nghiệm STT xong).

## Chạy thử trên CPU (không phải train thật)

| Mục đích | Cấu hình | Kết quả |
|---|---|---|
| Kiểm pipeline STT học được | 1M tham số, 1,5 h InfoRe, 1.500 bước | CER dev 0,23; ONNX offline/streaming/int8 khớp nhau |
| Kiểm pipeline TTS | 3,2M tham số, 1,3 h, 3.000 bước | Học được độ dài câu, chưa ra chữ (CER ~0,9). Chỉ để kiểm code |
