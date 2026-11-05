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

Kết quả sau 2 giờ (1 GPU / cấu hình, batch 40.000 frame, dev 696 câu):

| Bước | B offline CER / WER | B streaming 640 ms CER / WER | C offline CER / WER | C streaming CER / WER |
|---|---|---|---|---|
| 1.000 | 0,743 / 0,997 | 0,775 / 0,995 | 0,753 / 1,000 | 0,767 / 0,995 |
| 4.000 | 0,286 / 0,527 | 0,445 / 0,753 | 0,318 / 0,583 | 0,431 / 0,769 |
| 8.000 | 0,179 / 0,333 | 0,199 / 0,373 | 0,187 / 0,354 | 0,208 / 0,393 |
| 12.000 | 0,140 / 0,266 | 0,157 / 0,301 | 0,152 / 0,284 | 0,167 / 0,315 |
| cuối | **0,106 / 0,201** (17.570) | **0,126 / 0,239** | 0,123 / 0,240 (16.739) | 0,145 / 0,290 |

B (InterCTC) thắng C ở mọi mốc → chọn B. Tỉ lệ blank ổn định ~0,22 (đã thoát plateau). `clipped_frac` = 1,0
(gradient luôn bị cắt về 5 vì loss cộng theo câu), vẫn hội tụ tốt nên giữ nguyên.

Lưu ý: thí nghiệm này lỡ chạy **trùng 2 bản** (bản tự động + bản đẩy tay), chiếm cả 2 suất GPU trong 2 giờ.

### Lượt 2 chính thức: `stt-small-v2` (configs/stt_small_v2.yaml = cấu hình B), từ 2026-11-05 ~10:15 UTC

- Khởi tạo từ checkpoint B ở bước 17.570 (`--init-kernel nhmanhaidev/vi-speech-exp-plateau --init-run stt_b`),
  DDP 2× T4, thêm EMA (decay 0,9999), 11 giờ.

| Bước | Offline CER / WER | Streaming CER / WER | Streaming + EMA CER / WER |
|---|---|---|---|
| 26.000 | 0,087 / 0,164 | 0,108 / 0,201 | 0,091 / 0,170 |
| 28.000 | 0,080 / 0,151 | 0,102 / 0,192 | 0,085 / 0,161 |
| 30.000 | 0,079 / 0,146 | 0,098 / 0,181 | 0,086 / 0,161 |

EMA giảm WER streaming ~2 điểm. Dev có lẫn InfoRe (1 giọng, dễ) nên con số chính thức sẽ đo trên FLEURS-vi và VIVOS.

## TTS

### Lượt 1: `tts-matcha` (configs/tts_matcha.yaml), 04:59 → bị huỷ ở bước 2.550

- Matcha-TTS 18,5M + Vocos 24 kHz (pretrained, không train), InfoRe 13.044 câu sau lọc; DDP 2× T4, 0,52 s/bước.
- Loss giảm đều (bước 2.550: dur 0,56; prior 1,08; flow matching 0,49). Bị huỷ → mất output (Kaggle không lưu khi huỷ).

### Lượt 2: `tts-matcha` với điều khiển ngữ điệu (configs/tts_matcha_prosody.yaml)

- Thêm vector ngữ điệu 3 chiều (cao độ, biên độ cao độ, tốc độ), dropout điều kiện 0,2.
- Phiên 4 giờ, resume nối tiếp (tự tham chiếu output phiên trước: đã kiểm, phiên 2 resume đúng từ bước 26.603).
- Mỗi 3.000 bước lưu mẫu 3 câu × 3 kiểu (bình thường / hào hứng / trầm). Câu mẫu không có trong dữ liệu train.

Phiên 1 (07:37–11:40 UTC, 26.600 bước, 0,53 s/bước): loss cuối dur 0,39; prior 1,01; flow matching 0,36.
Chấm mẫu bằng PhoWhisper-small (3 câu, nên số dao động):

| Bước | CER (bình thường) |
|---|---|
| 3.000 | 0,388 |
| 12.000 | 0,123 |
| 24.000 | 0,150 (hào hứng 0,081; trầm 0,084) |

Điều khiển ngữ điệu có tác dụng đo được ở bước 24.000: cao độ trung vị bình thường 14,3 / hào hứng 15,7 /
trầm 12,9 semitone (±1,4 st), tổng thời lượng 18,2 / 17,1 / 19,3 s (±6%).

## Chạy thử trên CPU (không phải train thật)

| Mục đích | Cấu hình | Kết quả |
|---|---|---|
| Kiểm pipeline STT học được | 1M tham số, 1,5 h InfoRe, 1.500 bước | CER dev 0,23; ONNX offline/streaming/int8 khớp nhau |
| Kiểm pipeline TTS | 3,2M tham số, 1,3 h, 3.000 bước | Học được độ dài câu, chưa ra chữ (CER ~0,9). Chỉ để kiểm code |

## 2026-11-05 chiều: dừng STT v2 và TTS phiên 2, chuyển hướng

- Anh Mạnh dừng cả 2 lượt (khoảng 15:20 UTC). STT v2 dừng ở khoảng bước 31.000 (dev: offline WER 14,6%,
  streaming + EMA 16,1% ở bước 30.000). TTS Matcha phiên 2 dừng ở khoảng bước 37.000; Kaggle không lưu output khi huỷ.
- Lý do: tạm gác STT (đã có nhiều model STT tiếng Việt nhỏ đủ tốt), tập trung TTS với yêu cầu mới: giọng tự nhiên,
  clone từ ~10 s audio, thẻ cảm xúc / âm thanh không lời kiểu ElevenLabs. Train từ đầu trên quota Kaggle free không
  đạt được mức này → chuyển sang fine-tune model mở đã clone được (VieNeu v3 Turbo, Apache-2.0).
- Quota GPU tuần này còn khoảng 10 giờ; không chạy gì thêm trên Kaggle khi chưa được duyệt.
- Mốc nền VieNeu v3 Turbo gốc (CPU 4 vCPU Xeon có VNNI): clone 4 giọng từ 10 s, CER câu thường 0–4%
  (PhoWhisper-small); RTF fp32 ~1,0, int8 ~0,75 (không tính lần mã hoá audio mẫu đầu tiên 5–8 s).
