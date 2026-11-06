# Kết quả đo (số liệu thật, không phải ước lượng)

File nhỏ sinh ra trong quá trình làm, giữ lại để tái lập và để không mất khi máy cloud bị thu hồi.
Mô tả và diễn giải nằm ở `docs/` (DATA.md, TRAINING.md, TAG_PILOT.md, AGENT.md).

| Thư mục | Nội dung |
|---|---|
| `stt/` | Nhật ký train thí nghiệm B (InterCTC) và C: loss, CER/WER dev theo bước, blank_ratio |
| `tts/` | Nhật ký train Matcha-TTS có điều khiển ngữ điệu (phiên 1) và điểm CER / cao độ / thời lượng của mẫu giọng |
| `data/` | Báo cáo chất lượng dữ liệu: VLSP, InfoRe (STT), InfoRe (TTS: QC + kiểm nhãn bằng PhoWhisper), so sánh bộ ứng viên TTS |
| `tag_pilot/` | Phát hiện âm thanh không lời (AST) trong YODAS2-Sidon và LEMAS |
| `vieneu_baseline/` | Đo VieNeu v3 Turbo gốc trên CPU: RTF, thời lượng, từng giọng |

**Không có trong repo (quá lớn, nằm trên Kaggle hoặc chỉ ở máy cloud tạm):** audio, log-mel, checkpoint model.
Checkpoint STT B (thí nghiệm) nằm trong output của kernel Kaggle `nhmanhaidev/vi-speech-exp-plateau`.
Lượt STT v2 và TTS phiên 2 bị huỷ nên **không có checkpoint**; TTS phiên 1 (bước 26.600) chỉ còn bản trên máy cloud.
