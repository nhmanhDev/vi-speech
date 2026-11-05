# Roadmap

## Giai đoạn 1 — STT streaming (đang làm)
- [x] Chuẩn hoá văn bản tiếng Việt + bảng ký tự
- [x] Tiền xử lý parquet → log-mel memmap, CMVN
- [x] Streaming Conformer-CTC (dynamic chunk), train có resume
- [x] Export ONNX + int8, runtime numpy/onnxruntime, benchmark CPU
- [ ] Pipeline Kaggle: kernel CPU tiền xử lý → kernel GPU train nhiều phiên 12 giờ
- [ ] Train `stt_small` trên VLSP2020 100h, báo cáo WER (full / streaming 640 ms / 320 ms)
- [ ] Làm sạch dữ liệu kỹ hơn (lọc theo CER khi chạy lại ASR, SNR); thêm dữ liệu
- [ ] Decode beam search + n-gram LM (KenLM), so sánh WER và tốc độ
- [ ] Thử RNN-T / Zipformer-style encoder; đo trên Raspberry Pi / ARM

## Giai đoạn 2 — TTS chạy trên CPU
Chờ kết quả nghiên cứu (VieNeu-TTS, F5-TTS, VITS/Piper, Matcha…) để chọn kiến trúc.
- [ ] Pipeline dữ liệu TTS: VAD cắt câu, chuẩn hoá loudness, lọc bằng ASR, chuẩn hoá văn bản
- [ ] Model + vocoder, export ONNX, đo RTF trên CPU, MOS/UTMOS, WER khi nhận dạng lại

## Giai đoạn 3 — Voice Agent realtime
- [ ] VAD + endpointing, streaming STT → LLM → streaming TTS qua WebSocket
- [ ] Barge-in (người dùng nói chen thì ngắt TTS), đo độ trễ từng chặng
