# Roadmap

## Giai đoạn 1 — STT streaming (đang làm)
- [x] Chuẩn hoá văn bản tiếng Việt + bảng ký tự
- [x] Tiền xử lý parquet → log-mel memmap, CMVN
- [x] Streaming Conformer-CTC (dynamic chunk), train có resume
- [x] Export ONNX + int8, runtime numpy/onnxruntime, benchmark CPU
- [x] Pipeline Kaggle: kernel CPU tiền xử lý → kernel GPU (2× T4, DDP) train nhiều phiên 12 giờ
- [ ] Train `stt_small` trên VLSP2020 92h + InfoRe 25h (đang chạy), báo cáo WER (full / streaming 640 ms / 320 ms)
- [ ] Làm sạch dữ liệu kỹ hơn (lọc theo CER khi chạy lại ASR, SNR); thêm dữ liệu
- [ ] Decode beam search + n-gram LM (KenLM), so sánh WER và tốc độ
- [ ] Thử RNN-T / Zipformer-style encoder; đo trên Raspberry Pi / ARM

## Giai đoạn 2 — TTS chạy trên CPU
Đã chọn: tầng nền Matcha-TTS + Vocos (train từ đầu), tầng mục tiêu "mini-VieNeu" (LM ~100M + codec).
- [x] Pipeline dữ liệu TTS: VAD cắt im lặng, -20 LUFS, SNR theo VAD, lọc nhãn bằng PhoWhisper (CER)
- [x] Matcha-TTS + Vocos, export ONNX, runtime numpy, RTF CPU ~0,08
- [ ] Train Matcha trên InfoRe1 (Kaggle), đánh giá CER round-trip (ASR khác ASR lọc), UTMOS, nghe thử
- [ ] Int8 cho decoder/vocoder, đo trên ARM
- [ ] Đa giọng (speaker embedding), dữ liệu lớn hơn (NC, bản nghiên cứu)
- [ ] mini-VieNeu: mã hoá codec offline, LM nhỏ + depth decoder, streaming

## Giai đoạn 3 — Voice Agent realtime (xem docs/AGENT.md)
- [x] VAD + endpointing, streaming STT → LLM → TTS theo đoạn qua WebSocket, web client
- [x] Barge-in, đo độ trễ từng chặng; test giọng thật + Chromium headless
- [ ] Turn detector ngữ nghĩa cho tiếng Việt (giảm 640 ms chờ im lặng)
- [ ] WebRTC transport (LiveKit/Pipecat), nhiều phiên đồng thời + batching GPU
