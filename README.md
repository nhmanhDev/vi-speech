# vi-speech

Nhận dạng giọng nói (STT) và tổng hợp giọng nói (TTS) tiếng Việt **xây từ đầu**, ưu tiên
**chạy realtime trên CPU / edge device**. Train trên GPU (Kaggle T4/P100), deploy bằng ONNX Runtime int8.

> Trạng thái: STT streaming Conformer-CTC và TTS Matcha-TTS + Vocos đều đã chạy đầu-cuối
> (dữ liệu → train → ONNX → CPU). Đang train trên Kaggle (2× T4). Voice Agent: xem [docs/ROADMAP.md](docs/ROADMAP.md).
> Dữ liệu và các quyết định xử lý: [docs/DATA.md](docs/DATA.md).

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/DATA.md](docs/DATA.md) | Dùng bộ dữ liệu nào, lấy ở đâu, từng bước xử lý và lý do, còn lại bao nhiêu sau mỗi bước |
| [docs/TRAINING.md](docs/TRAINING.md) | Nhật ký mọi lượt train trên GPU: cấu hình, kết quả, vì sao dừng, giới hạn Kaggle |
| [docs/AGENT.md](docs/AGENT.md) | Voice agent: kiến trúc, độ trễ đo được, barge-in, quyết định thiết kế |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Việc đã xong / sắp làm |

## Kiến trúc STT

```
mic 16 kHz ──► log-mel 80 (numpy, 10 ms) ──► Conv2d subsampling x4 (40 ms/frame)
           ──► 12 × Conformer block [FFN½ · RoPE self-attn (chunk mask) · causal depthwise conv · FFN½]
           ──► Linear → CTC (96 ký tự tiếng Việt) ──► greedy decode
```

| Lựa chọn | Lý do |
|---|---|
| CTC, char-level (96 token) | Đơn giản, decode nhanh nhất trên CPU; chính tả tiếng Việt gần với cách đọc nên không cần BPE |
| Dynamic chunk training (WeNet U2) | Một model chạy cả offline lẫn streaming, chọn độ trễ lúc deploy (40 ms – 1 s/chunk) |
| Causal conv + LayerNorm, RoPE | Không phụ thuộc tương lai, KV-cache cố định → ONNX tĩnh, bộ nhớ không tăng theo thời gian |
| Subsampling không padding | Streaming cho kết quả **trùng khớp** chạy cả câu (đã kiểm tra trong test) |
| Feature extractor bằng numpy | Edge chỉ cần `numpy + onnxruntime`; train và inference dùng chung code |

### Tốc độ trên CPU (1 luồng, x86 server, int8, chunk 640 ms)

| Model | Tham số | ONNX int8 | Thời gian tính mỗi chunk (p50) | RTF streaming |
|---|---|---|---|---|
| `stt_small` | 20,1 M | 29,6 MB | 12,6 ms | 0,023 |
| `stt_tiny` | 6,4 M | 9,8 MB | 6,1 ms | 0,012 |

*(Đo bằng `vispeech.stt.bench`; WER sẽ cập nhật sau khi train trên VLSP2020.)*

## Kiến trúc TTS

```
văn bản ─► chuẩn hoá (số, ngày, viết tắt) ─► ký tự + dấu câu (chèn blank)
        ─► Text encoder (6 lớp Transformer, RoPE) ─► μ + duration predictor
        ─► length regulator ─► Decoder U-Net 1D, flow matching (2–10 bước Euler) ─► mel 100 bin
        ─► Vocos 24 kHz (ISTFT bằng numpy) ─► sóng âm
```

Matcha-TTS (ICASSP 2024): căn chỉnh chữ–âm tự học bằng MAS (không cần forced aligner), decoder flow matching
nên chỉ cần vài bước suy luận. 18,5M tham số + Vocos 13,5M.

| Cấu hình (CPU x86, ONNX fp32) | RTF |
|---|---|
| 2 luồng, 4 bước | 0,076 |
| 2 luồng, 2 bước | 0,056 |
| 1 luồng, 4 bước | 0,107 |

*(Đo bằng `vispeech.tts.bench`, trọng số ngẫu nhiên: chỉ nói lên tốc độ. Tham khảo: VieNeu-TTS v3 Turbo tự báo
RTF int8 ~0,35 trên i5 gen 12.)*

## Dữ liệu

| Dataset | Dùng cho | Sau xử lý | License |
|---|---|---|---|
| [doof-ferb/vlsp2020_vinai_100h](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h) | STT | 54.497 câu, 92,2 h | CC-BY-4.0 |
| [doof-ferb/infore1_25hours](https://huggingface.co/datasets/doof-ferb/infore1_25hours) | STT, TTS (1 giọng nữ) | STT 24,9 h; TTS 13.044 câu ≈ 20,7 h | CC-BY-4.0 |
| FLEURS-vi test, VIVOS test | chỉ đánh giá STT | 857 + 760 câu | CC-BY-4.0 / CC-BY-NC-SA-4.0 |

Chi tiết từng bước xử lý: [docs/DATA.md](docs/DATA.md).

Pipeline TTS: VAD cắt im lặng → -20 LUFS → 24 kHz → mel khớp Vocos → SNR (theo VAD) → kiểm nhãn bằng PhoWhisper (CER ≤ 5%).

## Cách dùng

```bash
pip install -e ".[dev,export]"

# 1. Tiền xử lý: parquet HF -> log-mel fp16 + index (chuẩn hoá văn bản, lọc độ dài, CMVN)
python -m vispeech.data.prepare --parquet 'data/vlsp/*.parquet' --out data/prep/vlsp

# 2. Train (tự resume từ last.pt; --max-hours để dừng an toàn trên Kaggle)
python -m vispeech.stt.train --config configs/stt_small.yaml --data data/prep/vlsp --out runs/stt_small

# 3. Export ONNX + int8, rồi đo tốc độ
python -m vispeech.stt.export --ckpt runs/stt_small/best.pt --out exported/stt_small --chunk 16 --left-chunks 4
python -m vispeech.stt.bench --model exported/stt_small --wav sample.wav
```

Dùng trên edge (không cần torch):

```python
from vispeech.stt.onnx_asr import StreamingASR
asr = StreamingASR("exported/stt_small", int8=True)
for pcm in mic_chunks():      # float32 16 kHz, gói 20 ms
    print(asr.accept(pcm))    # partial transcript
print(asr.finish())
```

TTS:

```bash
python -m vispeech.tts.prepare --parquet 'data/infore/*.parquet' --out data/tts/infore --speaker infore
python -m vispeech.tts.asr_filter --data data/tts/infore          # GPU
python -m vispeech.tts.train --data data/tts/infore --out runs/tts_matcha
python -m vispeech.tts.export --ckpt runs/tts_matcha/last.pt --out exported/tts_matcha
python -m vispeech.tts.bench --model exported/tts_matcha
```

Kaggle (cần `KAGGLE_API_TOKEN`): `scripts/kaggle_job.py push-code | prep | tts-prep | train | status | output`.

## Cấu trúc

```
src/vispeech/
  text/      chuẩn hoá văn bản (số, ngày, giờ, đơn vị, viết tắt, kiểu bỏ dấu), bảng ký tự, tách thanh
  audio.py   đọc audio, log-mel (offline + streaming)
  data/      tiền xử lý parquet -> memmap, dataset, dynamic batching
  stt/       model, train, export ONNX, runtime onnxruntime, benchmark
  tts/       prepare + asr_filter (dữ liệu), Matcha-TTS, train, export ONNX + Vocos, runtime, benchmark
configs/     stt_small / stt_tiny / stt_debug
tests/       pytest (streaming == offline, padding, overfit, chuẩn hoá văn bản)
```

## Test

```bash
pytest -q
```
