# vi-speech

Nhận dạng giọng nói (STT) và tổng hợp giọng nói (TTS) tiếng Việt **xây từ đầu**, ưu tiên
**chạy realtime trên CPU / edge device**. Train trên GPU (Kaggle T4/P100), deploy bằng ONNX Runtime int8.

> Trạng thái: STT streaming Conformer-CTC đã chạy đầu-cuối (dữ liệu → train → ONNX int8 → streaming
> trên CPU). TTS và Voice Agent: đang nghiên cứu, xem [docs/ROADMAP.md](docs/ROADMAP.md).

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

## Dữ liệu

| Dataset | Giờ | License | Dùng cho |
|---|---|---|---|
| [doof-ferb/vlsp2020_vinai_100h](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h) | ~100 | CC-BY-4.0 | STT |
| [doof-ferb/infore1_25hours](https://huggingface.co/datasets/doof-ferb/infore1_25hours) | 25 (1 giọng, 44,1 kHz) | CC-BY-4.0 | TTS, STT |

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

## Cấu trúc

```
src/vispeech/
  text/      chuẩn hoá văn bản (số, ngày, giờ, đơn vị, viết tắt, kiểu bỏ dấu), bảng ký tự, tách thanh
  audio.py   đọc audio, log-mel (offline + streaming)
  data/      tiền xử lý parquet -> memmap, dataset, dynamic batching
  stt/       model, train, export ONNX, runtime onnxruntime, benchmark
configs/     stt_small / stt_tiny / stt_debug
tests/       pytest (streaming == offline, padding, overfit, chuẩn hoá văn bản)
```

## Test

```bash
pytest -q
```
