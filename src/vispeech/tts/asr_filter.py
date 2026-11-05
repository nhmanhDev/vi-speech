"""Kiểm nhãn bằng ASR: nhận dạng lại từng câu, tính CER giữa nhãn và kết quả ASR.

Câu có CER cao thường là: nhãn sai/thiếu chữ, đọc vấp, cắt câu lệch, có tiếng khác chen vào.
TTS học thẳng các lỗi này (đọc sót chữ, lặp chữ), nên lọc chặt (mặc định giữ CER <= 5%).

Chạy trên GPU (Kaggle T4): PhoWhisper-medium ~25 h audio mất cỡ 1 giờ.

    python -m vispeech.tts.asr_filter --data data/tts/infore --model vinai/PhoWhisper-medium

Ghi `index.asr.jsonl` (thêm trường `asr`, `cer`). Lưu ý khi đánh giá TTS về sau: dùng một ASR
*khác* ASR đã dùng để lọc, nếu không sẽ tự chấm điểm cho mình.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..audio import resample
from ..metrics import edit_distance
from ..text.normalize import normalize


def cer(ref: str, hyp: str) -> float:
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    return edit_distance(r, h) / max(len(r), 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="thư mục output của vispeech.tts.prepare")
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    wav_all = np.memmap(os.path.join(args.data, "wav.i16"), dtype=np.int16, mode="r", shape=(meta["wav_samples"],))
    with open(os.path.join(args.data, "index.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    out_path = os.path.join(args.data, "index.asr.jsonl")
    cers = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for i in range(0, len(rows), args.batch):
            batch = rows[i : i + args.batch]
            wavs = [resample(wav_all[r["wav_offset"] : r["wav_offset"] + r["wav_len"]].astype(np.float32) / 32768, meta["sample_rate"], 16000)
                    for r in batch]
            feats = proc(wavs, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
            with torch.no_grad():
                ids = model.generate(feats, language="vi", task="transcribe", max_new_tokens=200, num_beams=1)
            hyps = proc.batch_decode(ids, skip_special_tokens=True)
            for r, h in zip(batch, hyps):
                r["asr"] = normalize(h)
                r["cer"] = round(cer(normalize(r["text"]), r["asr"]), 4)
                cers.append(r["cer"])
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{i + len(batch)}/{len(rows)}] mean CER={np.mean(cers):.4f}", flush=True)

    c = np.array(cers)
    summary = {"model": args.model, "n": len(c), "mean_cer": float(c.mean()),
               **{f"keep@{t}": float((c <= t).mean()) for t in (0.0, 0.02, 0.05, 0.1)}}
    with open(os.path.join(args.data, "asr_report.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
