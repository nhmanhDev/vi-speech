"""Đo tốc độ TTS ONNX trên CPU: RTF từng chặng (encoder, decoder × n bước, vocoder).

    python -m vispeech.tts.bench --model exported/tts_matcha --threads 2 --steps 4
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .onnx_tts import OnnxTTS, istft

TEXTS = [
    "Xin chào, tôi là trợ lý ảo của bạn.",
    "Hôm nay, ngày 5/11/2026, trời Hà Nội se lạnh và có mưa phùn nhẹ vào buổi sáng sớm.",
    "Quý khách vui lòng giữ máy, cuộc gọi của quý khách sẽ được chuyển tới tổng đài viên trong giây lát.",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args(argv)
    tts = OnnxTTS(args.model, threads=args.threads)
    tts(TEXTS[0], n_steps=args.steps)  # warmup
    tot = {"acoustic": 0.0, "vocoder": 0.0, "audio": 0.0}
    for text in TEXTS:
        t0 = time.perf_counter()
        mel = tts.mel(text, n_steps=args.steps).astype(np.float32)
        t1 = time.perf_counter()
        mag, ph = tts.voc.run(None, {"mel": mel})
        wav = istft(mag[0], ph[0])
        t2 = time.perf_counter()
        dur = len(wav) / 24000
        tot["acoustic"] += t1 - t0
        tot["vocoder"] += t2 - t1
        tot["audio"] += dur
        print(f"{dur:5.2f}s audio | acoustic {1000 * (t1 - t0):6.1f} ms | vocoder {1000 * (t2 - t1):6.1f} ms | RTF {(t2 - t0) / dur:.3f}")
    a = tot["audio"]
    print(f"threads={args.threads} steps={args.steps}: RTF acoustic={tot['acoustic'] / a:.3f} vocoder={tot['vocoder'] / a:.3f} "
          f"total={(tot['acoustic'] + tot['vocoder']) / a:.3f}")


if __name__ == "__main__":
    main()
