"""Đo tốc độ trên CPU: RTF (real-time factor) và độ trễ mỗi chunk.

    python -m vispeech.stt.bench --model exported/stt_small [--wav test.wav] [--threads 1]

RTF = thời gian xử lý / thời lượng audio. RTF < 1 là nhanh hơn thời gian thực;
với streaming, cái quan trọng hơn là độ trễ mỗi chunk phải << thời lượng chunk.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ..audio import load_audio
from .onnx_asr import OfflineASR, StreamingASR


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav")
    ap.add_argument("--seconds", type=float, default=10.0, help="độ dài audio giả nếu không có --wav")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    sr = 16000
    wav = load_audio(args.wav, sr) if args.wav else (np.random.randn(int(args.seconds * sr)) * 0.05).astype(np.float32)
    dur = len(wav) / sr
    print(f"audio {dur:.1f}s, threads={args.threads}")
    for int8 in (False, True):
        name = "int8" if int8 else "fp32"
        if not os.path.exists(os.path.join(args.model, f"stream{'.int8' if int8 else ''}.onnx")):
            continue
        off = OfflineASR(args.model, int8=int8, threads=args.threads)
        off.transcribe(wav[:sr])  # warmup
        t = time.perf_counter()
        text = off.transcribe(wav)
        rtf_off = (time.perf_counter() - t) / dur

        st = StreamingASR(args.model, int8=int8, threads=args.threads)
        step = int(0.02 * sr)  # gói 20 ms như WebRTC
        lat = []
        t = time.perf_counter()
        for i in range(0, len(wav), step):
            t0 = time.perf_counter()
            n_before = st.offset
            st.accept(wav[i : i + step])
            if st.offset != n_before:
                lat.append((time.perf_counter() - t0) * 1000)
        st.finish()
        rtf_st = (time.perf_counter() - t) / dur
        print(f"[{name}] offline RTF={rtf_off:.3f} | streaming RTF={rtf_st:.3f}, chunk={st.cfg['chunk_ms']}ms, "
              f"compute/chunk p50={np.median(lat):.1f}ms p95={np.percentile(lat, 95):.1f}ms")
        if args.wav:
            print(f"  offline: {text}")


if __name__ == "__main__":
    main()
