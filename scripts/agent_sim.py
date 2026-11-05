"""Client giả lập: stream file audio vào voice agent theo thời gian thực, đo độ trễ và barge-in.

    python scripts/agent_sim.py --url ws://localhost:8765 --wav a.flac [--barge b.flac]

In ra dòng thời gian các sự kiện và độ trễ "voice-to-voice" phía client:
từ lúc người dùng ngừng nói tới khi nhận được audio đầu tiên của agent (gồm cả end_ms chờ im lặng).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
import soundfile as sf


def load16(path):
    w, sr = sf.read(path, dtype="float32")
    assert sr == 16000, "cần 16 kHz"
    return w


async def stream(ws, wav, t0, log):
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    for i in range(0, len(pcm), 320):  # 20 ms
        await ws.send(pcm[i : i + 320].tobytes())
        await asyncio.sleep(0.02)


async def main():
    import websockets

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--barge", help="audio nói chen khi agent đang trả lời")
    args = ap.parse_args()
    silence = np.zeros(int(16000 * 2.0), dtype=np.float32)
    events = []
    first_audio = {}
    async with websockets.connect(args.url, max_size=2**22) as ws:
        t0 = time.perf_counter()
        st = lambda: round(time.perf_counter() - t0, 3)  # noqa: E731

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    turn = len([e for e in events if e[1].get("type") == "final"])
                    if turn not in first_audio:
                        first_audio[turn] = st()
                        events.append((st(), {"type": "<first audio>"}))
                else:
                    events.append((st(), json.loads(m)))

        rt = asyncio.create_task(reader())
        user = load16(args.wav)
        await stream(ws, user, t0, events)
        t_user_end = st()
        if args.barge:
            # chờ agent bắt đầu nói rồi nói chen
            while 1 not in first_audio:
                await stream(ws, np.zeros(320, dtype=np.float32), t0, events)
            await stream(ws, np.zeros(int(16000 * 0.5), dtype=np.float32), t0, events)
            t_barge = st()
            await stream(ws, load16(args.barge), t0, events)
        await stream(ws, silence, t0, events)
        await stream(ws, silence, t0, events)
        rt.cancel()
    for t, e in events:
        if e.get("type") != "partial":
            print(f"{t:7.3f}s  {json.dumps(e, ensure_ascii=False)}")
    if 1 in first_audio:
        print(f"\nvoice-to-voice (người dùng ngừng nói -> audio đầu tiên): {1000 * (first_audio[1] - t_user_end):.0f} ms")
    if args.barge:
        ti = next((t for t, e in events if e.get("type") == "interrupt"), None)
        tv = next((t for t, e in events if t > t_barge and e.get("type") == "vad" and e.get("speech")), None)
        if ti and tv:
            print(f"barge-in: server ngắt agent {1000 * (ti - tv):.0f} ms sau khi VAD thấy người dùng nói chen")
        else:
            print("barge-in: KHÔNG ngắt")


if __name__ == "__main__":
    asyncio.run(main())
