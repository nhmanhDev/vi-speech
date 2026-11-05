"""WebSocket server cho voice agent.

    python -m vispeech.agent.server --asr exported/stt_small --tts exported/tts_matcha [--llm echo|openai] [--port 8765]

Giao thức (1 kết nối = 1 phiên):
    client -> server: binary = PCM16 mono 16 kHz (khối bất kỳ, nên ~20 ms)
    server -> client: binary = PCM16 mono 24 kHz (audio của agent)
                      text JSON: partial | final | agent_text | interrupt | agent_done | vad | metrics
Khi nhận "interrupt", client phải xoá ngay hàng đợi phát audio (đó là barge-in phía client).

Mở http://localhost:8765 trong trình duyệt để nói chuyện bằng micro (server phục vụ luôn web/agent.html
trên cùng cổng; mở file trực tiếp bằng file:// sẽ không chạy vì trình duyệt chặn AudioWorklet).
Vì sao WebSocket chứ không WebRTC: đơn giản, đủ cho demo/LAN. Sản phẩm qua Internet / tổng đài
nên dùng WebRTC (UDP, jitter buffer, AEC chuẩn) hoặc SIP; xem docs/AGENT.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from .llm import EchoLLM, OpenAICompatLLM
from .session import AgentConfig, VoiceAgentSession


def build(args):
    from ..stt.onnx_asr import StreamingASR
    from ..tts.onnx_tts import OnnxTTS

    tts = OnnxTTS(args.tts, threads=2)
    llm = OpenAICompatLLM() if args.llm == "openai" else EchoLLM()
    return (lambda: StreamingASR(args.asr, int8=True)), (lambda text: tts(text, n_steps=args.tts_steps)), llm


async def serve(args):
    import websockets

    asr_factory, tts, llm = build(args)
    cfg = AgentConfig(end_ms=args.end_ms, barge_in=not args.no_barge_in)

    async def handler(ws):
        async def send_event(e):
            await ws.send(json.dumps(e, ensure_ascii=False))

        session = VoiceAgentSession(asr_factory(), tts, llm, send_event, ws.send, cfg)
        await send_event({"type": "ready"})
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    await session.on_audio(msg)
        finally:
            await session._cancel_response()

    html_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "web", "agent.html")

    def process_request(connection, request):
        # Yêu cầu HTTP thường (không phải WebSocket upgrade) -> trả trang web client
        if request.headers.get("Upgrade", "").lower() != "websocket":
            with open(html_path, "rb") as f:
                body = f.read()
            from websockets.datastructures import Headers
            from websockets.http11 import Response

            headers = Headers([("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
            return Response(200, "OK", headers, body)
        return None

    async with websockets.serve(handler, args.host, args.port, max_size=2**22, process_request=process_request):
        print(f"voice agent: http://localhost:{args.port}  (WebSocket cùng cổng)", flush=True)
        await asyncio.Future()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True)
    ap.add_argument("--tts", required=True)
    ap.add_argument("--llm", choices=["echo", "openai"], default="echo")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--end-ms", type=int, default=640)
    ap.add_argument("--tts-steps", type=int, default=4)
    ap.add_argument("--no-barge-in", action="store_true")
    asyncio.run(serve(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
