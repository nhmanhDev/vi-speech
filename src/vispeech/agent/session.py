"""Một phiên hội thoại giọng nói: VAD → streaming STT → LLM → TTS theo câu, có barge-in.

Không phụ thuộc giao thức truyền: server.py (WebSocket) chỉ việc gọi `on_audio()` và cung cấp
2 hàm gửi `send_event(dict)` / `send_audio(bytes)`. Nhờ vậy test được bằng client giả lập.

Trạng thái:
    LISTENING ──VAD start──► USER_SPEAKING ──VAD end──► THINKING ──câu đầu TTS xong──► SPEAKING
        ▲                                                                                 │
        └────────────── nói xong / bị ngắt (barge-in: người dùng nói chen) ◄──────────────┘

Các mốc độ trễ đo mỗi lượt (ms, tính từ lúc VAD xác nhận hết lượt):
    asr_final  → có transcript cuối
    llm_first  → LLM trả token đầu
    tts_first  → có audio câu đầu tiên (= độ trễ "voice-to-voice" phía server, chưa tính mạng
                 và thời gian chờ im lặng end_ms của VAD)
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from .llm import split_sentences
from .vad import Endpointer

SR_IN = 16000
SR_OUT = 24000


@dataclass
class AgentConfig:
    end_ms: int = 640  # im lặng bao lâu thì coi là hết lượt
    barge_in: bool = True
    barge_in_ms: int = 250  # khi agent đang nói, người dùng phải nói >= ngần này mới tính là ngắt (chống echo)
    preroll_ms: int = 300  # giữ lại audio trước khi VAD bật để không mất âm đầu
    audio_chunk_ms: int = 100
    lead_ms: int = 300  # gửi audio trước thời gian thực tối đa ngần này (giới hạn phần "nói thừa" khi bị ngắt)
    max_history: int = 12


@dataclass
class TurnMetrics:
    t_end: float = 0.0
    marks: dict = field(default_factory=dict)

    def mark(self, name: str):
        self.marks.setdefault(name, round((time.perf_counter() - self.t_end) * 1000, 1))


class VoiceAgentSession:
    def __init__(self, asr, tts, llm, send_event: Callable[[dict], Awaitable], send_audio: Callable[[bytes], Awaitable],
                 cfg: AgentConfig | None = None, endpointer: Endpointer | None = None):
        self.asr, self.tts, self.llm = asr, tts, llm
        self.send_event, self.send_audio = send_event, send_audio
        self.cfg = cfg or AgentConfig()
        self.ep = endpointer or Endpointer(end_ms=self.cfg.end_ms)
        self.state = "LISTENING"
        self.history: list[dict] = []
        self.preroll: deque = deque(maxlen=max(1, self.cfg.preroll_ms * SR_IN // 1000 // 512 + 1))
        self.response_task: asyncio.Task | None = None
        self.partial = ""
        self.speech_run = 0.0  # giây người dùng đã nói trong lúc agent đang nói (để quyết định barge-in)
        self.metrics: list[dict] = []

    # ------------------------------------------------------------------ input

    async def on_audio(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768
        events = self.ep.accept(x)
        for ev in events:
            if ev.kind == "start":
                await self._on_speech_start()
            else:
                await self._on_speech_end()
        if self.state == "USER_SPEAKING":
            text = await asyncio.to_thread(self.asr.accept, x)
            if text != self.partial:
                self.partial = text
                await self.send_event({"type": "partial", "text": text})
        elif self.state == "SPEAKING_PENDING_BARGE":
            self.pending_audio.append(x)
            # chỉ cộng thời lượng khi VAD thật sự thấy tiếng nói (không cộng khoảng im lặng
            # trong lúc chờ VAD báo "end"), nếu không tiếng động ngắn cũng thành ngắt lời
            if self.ep.last_prob >= self.ep.on:
                self.speech_run += len(x) / SR_IN
            if self.speech_run * 1000 >= self.cfg.barge_in_ms:
                await self._barge_in()
        else:
            self.preroll.append(x)

    async def _on_speech_start(self):
        await self.send_event({"type": "vad", "speech": True})
        if self.state in ("SPEAKING", "THINKING") and self.cfg.barge_in:
            # chưa ngắt ngay: chờ đủ barge_in_ms để chắc là người nói thật, không phải echo/tiếng động
            self.prev_state = self.state
            self.state = "SPEAKING_PENDING_BARGE"
            self.pending_audio = list(self.preroll)
            self.speech_run = 0.0
            return
        if self.state in ("SPEAKING", "THINKING"):
            return  # không cho ngắt
        self._begin_user_turn(list(self.preroll))

    def _begin_user_turn(self, preroll: list):
        self.state = "USER_SPEAKING"
        self.asr.reset()
        self.partial = ""
        for chunk in preroll:
            self.asr.accept(chunk)
        self.preroll.clear()

    async def _barge_in(self):
        await self._cancel_response()
        await self.send_event({"type": "interrupt"})  # client phải xoá ngay audio đang đệm
        self._begin_user_turn(self.pending_audio)
        self.pending_audio = []

    async def _on_speech_end(self):
        await self.send_event({"type": "vad", "speech": False})
        if self.state == "SPEAKING_PENDING_BARGE":  # tiếng ngắn, không đủ để ngắt -> bỏ qua
            self.state = self.prev_state if self.response_task and not self.response_task.done() else "LISTENING"
            return
        if self.state != "USER_SPEAKING":
            return
        m = TurnMetrics(t_end=time.perf_counter())
        text = (await asyncio.to_thread(self.asr.finish)).strip()
        m.mark("asr_final")
        await self.send_event({"type": "final", "text": text})
        if not text:
            self.state = "LISTENING"
            return
        self.history.append({"role": "user", "content": text})
        self.state = "THINKING"
        self.response_task = asyncio.create_task(self._respond(m))

    # ------------------------------------------------------------------ output

    async def _cancel_response(self):
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            try:
                await self.response_task
            except asyncio.CancelledError:
                pass

    async def _respond(self, m: TurnMetrics):
        spoken: list[str] = []
        buf = ""
        clock = None  # thời điểm audio đã gửi sẽ phát xong (theo đồng hồ thực)
        try:
            async for tok in self.llm.stream(self.history[-self.cfg.max_history :]):
                m.mark("llm_first")
                buf += tok
                sentences, buf = split_sentences(buf)
                for s in sentences:
                    clock = await self._speak(s, m, clock)
                    spoken.append(s)
            if buf.strip():
                clock = await self._speak(buf.strip(), m, clock)
                spoken.append(buf.strip())
            if clock:
                await asyncio.sleep(max(0.0, clock - time.perf_counter()))
            await self.send_event({"type": "agent_done"})
        except asyncio.CancelledError:
            spoken.append("…")  # ghi nhận câu trả lời bị ngắt giữa chừng
            raise
        finally:
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken)})
            self.metrics.append({"text": self.history[-2]["content"] if len(self.history) > 1 else "", **m.marks})
            await self.send_event({"type": "metrics", **m.marks})
            if self.state in ("THINKING", "SPEAKING"):
                self.state = "LISTENING"

    async def _speak(self, sentence: str, m: TurnMetrics, clock: float | None) -> float:
        await self.send_event({"type": "agent_text", "text": sentence})
        wav = await asyncio.to_thread(self.tts, sentence)
        m.mark("tts_first")
        if self.state == "THINKING":
            self.state = "SPEAKING"
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        step = SR_OUT * self.cfg.audio_chunk_ms // 1000
        now = time.perf_counter()
        clock = max(clock or now, now)
        for i in range(0, len(pcm), step):
            # pacing: không gửi trước quá lead_ms, để khi bị ngắt phần audio "thừa" ở client nhỏ
            wait = clock - time.perf_counter() - self.cfg.lead_ms / 1000
            if wait > 0:
                await asyncio.sleep(wait)
            chunk = pcm[i : i + step]
            await self.send_audio(chunk.tobytes())
            clock += len(chunk) / SR_OUT
        return clock
