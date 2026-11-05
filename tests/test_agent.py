"""Test voice agent bằng giọng thật (fixtures InfoRe), ASR/TTS giả lập.

Kiểm: VAD bắt đúng đầu/cuối lượt, pipeline sinh câu trả lời, barge-in ngắt được agent,
tiếng động ngắn khi agent đang nói KHÔNG làm ngắt.
"""

import asyncio
import json
import os

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("silero_vad")

from vispeech.agent.llm import EchoLLM, split_sentences  # noqa: E402
from vispeech.agent.session import AgentConfig, VoiceAgentSession  # noqa: E402
from vispeech.agent.vad import Endpointer  # noqa: E402


FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def speech_like(seconds: float, seed: int = 0) -> np.ndarray:
    """Giọng thật (InfoRe, xem fixtures/README.md), cắt/lặp cho đủ `seconds`."""
    w, _ = sf.read(os.path.join(FIX, f"infore_0000{1 + seed % 2}.flac"), dtype="float32")
    w = w[int(0.6 * 16000) :]  # bỏ im lặng đầu file
    return np.tile(w, int(np.ceil(seconds * 16000 / len(w))))[: int(seconds * 16000)]


def silence(seconds: float) -> np.ndarray:
    return (0.001 * np.random.default_rng(1).standard_normal(int(seconds * 16000))).astype(np.float32)


class FakeASR:
    def __init__(self):
        self.n = 0

    def reset(self):
        self.n = 0

    def accept(self, x):
        self.n += len(x)
        return "xin chào" if self.n > 8000 else ""

    def finish(self):
        text = "xin chào" if self.n > 8000 else ""
        self.reset()
        return text


def fake_tts(text: str) -> np.ndarray:
    return np.zeros(int(24000 * 0.08 * len(text.split())), dtype=np.float32)  # ~80 ms / từ


async def run_session(chunks, cfg=None):
    events, audio = [], []

    async def send_event(e):
        events.append(e)

    async def send_audio(b):
        audio.append(len(b))

    s = VoiceAgentSession(FakeASR(), fake_tts, EchoLLM(delay_s=0.0), send_event, send_audio, cfg or AgentConfig(lead_ms=50))
    for c in chunks:  # mô phỏng micro gửi 20 ms một lần, theo thời gian thực x4
        for i in range(0, len(c), 320):
            await s.on_audio((np.clip(c[i : i + 320], -1, 1) * 32767).astype(np.int16).tobytes())
            await asyncio.sleep(0.005)
    if s.response_task:
        try:
            await asyncio.wait_for(s.response_task, 10)
        except asyncio.CancelledError:
            pass
    return s, events, audio


def kinds(events):
    return [e["type"] for e in events if e["type"] != "vad"]


def test_endpointer_detects_speech():
    ep = Endpointer()
    evs = ep.accept(np.concatenate([silence(0.5), speech_like(1.5), silence(1.0)]))
    assert [e.kind for e in evs] == ["start", "end"]
    assert 0.3 < evs[0].t < 0.8


def test_turn_produces_reply():
    s, events, audio = asyncio.run(run_session([silence(0.4), speech_like(1.2), silence(1.2)]))
    k = kinds(events)
    assert "final" in k and "agent_text" in k and "agent_done" in k
    assert sum(audio) > 0
    m = next(e for e in events if e["type"] == "metrics")
    assert m["asr_final"] <= m["tts_first"]
    assert s.history[-1]["role"] == "assistant"


def test_barge_in_interrupts_agent():
    # người dùng nói, agent bắt đầu trả lời, người dùng nói chen 1 s -> phải có interrupt
    s, events, _ = asyncio.run(run_session([silence(0.3), speech_like(1.2), silence(0.9), speech_like(1.0), silence(1.2)]))
    k = kinds(events)
    assert "interrupt" in k
    assert k.count("final") == 2
    assert any(h["content"].endswith("…") for h in s.history if h["role"] == "assistant")


def test_short_noise_does_not_interrupt():
    cfg = AgentConfig(lead_ms=50, barge_in_ms=400)
    s, events, _ = asyncio.run(run_session([silence(0.3), speech_like(1.2), silence(0.9), speech_like(0.2), silence(2.0)], cfg))
    assert "interrupt" not in kinds(events)


def test_split_sentences():
    out, rest = split_sentences("Xin chào anh. Hôm nay trời đẹp quá, anh có muốn đi")
    assert out == ["Xin chào anh."] and rest.startswith(" Hôm nay")
    json.dumps(out)
