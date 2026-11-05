"""LLM streaming cho voice agent. Giao diện: async generator trả từng mẩu text.

- EchoLLM: không cần mạng, dùng để test pipeline / đo độ trễ phần speech.
- OpenAICompatLLM: mọi API tương thích OpenAI (OpenAI, OpenRouter, vLLM, llama.cpp server, Ollama…).
  Cấu hình qua biến môi trường LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

SYSTEM_PROMPT = (
    "Bạn là trợ lý giọng nói tiếng Việt. Trả lời ngắn gọn, tự nhiên như đang nói chuyện điện thoại, "
    "tối đa 2-3 câu. Không dùng markdown, ký hiệu đặc biệt hay danh sách. Viết số bằng chữ khi tiện."
)


class EchoLLM:
    def __init__(self, delay_s: float = 0.05):
        self.delay = delay_s

    async def stream(self, history: list[dict]) -> AsyncIterator[str]:
        text = history[-1]["content"] if history else ""
        reply = f"Bạn vừa nói: {text}. Tôi có thể giúp gì thêm không?"
        for word in reply.split(" "):
            await asyncio.sleep(self.delay)  # giả lập tốc độ sinh token
            yield word + " "


class OpenAICompatLLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

    async def stream(self, history: list[dict]) -> AsyncIterator[str]:
        import httpx

        body = {"model": self.model, "stream": True, "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=headers) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line.endswith("[DONE]"):
                        continue
                    delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta


def split_sentences(buffer: str, first: bool = False, min_chars: int = 12) -> tuple[list[str], str]:
    """Cắt đoạn hoàn chỉnh từ buffer token để đưa TTS sớm (không chờ LLM sinh xong).

    Độ trễ cảm nhận do đoạn *đầu tiên* quyết định, nên với đoạn đầu (`first=True`) cho cắt ngay ở
    dấu phẩy / hai chấm khi đã có >= 8 ký tự. Các đoạn sau ưu tiên câu trọn (ngữ điệu tự nhiên hơn),
    chỉ cắt ở dấu phẩy khi câu đã dài (>= 40 ký tự).
    """
    out, start = [], 0
    for i, ch in enumerate(buffer):
        soft_min = 8 if first and not out else 40
        if ch in ".?!\n" or (ch in ",;:" and i - start >= soft_min):
            piece = buffer[start : i + 1].strip()
            if len(piece) >= min_chars or ch in ".?!\n" or (first and not out and len(piece) >= 8):
                if piece:
                    out.append(piece)
                start = i + 1
    return out, buffer[start:]
