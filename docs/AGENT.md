# Voice Agent: kiến trúc, độ trễ, barge-in

```
trình duyệt (mic, AEC/NS/AGC) ──PCM16 16 kHz, gói 20 ms──► WebSocket server
                                                             │
     ┌───────────────────────────────────────────────────────┘
     ▼
 Silero VAD (32 ms/khung) ─► máy trạng thái lượt nói ─► Streaming STT (Conformer-CTC, chunk 640 ms)
                                   │ hết lượt (im lặng 640 ms)              │ partial transcript
                                   ▼                                         ▼
                         LLM streaming token ─► cắt đoạn ─► TTS từng đoạn ─► PCM16 24 kHz (pacing) ─► client phát
                                   ▲                                                         │
                                   └──────── barge-in: VAD thấy người dùng nói ≥ 250 ms ◄────┘
                                              → huỷ LLM/TTS, gửi "interrupt", client xoá hàng đợi phát
```

Code: `src/vispeech/agent/` (`vad.py`, `session.py`, `llm.py`, `server.py`), client `web/agent.html`.

## Độ trễ đo được (server CPU, model debug, LLM giả lập 50 ms/token)

| Chặng (tính từ lúc VAD chốt hết lượt) | ms |
|---|---|
| STT ra transcript cuối | 3–4 |
| LLM token đầu | ~55 |
| TTS xong đoạn đầu (≥ 8 ký tự) | ~200–220 |
| **Voice-to-voice phía client** (người dùng ngừng nói → nghe audio đầu) | **~800**, trong đó 640 là chờ im lặng |
| Barge-in: VAD thấy nói chen → server ngắt | ~250 (đúng ngưỡng `barge_in_ms`) |

Bài học từ lần đo đầu (2.048 ms): nút thắt không phải model mà là **chờ LLM sinh xong câu đầu**.
Cho đoạn đầu cắt sớm ở dấu phẩy/hai chấm → 802 ms. Phần lớn còn lại là `end_ms` của VAD.

## Các quyết định thiết kế (và câu trả lời khi phỏng vấn)

**Endpointing.** Chỉ dùng im lặng là đơn giản nhưng có đánh đổi: `end_ms` ngắn thì cắt ngang người đang ngừng
suy nghĩ, dài thì agent chậm. Hướng tốt hơn: *turn detector* ngữ nghĩa (ví dụ LiveKit turn detector, Pipecat
Smart Turn) xem transcript/âm điệu đã trọn ý chưa; khi chắc chắn thì chốt sớm (~200–300 ms), khi lửng lơ
("tôi muốn hỏi là…") thì chờ lâu hơn. Smart Turn được báo cáo chỉ ~81% với tiếng Việt → fine-tune cho
tiếng Việt là việc đáng làm.

**Barge-in.** Hai phía cùng phải làm: server huỷ task LLM/TTS ngay; client xoá audio đã đệm. Server gửi
audio theo nhịp thời gian thực, chỉ đi trước `lead_ms` = 300 ms, nên khi bị ngắt phần "nói thừa" ở client
tối đa ~300 ms, và server biết agent đã nói tới đâu (ghi lịch sử câu bị ngắt kèm "…"). Chống ngắt nhầm: phải
nói ≥ `barge_in_ms` (250 ms) mới tính, và chỉ đếm khung VAD thật sự là tiếng nói (lỗi đã gặp: đếm cả khoảng
im lặng làm tiếng động 0,2 s cũng ngắt được agent; test `test_short_noise_does_not_interrupt` giữ lỗi này
không quay lại).

**Echo.** Không đeo tai nghe thì micro thu lại giọng agent → tự ngắt chính mình. Lớp 1: AEC của trình duyệt
(`echoCancellation: true`). Lớp 2: ngưỡng barge-in cao hơn khi agent đang nói. Sản phẩm thật (tổng đài) cần
AEC phía server hoặc dựa vào AEC của WebRTC/SIP endpoint.

**Pre-roll.** Giữ 300 ms audio trước khi VAD bật để STT không mất phụ âm đầu.

**TTS theo đoạn.** LLM sinh tới đâu TTS tới đó; đoạn đầu ngắn để nhanh, các đoạn sau là câu trọn để ngữ điệu
tự nhiên.

**WebSocket hay WebRTC.** WebSocket (TCP) đơn giản, đủ cho demo/LAN. Qua Internet thật: WebRTC (UDP, jitter
buffer, FEC, AEC/NS chuẩn, NAT traversal) hoặc SIP/RTP cho tổng đài. Có thể dùng LiveKit/Pipecat làm lớp
transport và giữ nguyên `VoiceAgentSession`.

**Mở rộng nhiều cuộc gọi đồng thời.** Mỗi phiên hiện chạy STT/TTS trên CPU qua thread pool. Khi nhiều phiên:
gom batch STT chunk của nhiều phiên lên GPU (dynamic batching, Triton), TTS tương tự; LLM qua vLLM
(continuous batching). Theo dõi p95 độ trễ từng chặng chứ không chỉ trung bình.

## Chạy thử

```bash
python -m vispeech.agent.server --asr exported/stt_small --tts exported/tts_matcha            # LLM giả lập
LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... python -m vispeech.agent.server ... --llm openai
# mở http://localhost:8765 (đeo tai nghe)
python scripts/agent_sim.py --wav tests/fixtures/infore_00002.flac --barge tests/fixtures/infore_00001.flac
node scripts/agent_browser_test.js mic.wav    # Chromium headless + micro giả
```
