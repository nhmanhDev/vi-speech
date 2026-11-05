# Dữ liệu: nguồn, xử lý, và những gì đã đo

Tài liệu này ghi lại *vì sao* pipeline dữ liệu làm như vậy, kèm số đo thật trên dữ liệu.

## Nguồn đang dùng

| Dataset | Dùng cho | Sau xử lý | License |
|---|---|---|---|
| VLSP2020 VinAI 100h (`doof-ferb/vlsp2020_vinai_100h`) | STT | 92,2 h, 54.497 câu (loại 1.114 vì độ dài, 816 vì văn bản) | CC-BY-4.0 |
| InfoRe1 25h (`doof-ferb/infore1_25hours`), 1 giọng nữ, 44,1 kHz | STT + TTS | STT: 24,9 h, 14.933 câu | CC-BY-4.0 (theo mirror) |

Các bộ lớn hơn (viVoice, PhoAudiobook, Bud500…) hầu hết **phi thương mại**: chỉ dùng cho bản nghiên cứu,
giữ checkpoint riêng. Danh sách đầy đủ: note SecondBrain "VieNeu-TTS - Mổ xẻ kỹ thuật và chuẩn bị dữ liệu".

## Văn bản (`src/vispeech/text/`)

- Unicode NFC, chữ thường, số/ngày/giờ/đơn vị/viết tắt đọc thành chữ ("21" → "hai mươi mốt", "24" → "hai mươi tư").
- **Một kiểu bỏ dấu**: InfoRe trộn "hoà"/"hòa" (27/9 lần), "khoẻ"/"khỏe" (3/12). Nếu không thống nhất, model học
  2 cách viết cho 1 âm. Quy về "hòa, khỏe, thủy", nhưng không đụng "hoàng" (vần đóng) và "quý" (qu là phụ âm đầu).
- STT bỏ dấu câu; TTS giữ `, . ? !` làm điểm ngắt nghỉ.

## STT (`vispeech.data.prepare`)

- 16 kHz, log-mel 80 (25 ms / 10 ms), lưu fp16 memmap: ~16 KB mỗi giây audio, đọc ngẫu nhiên nhanh.
- Lọc 0,5–20 s, loại câu có ký tự ngoài bảng chữ cái.
- **Ràng buộc CTC**: số frame sau subsampling phải ≥ số ký tự. Đo trên InfoRe: tốc độ trung vị 15,5 ký tự/s
  (p99 19,9), tỉ lệ frame/ký tự thấp nhất 1,18 với subsampling 4×. Vậy 4× + char-level an toàn; 8× thì không
  (cần BPE/âm tiết). Bước prepare tự loại câu vi phạm.
- CMVN toàn cục tính trong lúc prepare, nhúng vào model (ONNX tự chuẩn hoá).
- Không denoise dữ liệu STT: giữ nhiễu thật vì lúc dùng cũng có nhiễu.

## TTS (`vispeech.tts.prepare` → `vispeech.tts.asr_filter`)

Từng bước và lý do:

1. **QC thô**: loại clip im lặng, clipping (> 0,1% mẫu chạm ±0,99).
2. **Silero VAD cắt im lặng đầu/cuối**, chừa 150 ms. InfoRe có 0,35–0,64 s im lặng đầu câu; nếu để nguyên, model
   học "im lặng ngẫu nhiên" trước mỗi câu và duration bị lệch. Loại câu có khoảng lặng giữa câu > 1,5 s.
3. **Loudness -20 LUFS** (ITU-R BS.1770), giới hạn đỉnh -1 dBFS. InfoRe gốc dao động -21 đến -15 LUFS.
4. **24 kHz + mel 100 bin giống hệt Vocos** (`charactr/vocos-mel-24khz`). Đã kiểm: cùng đầu vào thì mel của ta
   và feature extractor của Vocos *trùng tuyệt đối*; audio tái tạo bằng Vocos từ mel lưu trữ được PhoWhisper
   nhận dạng y hệt audio gốc.
5. **SNR đo bằng VAD** (công suất đoạn có tiếng / đoạn im lặng), ngưỡng 20 dB. *Bài học*: ban đầu dùng WADA-SNR
   (ước lượng mù) và nó loại nhầm nhiều câu sạch: SNR đo trực tiếp của nhóm bị loại có trung vị 27 dB.
   Chuyển sang SNR theo VAD: nhóm bị loại có trung vị 16 dB, nhóm giữ lại ~31 dB. Một shard 996 câu: loại 88 (9%).
6. **Kiểm nhãn bằng ASR** (PhoWhisper-medium, BSD-3): nhận dạng lại, giữ câu CER ≤ 5%. TTS học thẳng lỗi nhãn
   (đọc sót chữ, lặp chữ), nên đây là bước quan trọng nhất với dữ liệu cào. Khi *đánh giá* TTS phải dùng ASR khác
   ASR dùng để lọc.
7. Ghi kèm tốc độ nói (âm tiết/s, InfoRe p1–p99: 3,0–4,6), LUFS gốc, sample rate gốc để lọc/thống kê sau.

Mỗi lần chạy sinh `meta.json` (phân phối p1/p50/p99 của mọi chỉ số, số câu bị loại theo lý do) và
`rejected.jsonl` (câu bị loại + lý do) để nghe kiểm tra.

## Việc còn thiếu

- DNSMOS để chấm chất lượng cảm nhận (thay/bổ sung SNR).
- Dedupe khi trộn nhiều bộ cào từ YouTube (hash văn bản + x-vector).
- Chia dev/test theo người nói khi có dữ liệu nhiều giọng (hiện chia theo hash id câu).
- Nghe thử ngẫu nhiên câu bị loại / giữ lại (chưa làm, cần tai người).
