# Dữ liệu: dùng bộ nào, lấy ở đâu, xử lý ra sao

Tài liệu này trả lời 3 câu hỏi: **đang dùng dữ liệu gì**, **mỗi bước xử lý làm gì và vì sao**, **còn lại bao nhiêu
sau mỗi bước**. Mọi con số dưới đây lấy từ file `meta.json` / `asr_report.json` do chính pipeline sinh ra (ngày 2026-11-05),
không phải ước lượng.

## 1. Tổng quan

| Dùng cho | Dataset | Gốc | Sau xử lý (dùng để train) |
|---|---|---|---|
| STT (train) | VLSP2020 VinAI 100h | 56.427 câu | **54.497 câu, 92,2 giờ** |
| STT (train) | InfoRe1 25h | 14.935 câu | **14.933 câu, 24,9 giờ** |
| TTS (train) | InfoRe1 25h | 14.935 câu | **13.044 câu, ≈ 20,7 giờ** (12.909 train + 135 dev) |
| STT (chỉ đánh giá) | FLEURS-vi test | 857 câu | 857 câu |
| STT (chỉ đánh giá) | VIVOS test | 760 câu | 760 câu |

Tổng dữ liệu train STT: **117,1 giờ**, 69.430 câu. Dữ liệu TTS: **1 giọng nữ** (InfoRe).

## 2. Nguồn dữ liệu

| Dataset | Lấy ở đâu | Nội dung | License |
|---|---|---|---|
| VLSP2020 VinAI 100h | HF `doof-ferb/vlsp2020_vinai_100h` (bản mirror; gốc do VinBigData công bố cho VLSP 2020). Commit `7d58a85`, 35 file parquet, 11,6 GB | Tin tức + hội thoại tự nhiên, nhiều người nói, 16 kHz. Nhãn chữ thường, không dấu câu, số đã viết bằng chữ, **tên nước ngoài phiên âm kiểu Việt** ("sinh ga po", "ních phe rô") | CC-BY-4.0 (theo mirror) |
| InfoRe1 25h | HF `doof-ferb/infore1_25hours` (mirror; gốc InfoRe Technology). Commit `55062d7`, 15 file parquet, 7,8 GB | **1 giọng nữ**, đọc văn bản (người đọc thuê), 44,1 kHz, nhãn chữ thường không dấu câu | CC-BY-4.0 (theo mirror; card Piper ghi "Unknown") |
| FLEURS-vi test | HF `google/fleurs`, config `vi_vn`, split test | Đọc câu Wikipedia, nhiều người nói. Nhãn có chữ số và tên nước ngoài nguyên gốc | CC-BY-4.0 |
| VIVOS test | HF `AILAB-VNUHCM/vivos` (`data/vivos.tar.gz`, chỉ giải nén phần `test/`) | Đọc trong phòng yên tĩnh, 19 người, nhãn CHỮ HOA | CC-BY-NC-SA-4.0: **chỉ dùng để đánh giá**, không train |

Không dùng bộ nào phải xin quyền truy cập. Danh sách toàn bộ dữ liệu free đang được rà soát riêng
(note SecondBrain "dữ liệu tiếng nói tiếng Việt tải tự do").

## 3. Chuẩn hoá văn bản (dùng chung cho STT, TTS, đánh giá)

Code: `src/vispeech/text/normalize.py`, `numbers.py`. Mỗi câu đi qua:

1. **Unicode NFC**: "ạ" có thể là 1 hoặc 2 mã ký tự; không chuẩn hoá thì cùng một chữ thành 2 token khác nhau.
2. **Viết tắt**: TP.HCM → thành phố hồ chí minh, UBND → ủy ban nhân dân, THPT, GS., TS., VND… (danh sách trong code).
3. **Ngày, giờ**: 12/5/2024 → ngày mười hai tháng năm năm hai nghìn không trăm hai mươi tư; 10h30 → mười giờ ba mươi phút.
4. **Số điện thoại** đọc từng chữ số; **đơn vị**: km/h, kg, %, đ…
5. **Số**: dấu chấm ngăn nghìn (1.500.000), dấu phẩy thập phân (3,5 → ba phẩy năm). Quy tắc đọc: 21 → hai mươi **mốt**,
   24 → hai mươi **tư**, 15/25 → **lăm**, 105 → một trăm **linh** năm, 1005 → một nghìn **không trăm** linh năm.
6. **Chữ thường**, một **kiểu bỏ dấu** duy nhất: dữ liệu trộn "hoà/hòa" (InfoRe: 27/9 lần), "khoẻ/khỏe" (3/12).
   Quy về "hòa, khỏe, thủy" nhưng không đụng "hoàng" (vần đóng) và "quý" (qu là phụ âm đầu).
7. STT: bỏ toàn bộ dấu câu. TTS: giữ `, . ? !` làm điểm ngắt nghỉ.

Bảng ký tự: 96 token (blank, unk, khoảng trắng, 12 nguyên âm × 6 thanh, phụ âm, f/j/w/z); TTS thêm 4 dấu câu.

## 4. Pipeline STT (`python -m vispeech.data.prepare`)

Đầu vào: parquet HF (cột `audio{bytes}`, `transcription`). Từng câu:

| Bước | Làm gì | Vì sao |
|---|---|---|
| 1 | Chuẩn hoá văn bản (mục 3). Loại câu rỗng hoặc có ký tự ngoài bảng chữ (`unknown_chars`) | Model CTC char-level không biết ký tự lạ |
| 2 | Decode audio, về mono 16 kHz (soxr) | Chuẩn chung cho ASR |
| 3 | Loại câu < 0,5 s hoặc > 20 s (`duration`) | Câu quá dài tốn bộ nhớ GPU và hay lệch nhãn |
| 4 | Log-mel 80 chiều, cửa sổ 25 ms, bước 10 ms (numpy, `src/vispeech/audio.py`) | Cùng một code cho train và chạy trên CPU/edge |
| 5 | Loại câu có số frame sau subsampling ×4 < số ký tự + 1 (`ctc_too_fast`) | CTC bắt buộc T ≥ U; vi phạm thường do **nhãn lệch audio** |
| 6 | Lưu log-mel fp16 vào 1 file liền (`feats.f16`, đọc bằng memmap) + `index.jsonl` | Kaggle chỉ có 4 CPU; tính trước thì GPU không phải chờ |
| 7 | Tính mean/std toàn cục (CMVN), nhúng vào model | Model ONNX tự chuẩn hoá, edge không cần biết |

**Kết quả:**

| | VLSP2020 | InfoRe1 |
|---|---|---|
| Giữ | 54.497 câu (92,2 h) | 14.933 câu (24,9 h) |
| Loại vì độ dài | 1.114 | 1 |
| Loại vì văn bản / CTC | 816 | 1 |

Soi 1 shard VLSP (1.613 câu, file `rejected.jsonl`): 123 câu loại vì **dài 20–63 giây** (không câu nào < 0,5 s);
10 câu `ctc_too_fast`, ví dụ audio 1,34 s nhưng nhãn 35 ký tự ("nhiều người biểu tình chống ông tập", 26 ký tự/giây,
người không nói nhanh vậy được) → nhãn lệch audio, loại là đúng.

Không khử nhiễu dữ liệu STT: giữ nhiễu thật vì lúc dùng cũng có nhiễu.

Chia train/dev: theo hash id câu, 1% làm dev (696 câu). Hạn chế: VLSP không có nhãn người nói nên dev có thể trùng
giọng với train; vì vậy WER báo cáo chính thức lấy trên FLEURS và VIVOS (người nói không có trong train).

## 5. Pipeline TTS (`python -m vispeech.tts.prepare` → `vispeech.tts.asr_filter` → `vispeech.tts.prosody`)

TTS cần dữ liệu sạch hơn STT nhiều: model học thẳng mọi lỗi (đọc sót chữ, im lặng thừa, nhiễu).

| Bước | Làm gì | Vì sao | Đã kiểm thế nào |
|---|---|---|---|
| 1 | Decode, mono, giữ sample rate gốc để đo QC | | |
| 2 | Loại câu im lặng, câu **clipping** (> 0,1% mẫu chạm ±0,99) | Méo tiếng | |
| 3 | **Silero VAD** cắt im lặng đầu/cuối, chừa 150 ms | InfoRe có 0,35–0,64 s im lặng đầu câu; để nguyên thì model học "im lặng ngẫu nhiên" | Xem lại timestamp VAD trên câu thật |
| 4 | Loại câu có khoảng lặng giữa câu > 1,5 s | Đọc vấp, cắt câu sai | |
| 5 | Chuẩn **loudness -20 LUFS** (ITU-R BS.1770), đỉnh ≤ -1 dBFS | InfoRe gốc dao động -19,8 đến -15,7 LUFS | |
| 6 | Resample **24 kHz**, tính **mel 100 bin giống hệt Vocos** | Vocoder Vocos (pretrained) chỉ hiểu đúng định dạng mel này | Cùng đầu vào → mel trùng tuyệt đối với Vocos; audio tái tạo được ASR nhận đúng như audio gốc |
| 7 | Đo **SNR bằng VAD** (công suất đoạn có tiếng / đoạn im lặng), loại < 20 dB | Nhiễu nền | Xem bài học bên dưới |
| 8 | Chuẩn hoá văn bản, giữ dấu câu | | |
| 9 | **Kiểm nhãn bằng ASR**: PhoWhisper-medium (BSD-3) nghe lại từng câu, tính CER; train chỉ dùng câu **CER ≤ 5%** | Nhãn sai/đọc sót là lỗi TTS khó sửa nhất | |
| 10 | **Đo ngữ điệu** mỗi câu: cao độ trung vị, biên độ cao độ (p90–p10), tốc độ nói; z-score theo người nói | Để điều khiển ngữ điệu khi tổng hợp | Test trên giọng thật |

**Kết quả (InfoRe1):**

| Sau bước | Số câu | Ghi chú |
|---|---|---|
| Gốc | 14.935 | |
| QC âm thanh (bước 2–7) | **13.742 (21,8 h)** | loại 1.192 vì SNR < 20 dB, 1 câu im lặng |
| Kiểm nhãn ASR (bước 9) | **13.044 (≈ 20,7 h)** | loại 698 câu CER > 5% |

Kiểm nhãn bằng PhoWhisper-medium trên 13.742 câu: CER trung bình **0,90%**; 75,0% câu khớp hoàn toàn; 83,9% CER ≤ 2%;
94,9% CER ≤ 5%; 99,2% CER ≤ 10%. Nhãn InfoRe nhìn chung rất sạch.

Phân bố sau xử lý: thời lượng p1/p50/p99 = 2,95 / 5,49 / 9,89 s; SNR p50 = 30,3 dB; tốc độ 2,8–4,5 âm tiết/s.

Ngữ điệu (trên 908 câu mẫu): cao độ 13,5 ± 1,3 semitone trên 100 Hz (≈ 218 Hz), biên độ 7,2 ± 2,6 semitone,
tốc độ log 1,33 ± 0,10. **Cao độ trung bình chỉ dao động ±1,3 semitone giữa các câu: giọng đọc đều**, khớp với cảm nhận
"nghe robot". Không dùng năng lượng làm đặc trưng vì sau bước 5 mọi câu cùng loudness (độ lệch chuẩn 0,73 dB).

### Bài học khi làm pipeline TTS (đã sửa)

- **WADA-SNR loại nhầm câu sạch.** Ban đầu đo SNR bằng WADA (ước lượng mù): loại 46/996 câu, nhưng đo trực tiếp thì
  nhóm bị loại có SNR trung vị 27 dB (sạch). Đổi sang SNR theo VAD: nhóm bị loại 16 dB, nhóm giữ 31 dB.
- **Tràn số int8** trong phép tính vùng im lặng làm số câu bị loại nhảy lên 440/996; phát hiện nhờ tỉ lệ loại bất thường.

### Vì sao giọng gốc nghe "robot"

Đã kiểm 3 cách: (1) file gốc InfoRe chưa qua xử lý cho kết quả ASR y hệt bản đã xử lý và bản Vocos; (2) phổ bình thường,
còn năng lượng tới ~12 kHz; (3) cao độ trung bình giữa các câu ít biến thiên. Kết luận: do **cách đọc của bộ dữ liệu**,
không phải do pipeline. Model train trên InfoRe sẽ không tự nhiên hơn InfoRe → cần thêm dữ liệu biểu cảm.

## 6. Dữ liệu nằm ở đâu

| Bước | Kaggle kernel (output) | Nội dung |
|---|---|---|
| STT VLSP | `nhmanhaidev/vi-speech-prep-vlsp` → `prep/vlsp/` | `feats.f16` (5,3 GB), `index.jsonl`, `meta.json` |
| STT InfoRe | `nhmanhaidev/vi-speech-prep-infore` → `prep/infore/` | như trên |
| TTS InfoRe | `nhmanhaidev/vi-speech-tts-prep-infore` → `tts/infore/` | `wav.i16` (24 kHz), `mel.f16`, `index.jsonl`, `index.asr.jsonl`, `rejected.jsonl`, `meta.json`, `asr_report.json` |

Kernel train gắn các output này làm input và copy sang ổ local `/tmp` trước khi train (đọc ngẫu nhiên trên ổ mạng
`/kaggle/input` làm train chậm 14 lần).

## 7. Chạy lại từ đầu

```bash
# STT
python scripts/kaggle_job.py prep --hf doof-ferb/vlsp2020_vinai_100h --name vlsp
python scripts/kaggle_job.py prep --hf doof-ferb/infore1_25hours --name infore
# TTS (GPU vì có bước PhoWhisper)
python scripts/kaggle_job.py tts-prep --hf doof-ferb/infore1_25hours --name infore
# chạy local
python -m vispeech.data.prepare --parquet 'data/vlsp/*.parquet' --out data/prep/vlsp
python -m vispeech.tts.prepare --parquet 'data/infore/*.parquet' --out data/tts/infore --speaker infore
python -m vispeech.tts.asr_filter --data data/tts/infore
python -m vispeech.tts.prosody --data data/tts/infore
```

## 8. Hạn chế và việc tiếp theo

- TTS chỉ có 1 giọng đọc đều → cần dữ liệu tự nhiên/biểu cảm hơn (đang rà nguồn free, không phải xin quyền).
- Chưa dedupe khi trộn nhiều bộ (cần khi thêm dữ liệu cào từ YouTube).
- Chưa có DNSMOS; chưa speed perturb / trộn nhiễu cho STT.
- Dev STT chia theo câu, chưa theo người nói (VLSP không có nhãn người nói).
- Nhãn VLSP phiên âm tên nước ngoài kiểu Việt, FLEURS giữ nguyên tên gốc → WER trên FLEURS bị phạt ở các từ này.
