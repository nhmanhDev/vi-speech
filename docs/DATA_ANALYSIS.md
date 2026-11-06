# Phân tích dữ liệu: có gì, đo được gì, thiếu gì, cần gì để vượt VieNeu

Tài liệu này khác `DATA.md`: `DATA.md` mô tả **bộ nào đã dùng và xử lý ra sao**; tài liệu này **phân tích sâu chất lượng dữ liệu**, so sánh các nguồn, chỉ ra **khoảng trống** so với mục tiêu
(giọng tự nhiên, clone từ ~10 s, thẻ cảm xúc / âm thanh không lời, chạy CPU), và nêu **những gì chưa đo** để khỏi làm đi làm lại.

Nhãn độ tin cậy dùng xuyên suốt:

| Nhãn | Nghĩa |
|---|---|
| **[đo]** | em tự đo trong dự án này; ghi cỡ mẫu `n` |
| **[card]** | tác giả bộ dữ liệu tự ghi, em chưa kiểm |
| **[ước lượng]** | em suy từ số khác |
| **[giả thuyết]** | suy luận, chưa kiểm chứng |

---

## 0. Tóm tắt (đọc phần này nếu chỉ có 2 phút)

1. **Dữ liệu đã dùng nhỏ và sai loại cho mục tiêu mới.** Đã train trên 117 giờ (VLSP 92,2 h + InfoRe 24,9 h; TTS chỉ 20,7 h một giọng nữ đọc đều). Mục tiêu "tự nhiên, clone, cảm xúc" cần đa giọng, nói tự nhiên, có tiếng cười, thở.
2. **Dữ liệu tải tự do có nhiều giờ nhưng thiếu đúng thứ cần.** Có > 17.000 giờ tải được ngay, nhưng đo thực tế ~5,5 giờ trên 3 nguồn cho thấy **tiếng cười ≤ 0,5 lần/giờ, khóc/ho/thì thầm = 0**. Dữ liệu biểu cảm là khoảng trống lớn nhất, và không có đường đào từ dữ liệu có sẵn.
3. **Dữ liệu đa giọng tốt nhất cho TTS đều có vấn đề license hoặc nguồn:** vi-podcast (license không ghi, một shard bị 2 người nói áp đảo), VieNeu-140h (NC), YODAS2-Sidon (đã qua khôi phục, nghi mất âm thanh không lời), LEMAS (18% clip có nhạc nền).
4. **Em đã nhiều lần ước lượng sai dữ liệu** (xem mục 9). Vì vậy tài liệu này tách rõ đo và ước lượng, và mục 8 liệt kê **các phép đo phải làm trước khi tốn GPU**.

---

## 1. Dữ liệu đã dùng: phân tích sâu

### 1.1 VLSP2020 VinAI 100h (STT)

| Chỉ số | Giá trị |
|---|---|
| Gốc → sau xử lý | 56.427 → **54.497 câu**, **92,17 h** [đo] |
| Loại | 1.114 vì độ dài, 816 vì văn bản/CTC, 0 lỗi decode [đo] |
| Mẫu soi kỹ | 1 shard / 1.613 câu: giữ 1.480 (3,26 h, TB ~7,9 s/câu); loại 123 vì dài, 10 vì `ctc_too_fast` [đo] |
| Độ dài bị loại | 123 câu **toàn bộ dài 20,02–63,36 s**, không câu nào < 0,5 s [đo] |
| Nhãn | chữ thường, không dấu câu, số viết bằng chữ, **tên nước ngoài phiên âm kiểu Việt** ("sinh ga po", "ních phe rô", "phứt xa") [đo] |
| Nhãn lệch audio | 10/1.613 = 0,6% câu có tốc độ vô lý (vd 35 ký tự trong 1,34 s = 26 ký tự/giây) → loại bằng ràng buộc CTC [đo] |
| License | CC-BY-4.0 theo mirror; gốc VLSP có thể chỉ research [card, chưa kiểm] |

Nhận xét: nhãn sạch tương đối (0,6% lệch rõ), nhưng **phong cách nhãn khác FLEURS** (số, tên riêng) nên WER trên FLEURS bị phạt ở các từ đó. Không có nhãn người nói, nên không chia dev theo người nói được.

### 1.2 InfoRe1 25h (STT + TTS)

| Chỉ số | Giá trị |
|---|---|
| Người nói | **1 nữ**, người đọc thuê [card]. Phân bố cao độ hẹp, nhất quán với một người, nhưng em chưa kiểm bằng x-vector |
| SR gốc | 44,1 kHz mono PCM16 [đo] |
| STT | 14.935 → 14.933 câu, 24,92 h [đo] |
| TTS: QC âm thanh | 14.935 → **13.742** câu (21,80 h); loại 1.192 vì SNR < 20 dB, 1 câu im lặng [đo] |
| TTS: kiểm nhãn ASR | 13.742 → **13.044** câu (≈ 20,7 h); loại 698 câu có CER > 5% [đo, PhoWhisper-medium] |
| CER trung bình | **0,90%**; 75,0% câu khớp hoàn toàn; 83,9% ≤ 2%; 94,9% ≤ 5%; 99,2% ≤ 10% [đo] |
| Độ dài p1 / p50 / p99 | 2,95 / 5,49 / 9,89 s [đo] |
| SNR (theo VAD) p1 / p50 / p99 | 20,8 / 30,3 / 51,1 dB [đo] |
| Tốc độ nói p1 / p50 / p99 | 2,76 / 3,58 / 4,50 âm tiết/s [đo] |
| LUFS gốc p1 / p50 / p99 | −19,8 / −17,5 / −15,7 [đo] (đã chuẩn về −20) |
| Phổ gốc | 99% năng lượng dưới 6,2 kHz (trung vị), 99,9% dưới 12 kHz; năng lượng > 8 kHz chiếm 0,6% [đo, n = 40] |

**Ngữ điệu (n = 908 câu) [đo]:** cao độ trung vị 13,5 ± 1,3 semitone trên 100 Hz (≈ 218 Hz); biên độ (p90 − p10) 7,2 ± 2,6 st; tốc độ log 1,33 ± 0,10 (≈ 3,8 âm tiết/s); độ lệch chuẩn năng lượng giữa các câu 0,73 dB.
Ý nghĩa: **cao độ trung bình giữa các câu chỉ dao động ±1,3 st** → giọng đọc đều. Điều này khớp cảm nhận "robot" của anh Mạnh và là **tính chất của bộ dữ liệu**, không phải lỗi pipeline
(file gốc chưa xử lý cho kết quả ASR y hệt bản đã xử lý; phổ bình thường).

So sánh cao độ giữa các bộ đọc (đo trên 60 câu mỗi bộ, độ lệch chuẩn cao độ / biên độ p5–p95) [đo]:

| Bộ | F0 std (st) | Biên độ p5–p95 (st) | Tỉ lệ có thanh |
|---|---|---|---|
| InfoRe1 | 3,80 | 10,2 | 0,63 |
| FLEURS-vi | 3,48 | 10,1 | 0,51 |
| VIVOS | 2,84 | 8,4 | 0,56 |

(Hai lần đo cao độ của InfoRe1, ở đây và ở bảng mục 2, khác mẫu và cách tính nên lệch nhẹ: 3,80 và 3,38.)

→ InfoRe1 **không** kém đa dạng cao độ hơn các bộ đọc khác; "đều" là đặc trưng của **đọc văn bản** nói chung. Muốn tự nhiên hơn phải chuyển sang **nói chuyện / kể chuyện / hội thoại**, không phải đổi bộ đọc khác.

### 1.3 Bộ chỉ dùng để đánh giá STT

| Bộ | Quy mô | Ghi chú |
|---|---|---|
| FLEURS-vi test | 857 câu | đọc câu Wikipedia, có chữ số và tên nước ngoài nguyên gốc; CC-BY-4.0 |
| VIVOS test | 760 câu, 19 người | nhãn CHỮ HOA; CC-BY-NC-SA → chỉ đánh giá |

---

## 2. So sánh trực tiếp các nguồn ứng viên cho TTS (đo cùng một bộ chỉ số)

Phương pháp: mỗi bộ lấy `n = 40` câu ngẫu nhiên **từ một shard**, đo bằng cùng một script (log: `results/data/tts_dataset_candidates_analysis.log`):
cao độ (parselmouth), SNR theo VAD, CER do PhoWhisper-**small** (yếu hơn bản medium dùng khi lọc InfoRe), số người nói trong mẫu.
**Cảnh báo cỡ mẫu:** n = 40 và một shard, nên đây là **chỉ báo**, không phải thống kê của cả bộ. VietBibleVox chưa đo xong (phiên làm việc bị khởi động lại).

| Bộ | SR | Dài TB (s) | F0 std (st) | Biên độ p10–p90 (st) | Tốc độ (âm tiết/s) | SNR (dB) | Năng lượng 99,5% dưới (Hz)† | CER trung vị | Câu CER > 5% | Số người nói trong mẫu |
|---|---|---|---|---|---|---|---|---|---|---|
| InfoRe1 | 44,1 k | 5,96 | 3,38 | 5,93 | 3,50 | 29,5 | 7.440 | 0,00 | 5% | 1 (theo card) |
| InfoRe2 (sách nói) | 16 k | 5,18 | 4,08 | 10,27 | 4,06 | 26,6 | **2.203** | 0,08 | **60%** | không biết‡ |
| vi-podcast tier A | 24 k | 10,85 | **4,97** | **13,20** | 4,28 | **34,4** | 4.008 | 0,02 | 12% | **2** |
| VieNeu-140h (mirror) | 24 k | 5,94 | 3,60 | 9,51 | 3,48 | **53,0** | 4.061 | 0,03 | **38%** | 34 |

‡ Bộ này không có nhãn người nói; script gán một nhãn chung nên **không đếm được**. Card ghi nhiều người đọc.

† Cột này là "tần số chứa 99,5% năng lượng", **không phải băng thông thật**: lời nói vốn dồn năng lượng dưới ~4 kHz nên chỉ so sánh được tương đối. Giá trị **2,2 kHz** của InfoRe2 thấp bất thường (nhiều khả năng bị lọc thông thấp / nén mạnh) và là dấu hiệu cần nghe.

Đọc bảng:

1. **vi-podcast** có cao độ nhấn nhá nhiều nhất (biên độ 13,2 st so với 5,9 của InfoRe1; cả 2 cùng đo p10–p90) và SNR tốt, nhưng **mẫu chỉ có 2 người nói dù card ghi 483**. Đếm thêm trên các dòng tier A dài ≥ 10 s của shard: người nói nhiều nhất có 134 dòng, người nhì 92 dòng.
   → **shard này bị vài người nói áp đảo**; muốn đa giọng phải **lấy mẫu rải qua nhiều shard** (176 file), không dùng 1 shard.
2. **vi-podcast chủ yếu là người đọc truyện kiếm hiệp / tiên hiệp** (câu mẫu: "La Trinh đang suy nghĩ thì nghe thấy tiếng nói của thiên tôn…"). Bộ dò âm thanh (AST) còn chấm lớp "Speech synthesizer" 0,52 / 0,21 / 0,07 trên 3 mẫu → **[giả thuyết]** một phần có thể là giọng máy hoặc kể chuyện rất đều. Chưa nghe, chưa kết luận.
3. **VieNeu-140h**: rất sạch (SNR 53 dB, đã denoise), 34/40 người khác nhau (đúng là đa giọng), nhưng **38% câu CER > 5%**. Hai khả năng: (a) nhãn gốc lệch; (b) bản `text` chưa chuẩn hoá nên so với đầu ra ASR đã chuẩn hoá bị phạt. **Chưa phân biệt được.** Dù sao phải lọc bằng ASR trước khi dùng. License: **NC** (mirror tự gắn Apache là sai).
4. **InfoRe2**: 60% câu CER > 5% (nhãn tự động, card cũng ghi "cần sửa chính tả"), 16 kHz, phổ dồn thấp bất thường → **không phù hợp cho TTS**, tạm dùng cho ASR sau khi lọc.
5. **Không bộ nào cho thấy "nói chuyện tự nhiên giữa người với người"**: vi-podcast là kể truyện, VieNeu-140h là giọng đọc đa người nhưng đã làm sạch. Vấn đề này dẫn tới mục 3.

---

## 3. Khoảng trống lớn nhất: thẻ không lời và cảm xúc

### 3.1 Phương pháp pilot (`scripts/tag_pilot/`, `docs/TAG_PILOT.md`)
- Bộ dò: AST `MIT/ast-finetuned-audioset-10-10-0.4593` (527 lớp AudioSet), sigmoid, ngưỡng 0,2.
- Gom các lớp: Laughter/Giggle/Chuckle/Snicker/Belly laugh → cười; Breathing/Gasp → thở; Sigh → thở dài; Cough → ho; Sniff → sụt sịt; Throat clearing → hắng giọng; Crying/Whimper → khóc; Whispering → thì thầm.
- Cửa sổ 10 s (= đầu vào gốc của AST), bước nhảy 5 s với audio dài; mỗi clip một cửa sổ với clip ngắn. Tốc độ ≈ 9× thời gian thực trên 4 vCPU.
- **Lỗi đã sửa:** bản đầu dùng cửa sổ 1 s đệm thành 10 s, chậm gấp ~10 lần và không lưu dần nên mất kết quả khi tiến trình chết.

### 3.2 Kết quả [đo] (ngưỡng 0,2; trong ngoặc: số lần mỗi giờ audio)

| Nguồn | Giờ quét | Cười | Thở | Thở dài | Hắng giọng | Ho / khóc / thì thầm |
|---|---|---|---|---|---|---|
| vi-podcast (tier A, 1 shard) | 1,14 | 0 | 0 | 0 | 0 | 0 |
| YODAS2-Sidon vi000 (8 video đầu của mẫu 400 MB) | 2,34 | 0 | 5 (2,1) | 136 (58)‡ | 0 | 0 |
| LEMAS vi000 (1.944 clip) | 2,00 | 1 (0,5), p = 0,22 | 5 (2,5) | 0 | 3 (1,5) | 0 |

‡ **106/136 nằm trong một video**, trong đó 72 cửa sổ liền nhau → một đoạn âm thanh kéo dài bị gán nhãn, không phải 106 tiếng thở dài rời. Bỏ video đó còn ~30 cửa sổ, chưa kiểm chứng.

Phát hiện phụ [đo]: **358/1.944 clip LEMAS (18%) có nhạc nền** (Music ≥ 0,3). YODAS2-Sidon chỉ 9 cửa sổ có nhạc ở p ≥ 0,3 (khớp với việc bộ khôi phục lọc nhạc).

### 3.3 Điều có thể và không thể kết luận
- **Có thể:** trong ~5,5 giờ đã quét, **tiếng cười thật gần như không có** (≤ 0,5 lần/giờ), khóc/ho/thì thầm bằng 0. Để có 5–20 giờ tiếng cười cần quét **hơn 10.000 giờ** → ước lượng trước đó "5–20 giờ cười từ 500–600 giờ dữ liệu" **sai nhiều lần**.
- **Không thể:** kết luận "tiếng Việt không có tiếng cười trong dữ liệu". Hạn chế:
  1. AST train trên tiếng Anh, **chưa có đối chứng dương** (chưa có clip tiếng cười thật đã biết để thử bộ dò).
  2. Mẫu nhỏ (5,5 giờ), một shard mỗi nguồn, có thể không đại diện.
  3. YODAS2-Sidon **đã qua bộ khôi phục giọng** — **[giả thuyết]** nó xoá âm thanh không lời; LEMAS (chưa khôi phục) cũng gần 0 nên giả thuyết này **chỉ giải thích một phần**.
  4. Podcast/sách nói vốn ít cười hơn hội thoại, phim, talkshow.
- **Thẻ có sẵn của VieNeu:** câu có `[cười]` cho xác suất Laughter = 0,00 ở mọi cửa sổ 1 s trong khi ASR chép ra "hê hê". **[giả thuyết]** model đọc thành chữ chứ không phát ra tiếng cười. Cần **tai người** xác nhận.

### 3.4 Dữ liệu cảm xúc có thể dùng được (từ `Reference/Speech data - Danh mục tải tự do…`)

| Loại | Bộ | Quy mô | License | Dùng được cho |
|---|---|---|---|---|
| Tiếng Việt, SER | ViSEC | 5.280 clip ≈ 3,2 h, 147 người, 4 lớp (neutral 1.507, angry 1.466, happy 1.228, sad 1.079), **không transcript** | mirror CC-BY-4.0, gốc không rõ | train/đánh giá bộ nhận dạng cảm xúc (SER), không train TTS |
| Tiếng Việt, nhãn giả | `qdovan03/vietspeech-emo` | 20.000 câu ≈ 22,7 h | không ghi | cho thấy emotion2vec **lệch về "happy" 37%** trên tiếng Việt → phải hiệu chỉnh |
| Ngoại ngữ, sạch TM | CREMA-D (en, ODbL), JL-Corpus (en NZ, CC0), Thorsten-emotional (de, CC0), EmoDB 2.0 / Emozionalmente (CC-BY) | nhỏ, kịch bản cố định | ✅ | tiếng cười/cảm xúc xuyên ngôn ngữ, **ghép** vào câu Việt hoặc huấn luyện SER |
| Ngoại ngữ, NC | RAVDESS, TESS, EmoV-DB, Expresso, EARS | | ❌ | chỉ research |

Kết luận không đổi: **không có bộ TTS cảm xúc tiếng Việt nào vừa mở, vừa có transcript, vừa đủ lớn, vừa dùng thương mại được.**

---

## 4. Ma trận nguồn × mục đích (cập nhật theo số đo)

Điểm 0–5 (5 = tốt nhất). **Điểm đã hiệu chỉnh theo phép đo ở mục 2–3**, khác bảng ban đầu trong note danh mục.

| Nguồn | Giờ [card] | TTS một giọng | TTS đa giọng / clone | ASR | Biểu cảm / thẻ không lời | License | Ghi chú đo được |
|---|---|---|---|---|---|---|---|
| InfoRe1 | 25 | 3 | 0 | 2 | 0 | ❓ | đọc đều; sạch |
| InfoRe2 | 415 | 1 | 1 | 2 | 1 | ❓ | 60% CER > 5%, 16 kHz, phổ dồn thấp |
| vi-podcast | 392 | 2 | **3** | 3 | **1** | ❓ | **0 sự kiện không lời**, shard bị 2 người áp đảo, chủ yếu đọc truyện, nghi một phần giọng máy |
| VieNeu-140h (mirror) | 140,7 | 2 | **4** | 2 | 1 | ❌ NC | 193 người, rất sạch (SNR 53), 38% CER > 5% (chưa rõ nguyên nhân) |
| YODAS2-Sidon vi | ~10.000 | 0 | 3 | 2 | **0–1** | ⚠️ | đã khôi phục; 0 cười; nghi mất âm thanh không lời |
| LEMAS vi | 6.600 | 0 | 2 | 4 | 1 | ⚠️ | 18% có nhạc nền; có word timestamp + score; 16 kHz |
| VietBibleVox | ~66 | **4** | 1 | 2 | 0 | ✅ SA | một giọng nam 48 kHz; **chưa đo cùng bộ chỉ số** |
| VietSpeech-opus | 1.144 | 0 | 1 | 4 | 1 | ⚠️ | Apache; Opus 32 kbps |
| vi-asr-relabel | ~373 | 1 | 1 | **5** | 0 | theo nguồn | nhãn tốt nhất có hoa, dấu câu |
| CREMA-D, JL, Thorsten… | nhỏ | 0 | 0 | 0 | **2** (xuyên ngôn ngữ) | ✅ | cười/cảm xúc thật nhưng ngoại ngữ, kịch bản cố định |

Giảm điểm biểu cảm của vi-podcast, YODAS2-Sidon, LEMAS so với bản đầu là **hệ quả trực tiếp của phép đo** ở mục 3.

---

## 5. Bài toán dữ liệu của mục tiêu "tốt hơn VieNeu"

Mục tiêu của anh Mạnh: giọng tự nhiên, clone từ ~10 s, có thẻ cảm xúc (<cười>, <khóc>, …), chạy CPU, **tốt hơn VieNeu**.

| Cần | VieNeu đã có | Mình cần thêm | Có nguồn free chưa? |
|---|---|---|---|
| Tiếng Việt chuẩn, clone zero-shot | ~10.000 giờ [card] | Không cần train lại, **xuất phát từ trọng số của họ** | ✅ trọng số Apache-2.0 |
| Nhiều giọng, nhiều phong cách | nhiều | 500–3.000 giờ đa giọng đã lọc để không quên và mở rộng | ⚠️ vi-podcast, VieNeu-140h, YODAS2/LEMAS lọc (đều có vấn đề ở mục 2) |
| **Tiếng cười, thở, ho, khóc thật** | chỉ 3 thẻ, chất lượng chưa xác nhận | ≥ 50–100 giờ có sự kiện thật [**ước lượng thô, không có căn cứ chắc**] | ❌ **không có** (mục 3) |
| **Cảm xúc có nhãn** | không có nhãn vui/buồn/giận | vài chục đến vài trăm giờ có nhãn đáng tin | ❌ chỉ ngoại ngữ nhỏ, hoặc sinh bằng model "thầy" |
| Đánh giá | không công bố MOS/WER/SIM | bộ test cố định + nghe thử có mù | ❌ phải xây |

**Hệ quả:** "tốt hơn VieNeu" chỉ có thể xảy ra **ở trục cảm xúc và thẻ không lời**, và trục đó **đang thiếu đúng loại dữ liệu**. Ba đường lấp khoảng trống (mỗi đường có chi phí khác nhau, anh Mạnh chọn):

| Đường | Cách | Ưu | Nhược / rủi ro |
|---|---|---|---|
| 1. Thu/mua dữ liệu thật | thu âm hoặc mua dữ liệu có cảm xúc, tiếng cười | chất lượng cao nhất | tốn tiền/công; anh Mạnh đã nói không muốn tự thu |
| 2. Model "thầy" sinh dữ liệu | Higgs TTS 3, Fish S2 Pro (có thẻ, có tiếng Việt, **chỉ NC**), hoặc VoxCPM2 / VieNeu (Apache, thẻ yếu) rồi lọc bằng SER + ASR + độ giống giọng | rẻ, kiểm soát được | bản có thẻ mạnh chỉ **NC** → mô hình cuối chỉ dùng research/portfolio; học cả lỗi của "thầy" |
| 3. Ghép clip thật | chèn tiếng cười/thở từ bộ CC0/ODbL vào câu Việt, gắn thẻ | rẻ nhất, thương mại được | tiếng cười ghép **lệch người nói** → model có thể học tiếng cười "không cùng giọng"; chất lượng thô |

---

## 6. Rủi ro dữ liệu (checklist)

| Rủi ro | Biểu hiện đã gặp / nguy cơ | Cách giảm |
|---|---|---|
| **Số đo sai** | WADA-SNR loại nhầm câu sạch; tràn số int8 làm 440/996 câu bị loại | kiểm chéo bằng phép đo độc lập; tỉ lệ loại bất thường = có bug |
| **Dữ liệu tổng hợp lẫn vào** | nhiều bộ "dataset" thực chất là giọng TTS (Dolly-Audio, ccvoice, Ngọc Huyền, ShiniChien, ...); vi-podcast nghi một phần | không dùng nguồn tổng hợp; chấm "Speech synthesizer" của AST, nghe thử |
| **Mất âm thanh không lời** | YODAS2-Sidon qua bộ khôi phục (giả thuyết) | dùng bản chưa khôi phục để so (chưa làm) |
| **Nhạc nền** | 18% clip LEMAS | tách nhạc (Demucs) hoặc loại theo điểm Music |
| **Đa giọng giả** | vi-podcast shard 0: 2 người nói áp đảo | lấy mẫu rải qua shard, chia theo `speaker_id` |
| **Nhãn lệch** | VLSP 0,6% lệch audio; VieNeu-140h 38% CER > 5% (chưa rõ nguyên nhân); InfoRe2 60% | lọc bằng ASR khác với ASR dùng để chấm (tránh tự chấm điểm mình) |
| **Chồng lặp giữa bộ** | cùng nguồn YouTube: YODAS, YODAS2, Sidon, LEMAS, VietSpeech, InfoRe2, vi-podcast… | dedupe theo `video_id` và hash văn bản; không cộng dồn giờ của các bộ chồng nhau |
| **Rò rỉ train/test** | dev chia theo câu; VLSP không có nhãn người nói | test chỉ dùng FLEURS/VIVOS (người nói ngoài tập train); chia theo người nói khi có nhãn |
| **License** | mirror gắn sai license (VieNeu-140h gắn Apache, gốc NC); mirror lách điều khoản gốc (mọi bản PhoAudiobook) | tin license **gốc**, không tin license mirror; tách 2 checkpoint "sạch thương mại" và "research" |
| **Quyền giọng nói người thật** | clone giọng người thật (podcast, YouTube) | chỉ clone giọng có đồng ý; ghi vào ToS sản phẩm |
| **Phong cách một chiều** | sách nói/kể truyện đều giọng | trộn thêm hội thoại, nhưng nguồn hội thoại free lại thiếu tiếng cười (mục 3) |

---

## 7. Đặc tả pipeline dữ liệu cho hướng "fine-tune VieNeu + thẻ" (đề xuất, chưa chạy)

Đây là **đặc tả để duyệt**, không phải kết quả. Ngưỡng là điểm xuất phát, phải chỉnh theo phân phối đo thật.

1. **Chọn nguồn theo shard rải đều**, ghi `source`, `video_id`/`speaker_id`, `license_source` cho từng dòng.
2. **QC thô:** clipping, im lặng, 1–15 s (LM codec: nên 3–15 s), loại câu có nhạc (Music ≥ 0,3 hoặc dùng Demucs).
3. **Chia câu giữ âm thanh không lời:** VAD **không cắt sát** (chừa ≥ 150–300 ms), **không denoise mạnh** (làm mất hơi thở, tiếng cười) — ngược với pipeline sạch của InfoRe.
4. **ASR kép + lọc:** hai ASR khác họ (một Whisper-họ như PhoWhisper, một CTC như Zipformer/ChunkFormer/Parakeet); giữ câu CER giữa hai ASR ≤ 3% hoặc CER với nhãn ≤ 5%. Dùng **ASR thứ ba khác** để đánh giá cuối.
5. **Phát hiện sự kiện không lời:** AST (đã có) + một bộ dò thứ hai độc lập (CLAP `laion/larger_clap_general`, PANNs) → **bỏ phiếu**; ngưỡng ban đầu AST ≥ 0,3 & CLAP ≥ 0,3. **Bắt buộc có đối chứng dương trước** (xem mục 8).
6. **Căn vị trí thẻ vào từ:** forced alignment (word timestamp) rồi chèn `[cười]`… đúng vị trí trong văn bản.
7. **Người nghe duyệt:** ≥ 50 mẫu mỗi lớp thẻ, tính độ chính xác thật của nhãn trước khi train.
8. **Mã hoá:** codec của model nền (MOSS-Audio-Tokenizer-Nano, 48 kHz, 12,5 frame/s, 16 codebook); lưu `phones` (sea-g2p), `codes`, x-vector; kiểm bug "mã cuối giống nhau" (issue #198 của VieNeu).
9. **Trộn tập:** gợi ý [ước lượng] ~40% đoạn có thẻ, 40% hội thoại không thẻ, 20% giọng đọc sạch để chống quên.
10. **Tập test cố định** (trước khi train): 200–500 câu khó (số, viết tắt, tiếng Anh chen, câu dài) + ~20 giọng tham chiếu ngoài tập train + 30–50 câu mỗi thẻ.

Sản lượng dự kiến: **không ước lượng**, vì số đo ở mục 3 bác bỏ ước lượng cũ; phải đo lại sau bước 5.

---

## 8. Phải đo thêm trước khi tốn GPU (để khỏi làm đi làm lại)

Thứ tự theo giá trị thông tin, rẻ trước:

| # | Phép đo | Giải quyết câu hỏi | Chi phí |
|---|---|---|---|
| 1 | **Anh Mạnh nghe** câu `[cười]`, `[thở dài]`, `[hắng giọng]` của VieNeu gốc (trang nghe thử) | thẻ có sẵn là tiếng thật hay đọc "hê hê"? Quyết định có cần fine-tune thẻ không | 5 phút của anh |
| 2 | **Đối chứng dương cho bộ dò**: tạo/tìm vài clip tiếng cười thật (CREMA-D, JL-Corpus) đưa qua AST, CLAP | bộ dò có phát hiện được không? kết quả "0" ở mục 3 có tin được không | CPU, ~1 giờ |
| 3 | Dò lại bằng **bộ dò thứ hai (CLAP/PANNs)** trên cùng 5,5 giờ | AST có bỏ sót không | CPU, ~1–2 giờ |
| 4 | So **YODAS2 gốc (chưa khôi phục) với bản Sidon** trên cùng video | giả thuyết "Sidon xoá âm thanh không lời" | CPU + tải ~GB |
| 5 | Đếm người nói thật của vi-podcast **qua nhiều shard** (metadata.csv) | 483 người nói có thật dùng được không | CPU, nhẹ |
| 6 | Chạy lại mục 2 cho **VietBibleVox** và cho **nhiều shard** mỗi bộ, n ≥ 200 | thay chỉ báo n = 40 bằng thống kê | CPU, ~1–2 giờ |
| 7 | DNSMOS / UTMOS cho các bộ ứng viên | chất lượng cảm nhận, không chỉ SNR | CPU/GPU nhẹ |
| 8 | Nghe thử 20–30 mẫu ngẫu nhiên mỗi bộ ứng viên (anh Mạnh) | phát hiện giọng máy, artifact mà số đo không thấy | giờ của anh |
| 9 | **Pilot fine-tune 1–2 giờ GPU free**: không NaN, vẫn clone, thẻ mới có tác động, xuất ra chạy được CPU | toàn bộ đường train có thông không | 1–2 giờ T4 |

Chỉ khi 1, 2, 9 đạt mới nên cân nhắc thuê GPU.

---

## 9. Sổ ghi sai sót (để không lặp lại)

| Điều em từng nói / làm | Thực tế | Bài học |
|---|---|---|
| Bắt đầu train trước khi research dữ liệu xong | Train trên 117 giờ không dẫn tới mục tiêu | Research dữ liệu và kiến trúc **trước**, train **sau** |
| WADA-SNR dùng để lọc | Loại nhầm câu sạch | Mọi bộ lọc phải kiểm chéo bằng phép đo độc lập |
| Tỉ lệ loại 440/996 sau khi đổi SNR | Do tràn số `int8` | Tỉ lệ loại bất thường = có bug |
| "InfoRe2 nên dùng cho TTS kể chuyện" (điểm 2 trong bảng ban đầu) | 60% câu CER > 5%, 16 kHz, phổ dồn thấp | Điểm phải dựa trên đo, không dựa trên mô tả |
| "5–20 giờ tiếng cười từ 500–600 giờ" | Đo được ≤ 0,5 lần/giờ | Ước lượng không có đo = giả thuyết |
| "vi-podcast 483 người nói, tốt cho đa giọng" | Một shard chỉ có 2 người áp đảo | Quy mô trên card ≠ độ đa dạng trong shard đang dùng |
| Ước lượng tốc độ train STT | Chậm 14 lần vì đọc từ ổ mạng | Đo vài bước thật trước khi ước lượng |
| Thí nghiệm STT chạy trùng 2 bản | Chiếm cả 2 suất GPU | Kiểm tra trước khi đẩy tự động |
| Một lượt TTS bị huỷ | Mất output (Kaggle không giữ khi huỷ) | Chia phiên ngắn, lưu mẫu định kỳ |

---

## 10. Từ khoá dữ liệu (rút gọn)

Bảng từ khoá đầy đủ toàn lĩnh vực nằm ở note SecondBrain **"Speech AI - Từ khoá và khái niệm toàn lĩnh vực"**. Riêng phần dữ liệu:

`gated` · `tier A/B` · `speaker_id` · `diarization` · `VAD` · `forced alignment` · `word timestamp` · `re-ASR filtering` · `CER round-trip` ·
`pseudo-label` · `dedupe` · `train/dev/test leakage` · `LUFS` · `SNR` · `DNSMOS` · `UTMOS` · `bandwidth` · `denoise / dereverb` · `speech restoration (Sidon)` ·
`source separation (Demucs)` · `non-verbal vocalization` · `audio event detection (AST, PANNs, CLAP)` · `SER (emotion2vec)` · `emotion intensity` ·
`synthetic data / teacher distillation` · `license: CC-BY / CC-BY-SA / CC-BY-NC / ODbL / OpenRAIL-M / Apache-2.0` · `mirror license mismatch` ·
`codec tokens (RVQ)` · `phoneme (IPA, sea-g2p)` · `text normalization (NFC, kiểu bỏ dấu, số, viết tắt)`.
