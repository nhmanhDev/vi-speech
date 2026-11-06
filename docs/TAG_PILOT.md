# Pilot: đào tiếng cười, thở, ho… từ dữ liệu tiếng Việt tải tự do

Mục tiêu: xem có thể **đào** dữ liệu cho các thẻ không lời (`[cười]`, `[thở]`, `[ho]`, `[khóc]`…) từ các bộ free hay không.
Bộ dò: AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, 527 lớp AudioSet), cửa sổ 10 s, ngưỡng xác suất ≥ 0,2.
Script: `scripts/tag_pilot/` (chạy tiếp được, lưu dần). Tốc độ ~9× thời gian thực trên 4 vCPU.

## Kết quả (số sự kiện ở ngưỡng 0,2; trong ngoặc là số lần mỗi giờ audio)

| Nguồn | Giờ quét | Cười | Thở | Thở dài | Hắng giọng | Ho / khóc / thì thầm |
|---|---|---|---|---|---|---|
| vi-podcast (tier A, 1 shard) | 1,14 | 0 | 0 | 0 | 0 | 0 |
| YODAS2-Sidon vi000 (video YouTube, đã qua khôi phục) | 2,34 | 0 | 5 (2,1) | 136 (58) *| 0 | 0 |
| LEMAS vi000 (clip cắt từ YouTube) | 2,00 | 1 (0,5), p = 0,22 | 5 (2,5) | 0 | 3 (1,5) | 0 |

\* 106/136 nằm ở **một video** (72 cửa sổ liền nhau): một đoạn âm thanh kéo dài bị gán nhãn, không phải 106 tiếng thở dài.
Bỏ video đó còn ~30 cửa sổ chưa kiểm chứng.

LEMAS: 358/1.944 clip (18%) có nhạc nền (p ≥ 0,3). vi-podcast chủ yếu là người đọc truyện kiếm hiệp (giọng kể đều); bộ dò còn chấm
"giọng tổng hợp" 0,21–0,52 ở 2/3 mẫu kiểm, nên nghi một phần là giọng máy (chưa khẳng định).

## Kiểm bộ dò
- Nhận lời nói đúng (xác suất Speech 0,78–0,93 trên cả câu thật lẫn giọng tổng hợp).
- **Chưa có đối chứng dương** (không có clip tiếng cười thật đã biết để thử). AST được train trên tiếng Anh, có thể bỏ sót ở tiếng Việt.
- Câu có `[cười]` do VieNeu v3 Turbo sinh: xác suất "Laughter" 0,00 ở mọi cửa sổ 1 s, trong khi ASR chép ra "hê hê".
  Gợi ý: model **đọc thành chữ** chứ chưa phát ra tiếng cười thật. Cần nghe để xác nhận.

## Kết luận
- Trong ~5,5 giờ đã quét, tiếng cười thật gần như không có (≤ 0,5 lần/giờ); khóc, ho, thì thầm = 0. Ước lượng trước đó
  "5–20 giờ tiếng cười từ 500–600 giờ dữ liệu" là **quá lạc quan** so với số đo này (mẫu nhỏ, chưa đối chứng dương).
- Đào từ dữ liệu có sẵn không khả thi cho cười / khóc. Hai đường còn lại: (a) ghép clip tiếng cười thật từ dữ liệu cảm xúc free
  (CC0/ODbL, tiếng cười không phụ thuộc ngôn ngữ) vào câu tiếng Việt; (b) dùng model có thẻ làm "thầy" sinh dữ liệu rồi lọc (phần lớn chỉ phi thương mại).
