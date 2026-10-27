# Hướng dẫn sử dụng Swish

Swish dự đoán từng trận NBA trước giờ bóng lăn: xác suất thắng của đội nhà, chênh lệch điểm kỳ vọng (spread) và tổng điểm (total). App chạy trên máy bạn dưới dạng web app, mở bằng trình duyệt.

## Model hiện tại

Model v6 dùng **Kalman-Elo**:

- Mỗi đội có một rating tính bằng điểm (mạnh hơn đội trung bình bao nhiêu điểm), kèm độ bất định.
- Sau mỗi trận, rating thay đổi nhiều hay ít tuỳ model đang chắc chắn về đội đó tới đâu. Đầu mùa hay sau quãng nghỉ dài, rating thay đổi nhanh. Giữa mùa, rating thay đổi chậm.
- Sang mùa mới, rating bị kéo 20% về mức trung bình, vì đội hình có xáo trộn.
- Trận thứ hai trong hai đêm liên tiếp (back-to-back) bị trừ 1 điểm.
- Mỗi cầu thủ chấn thương bị trừ điểm theo mức độ quan trọng và tình trạng (out, doubtful, questionable, day-to-day).

Kết quả kiểm tra trên 8.874 trận thật từ mùa 2017-18 đến 2023-24. Model chỉ được thấy các trận đã đá trước trận cần đoán.

| Model | Log loss (thấp hơn là tốt hơn) | Đoán đúng |
|---|---:|---:|
| **v6 Kalman-Elo (hiện tại)** | **0.628** | **64,5%** |
| Elo cũ | 0.634 | 64,0% |
| v5 XGBoost (trước đây) | 0.652 | 62,6% |

## Cài đặt và chạy app

**Bạn cần có:**
- Python 3.11 trở lên. Kiểm tra bằng lệnh `python --version`.
- Git.
- Mạng Internet bình thường. NBA.com hay chặn VPN và máy chủ cloud, nên nếu tải dữ liệu bị timeout thì hãy tắt VPN.

**Tải code về (chỉ làm một lần):**

```bash
git clone https://github.com/bigboybigbet/Swish.git
cd Swish
```

**Chạy app (mỗi lần dùng):**

| Hệ điều hành | Cách chạy |
|---|---|
| Windows | Bấm đúp vào file `start.bat` |
| macOS / Linux | Chạy `./start.sh` |

**Bạn không phải cài lại thư viện mỗi lần chạy.**
- Lần chạy đầu tiên, launcher tạo môi trường ảo `.venv` và cài thư viện. Bước này mất vài phút và chỉ xảy ra một lần.
- Từ lần thứ hai trở đi, app khởi động ngay. Launcher so `requirements.txt` với lần cài trước và chỉ cài lại khi file này thay đổi, ví dụ sau khi `git pull` về bản có thêm thư viện.

Đợi terminal in dòng `Ready: open http://127.0.0.1:8000`, trình duyệt sẽ tự mở. Giữ cửa sổ terminal mở trong suốt lúc dùng app. Bấm `Ctrl+C` để tắt.

**Cập nhật bản mới:** chạy `git pull` rồi khởi động như bình thường.

**Tuỳ chọn khi chạy:**
- `start.bat --port 8010`: chạy ở cổng 8010 nếu cổng 8000 đang bận.
- `./start.sh --no-browser`: không tự mở trình duyệt.

<details>
<summary>Cài thủ công (không dùng launcher)</summary>

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell: .venv\Scripts\Activate.ps1)
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt    # chỉ một lần, và khi requirements.txt thay đổi
python -m server                   # mỗi lần chạy, sau khi đã kích hoạt .venv
```

Nếu PowerShell báo không cho chạy script, chạy lệnh này một lần: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
</details>

## Lần đầu: dựng model

1. Vào trang **Model**.
2. Tick ô **Force full rebuild** rồi bấm **Train**.
3. Đợi thanh tiến độ chạy xong, mất vài phút.

Trong lúc này app tải các mùa 2023-24 đến 2026-27, tính rating cho cả 30 đội qua từng trận, và lưu vào file `team_strength.json`. Nếu chưa Train lần nào, app chỉ dự đoán bằng Elo đơn giản và hiện nhãn "Elo only". Sau khi Train xong, nhãn sẽ đổi thành "Kalman-Elo v6".

## Dùng hằng ngày

**Xem lịch hôm nay:** trang chủ (**Slate**) liệt kê mọi trận trong ngày, tính theo ngày giờ miền Đông Mỹ. Dùng nút ‹ › hoặc phím mũi tên để chuyển ngày. Bấm **Run predictions**, mỗi thẻ trận sẽ hiện dần: % thắng, spread, total, rating hai đội và tỷ lệ của nhà cái. Phía trên có bộ đếm kèo VIP (từ 65% trở lên và hai engine cùng đồng ý) và các kèo có lợi thế (value) so với nhà cái, bấm vào để lọc. Trận đã đá xong được đánh dấu đúng/sai. Bấm vào một thẻ để mở phân tích chi tiết.

**Ratings:** bảng xếp hạng 30 đội theo rating Kalman-Elo (số điểm mạnh hơn đội trung bình, ± độ bất định), thay đổi 7 ngày và đường xu hướng cả mùa. Tab Players hiện mức ảnh hưởng của từng cầu thủ, dùng để tính chấn thương.

**Accuracy:** chấm điểm mọi trận bằng dự đoán đưa ra *trước* giờ bóng lăn: tỷ lệ đoán đúng, log loss, Brier, biểu đồ calibration (model nói 70% thì đội đó có thắng khoảng 70% không), tỷ lệ đúng theo mức tự tin và theo từng tuần.

**Dự đoán một trận:** vào trang **Game** và chọn đội nhà, đội khách. App hiển thị:
- xác suất thắng và spread;
- tổng điểm kèm khoảng dao động;
- rating của hai đội kèm độ bất định;
- danh sách chấn thương lấy từ ESPN;
- bảng tách xác suất thành từng phần: chênh lệch rating, lợi thế sân nhà, back-to-back, chấn thương.

Nếu bạn nhập tỷ lệ cược (moneyline) của nhà cái, app tính thêm lợi thế (edge) và mức cược Kelly đã giới hạn.

**Cập nhật rating sau khi có kết quả mới:** vào trang **Model** và bấm **Grade new games**. App sẽ chấm các trận vừa đá, rồi tính lại rating từ đầu, nên không trận nào bị cộng hai lần.

**Train lại:** khoảng mỗi tuần một lần, hoặc khi thấy dự đoán có vẻ lỗi thời.

Mùa giải tự chuyển vào tháng 10 hằng năm, bạn không cần sửa gì.

## Khoá API (không bắt buộc)

Bạn nhập khoá ở trang **Settings**, hoặc chép `.env.example` thành `.env` rồi điền vào đó.

| Khoá | Dùng để làm gì |
|---|---|
| `ODDS_API_KEY` | Lấy tỷ lệ cược trực tiếp từ [The Odds API](https://the-odds-api.com). Gói miễn phí có 500 lượt mỗi tháng. Không có khoá thì app dùng tỷ lệ của ESPN nếu có. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Gửi kèo qua Telegram. |

Khoá chỉ nằm trong file `.env` trên máy bạn. File này không bao giờ được đưa lên Git, và app không gửi khoá ra trình duyệt.

## Chạy thử không cần mạng (chế độ demo)

Chế độ demo phát lại một ngày bất kỳ từ mùa 2010-11 đến 2023-24 bằng dữ liệu thật:

```bash
git clone --depth 1 https://github.com/NocturneBear/NBA-Data-2010-2024 ../nba-data-2010-2024
```

Trên Windows PowerShell:

```powershell
$env:NBA_HISTORY_DIR="..\nba-data-2010-2024"; $env:NBA_QUANT_DEMO_DATE="2024-03-15"; .\start.bat
```

Trên macOS / Linux:

```bash
NBA_HISTORY_DIR=../nba-data-2010-2024 NBA_QUANT_DEMO_DATE=2024-03-15 ./start.sh
```

## Giao diện cũ (Streamlit)

Trang Bets (ROI) của web app chưa làm xong. Trong lúc chờ, bạn vẫn dùng được app cũ, chạy cùng model mới:

```bash
.venv\Scripts\streamlit run nba_app.py     # Windows
.venv/bin/streamlit run nba_app.py         # macOS / Linux
```

## Gặp lỗi thì làm gì

| Lỗi | Cách xử lý |
|---|---|
| Trình duyệt báo "can't reach this page" hoặc trang trắng | Đợi terminal in dòng `Ready: open http://127.0.0.1:8000` rồi tải lại trang. Lần chạy đầu có thể mất tới 1 phút. Giữ cửa sổ terminal mở trong suốt lúc dùng app. |
| Train báo timeout khi kết nối stats.nba.com | Tắt VPN hoặc proxy rồi thử lại. NBA.com giới hạn số lượt gọi, nên hãy đợi khoảng 1 phút giữa các lần thử. |
| `Address already in use` | Cổng 8000 đang bận. Chạy `start.bat --port 8010` rồi mở http://127.0.0.1:8010 |
| Dự đoán hiện "Elo only" | Chưa Train. Làm theo mục "Lần đầu: dựng model". |
| Chấn thương hiện "unavailable" | Không lấy được dữ liệu từ ESPN. App vẫn dự đoán nhưng không tính chấn thương, và có ghi chú điều này. |
| `ModuleNotFoundError` | Hãy chạy bằng `start.bat` hoặc `./start.sh`: launcher dùng `.venv` và tự cài phần còn thiếu. Nếu cài thủ công, nhớ kích hoạt `.venv` trước. |

## Lưu ý

Dự đoán là xác suất, không phải điều chắc chắn. Đây không phải lời khuyên cá cược. Nếu cược, chỉ dùng số tiền bạn chấp nhận mất.
