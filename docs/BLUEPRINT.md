# Bản thiết kế hệ thống Swish và cách dùng lại cho app tài chính

Tài liệu này mô tả Swish vận hành ra sao: từng lớp, từng quy trình, và lý do chọn cách làm đó. Mục tiêu là dựng một app cùng kiểu cho tài chính (cổ phiếu, quỹ, crypto) mà không phải làm lại từ đầu. Mỗi phần có một mục **Sang tài chính** nói rõ cái gì giữ nguyên, cái gì phải đổi.

Swish chạy tốt vì bốn quyết định, quan trọng hơn bất kỳ thuật toán nào:

1. **Có một thước đo duy nhất và trung thực.** Đó là benchmark walk-forward trên dữ liệu thật. Mọi thay đổi model đều phải qua thước đo này.
2. **Model đơn giản, đã kiểm chứng, thắng model phức tạp chưa kiểm chứng.** XGBoost 17 feature thua Elo thuần, và Kalman-Elo 2 tham số thắng cả hai.
3. **Mỗi lớp chỉ làm một việc, và trạng thái luôn dựng lại được từ dữ liệu gốc.** Không có chỗ nào "cộng dồn" mà có thể bị cộng hai lần.
4. **App tự vận hành.** Dữ liệu tự làm mới, rating tự cập nhật, giao diện tự refresh. Người dùng chỉ bấm chạy app.

---

## 1. Kiến trúc tổng thể

```
Nguồn dữ liệu ──► Cache ──► Engine (model) ──► Service ──► API JSON ──► Giao diện web
 (nba_api, ESPN)   (SQLite,    (team_strength,    (predict,     (FastAPI)    (HTML + Alpine
                    TTL)        features,          scan,                      + ECharts,
                                simulate)          recap, news)               không build)
                                     ▲
                       Job nền (train, ratings, scan) ── tiến độ qua SSE
```

| Lớp | Thư mục | Việc duy nhất của nó |
|---|---|---|
| Nguồn dữ liệu | `data/` | Gọi API bên ngoài, trả dữ liệu thô. Không tính toán. |
| Cache | `data/cache.py`, `server/services/espn.py` | Giữ dữ liệu theo thời hạn (TTL). Khi mất mạng thì dùng bản cũ và **báo là cũ**. |
| Engine | `engine/` | Toán thuần: dữ liệu vào, con số ra. Không đọc mạng, không biết web tồn tại. |
| Service | `server/services/` | Ghép engine với dữ liệu thành câu trả lời cho một màn hình. |
| API | `server/routers/` | Kiểm tra input, trả JSON theo một format lỗi thống nhất. |
| Giao diện | `web/` | Hiển thị. Không chứa logic nghiệp vụ. Ngưỡng (VIP, value…) do backend quyết định. |
| Job nền | `server/jobs.py` | Việc chậm (train, quét lịch, cập nhật rating) chạy nền. Có tiến độ, có nút huỷ, và chỉ một job ghi dữ liệu tại một thời điểm. |

**Lý do tách như vậy:** engine thuần nên benchmark gọi đúng cùng một đoạn code với app thật. Khi kết quả benchmark tốt thì app thật cũng tốt, không có hai phiên bản lệch nhau.

**Sang tài chính:** giữ nguyên kiến trúc. Chỉ thay nguồn dữ liệu (giá, khối lượng, báo cáo tài chính, tin doanh nghiệp) và engine (model định giá, model rủi ro, tín hiệu).

---

## 2. Dữ liệu: nguyên tắc "tại thời điểm đó"

Quy tắc số một: **mọi con số dùng để dự đoán ngày D chỉ được dùng thông tin có trước ngày D.**

Swish áp dụng quy tắc này ở mọi chỗ:
- Feature dịch lùi một trận (`shift(1)`), nên trận đấu không bao giờ thấy kết quả của chính nó.
- Rating lưu lịch sử theo ngày. Backtest một ngày trong quá khứ sẽ dùng rating **tại ngày đó**.
- Có test tự động: đổi kết quả một trận rồi kiểm tra dự đoán của chính trận đó và các trận trước không đổi (`tests/test_core.py`, `tests/test_team_strength.py`).
- Ngày luôn tính theo giờ của thị trường (US Eastern), không theo giờ máy người dùng.

**Sang tài chính:** đây là nơi dễ sai nhất, và sai ở đây làm mọi kết quả backtest vô nghĩa.

| Bẫy | Cách xử lý |
|---|---|
| Báo cáo tài chính quý 3 "của ngày 30/9" nhưng **công bố** ngày 20/10 | Luôn dùng **ngày công bố**, không dùng ngày kỳ báo cáo. |
| Giá điều chỉnh (cổ tức, chia tách) tính lại cả quá khứ | Lưu cả giá gốc lẫn hệ số điều chỉnh. Backtest dùng giá điều chỉnh, nhưng chỉ tính đến đúng ngày đó. |
| Danh sách mã hiện tại bỏ sót mã đã huỷ niêm yết (survivorship bias) | Dùng danh sách mã **tại từng thời điểm**. |
| Tin tức có giờ đăng | So giờ đăng với giờ khớp lệnh. Tin ra sau 14:45 chỉ ảnh hưởng phiên sau. |
| Múi giờ và giờ giao dịch (ATO, ATC, nghỉ trưa, ngày lễ) | Một module lịch giao dịch duy nhất, giống `clock.py` của Swish. |

Viết test "lật dữ liệu" tương tự cho mọi feature: sửa giá của ngày D, rồi kiểm tra tín hiệu của ngày D và các ngày trước không đổi.

---

## 3. Model: từ đơn giản tới phức tạp, có thước đo

### Swish đang làm gì
- **Model chính:** Kalman-Elo (`engine/team_strength.py`). Mỗi đội có một sức mạnh ẩn kèm độ bất định. Sau mỗi trận, sức mạnh được cập nhật theo "độ bất ngờ" của kết quả, có trọng số bằng hệ số Kalman: đầu mùa hoặc sau khi nghỉ dài thì độ bất định cao, nên rating đổi nhanh. Sang mùa mới, rating co 20% về mức trung bình.
- **Đầu ra:** chênh lệch điểm kỳ vọng, đổi sang xác suất thắng bằng hàm logistic 2 tham số.
- **Mô phỏng Monte Carlo** chỉ dùng để vẽ khoảng dao động của tổng điểm, không dùng cho xác suất chính.
- **Chấn thương:** trừ thẳng số điểm sức mạnh, không nhân hệ số.

### Vì sao chọn nó
Ứng viên được chọn qua benchmark (mục 4), không phải theo cảm tính:

| Model | Log loss | Chính xác |
|---|---:|---:|
| Kalman-Elo (chọn) | 0.628 | 64,5% |
| Elo thường | 0.634 | 64,0% |
| XGBoost 17 feature | 0.652 | 62,6% |

Những thứ đã thử và **bỏ** vì không vượt ngưỡng ý nghĩa thống kê: stacking nhiều loại rating (tương quan 0,98 nên không thêm thông tin), cập nhật trọng số online sau mỗi trận, và feature động lực theo bảng xếp hạng (có thật nhưng quá nhỏ). Giữ một model ít tham số thì ít chỗ để overfit.

**Sang tài chính:**
- Bộ lọc Kalman rất hợp với tài chính: beta và alpha thay đổi theo thời gian, xu hướng ẩn của giá, spread giữa hai mã đồng biến (pairs trading). Cấu trúc giữ nguyên: một trạng thái ẩn kèm độ bất định, cập nhật theo độ bất ngờ, và co về trung bình khi có thay đổi chế độ thị trường (tương đương "sang mùa mới").
- Bắt đầu bằng **baseline cực đơn giản**: mua và giữ, momentum 12-1 tháng, trung bình động. Model nào không thắng baseline sau chi phí giao dịch thì bỏ, đúng như XGBoost đã thua Elo.
- Tín hiệu tài chính yếu hơn bóng rổ nhiều. Accuracy 52–55% đã là tốt. Vì vậy ưu tiên kiểm soát rủi ro và chi phí hơn là đuổi theo độ chính xác.

---

## 4. Quy trình kiểm định: phần quan trọng nhất

`research/benchmark.py` là "toà án" của dự án. Mọi ý tưởng đều phải qua:

1. **Walk-forward:** train trên 3 mùa trước, giữ nguyên model, dự đoán toàn bộ mùa sau, lặp cho 7 mùa (8.874 trận). Không bao giờ trộn ngẫu nhiên, vì đây là dữ liệu chuỗi thời gian.
2. **Tách dev và holdout:** tinh chỉnh tham số **chỉ** trên các mùa dev (2014–17). Holdout (2017–24) chỉ chạy **một lần** ở cuối, và báo kết quả đúng như nó ra, kể cả khi tệ.
3. **Thước đo chính là log loss**, không phải accuracy. Accuracy dao động mạnh với mẫu nhỏ: mùa này accuracy giảm từ 63% xuống 61% nhưng log loss vẫn đúng 0.627.
4. **Ngưỡng thắng:** chênh lệch log loss từng trận (paired difference) phải nhỏ hơn −2 lần sai số chuẩn (SE). Dưới ngưỡng này coi như nhiễu.
5. **Audit độc lập:** một người hoặc agent khác chạy lại từ đầu và tìm cách chứng minh kết quả là giả. Lần audit `incentives` đã bắt được một lỗi dùng dữ liệu tương lai mà người xây không thấy.
6. **Calibration:** khi model nói 70% thì thực tế phải đúng khoảng 70%. Trang Accuracy vẽ đúng biểu đồ này.

**Sang tài chính:** giữ nguyên cả 6 bước, và thêm:

| Thêm | Lý do |
|---|---|
| Chi phí giao dịch, thuế, trượt giá (slippage), giới hạn biên độ, T+2 | Lợi nhuận trước phí gần như luôn đẹp hơn thực tế. |
| Purged và embargo CV (López de Prado) | Nhãn kéo dài nhiều ngày (lợi nhuận 5 ngày) sẽ chồng lên tập test nếu không làm vậy. |
| Hiệu chỉnh khi thử nhiều chiến lược (deflated Sharpe, Bonferroni) | Thử 100 chiến lược thì gần như chắc chắn có vài cái "thắng" do may mắn. |
| Chỉ số: Sharpe, max drawdown, turnover, hit rate, Brier score cho tín hiệu xác suất | Một con số không đủ mô tả rủi ro. |
| Kiểm tra theo từng chế độ thị trường (tăng, giảm, đi ngang) | Model chỉ thắng trong một chế độ là model nguy hiểm. |

---

## 5. Vận hành tự động

| Cơ chế | Swish | Tài chính |
|---|---|---|
| Dựng lại trạng thái từ đầu | Rating replay toàn bộ log mỗi lần cập nhật, nên không có trận nào bị cộng hai lần | Danh mục, NAV và tín hiệu dựng lại từ nhật ký giao dịch, không cộng dồn |
| Tự cập nhật | Server kiểm tra mỗi 30 phút. Nếu thiếu kết quả hôm qua thì tự tải và dựng lại, tối đa 3 tiếng một lần | Job cuối ngày sau giờ đóng cửa; job trong phiên theo lịch giao dịch |
| Cache theo trạng thái | 30 giây khi có trận đang đá, 60 giây trong ngày, 24 giờ khi đã kết thúc | Vài giây trong phiên, theo ngày ngoài phiên |
| Mất mạng | Dùng bản cache cũ và báo là cũ, không im lặng | Giống vậy. Tuyệt đối không giao dịch trên dữ liệu cũ mà không cảnh báo |
| Ghi file an toàn | Ghi ra file tạm rồi đổi tên (`data/atomic.py`), nên không có file ghi dở | Bắt buộc với sổ lệnh và vị thế |
| Một job ghi tại một thời điểm | Khoá ghi trong `server/jobs.py` | Bắt buộc. Hai job cùng sửa danh mục là thảm hoạ |
| Chế độ demo | Phát lại một ngày trong quá khứ từ dữ liệu lịch sử, không cần mạng | "Paper trading" và phát lại một phiên cũ để thử giao diện và chiến lược |

---

## 6. Giao diện và trải nghiệm

Nguyên tắc rút ra (theo UI/UX Pro Max và taste-skill):
- **Màu dùng cho dữ liệu, không dùng để trang trí.** Màu đội chỉ xuất hiện trên dữ liệu của đội đó.
- **Mỗi màn hình trả lời một câu hỏi chính** trước khi hiện mọi thứ khác: Slate là "hôm nay kèo nào đáng chú ý", Recap là "trận này thắng thua thế nào".
- **Dữ liệu trung thực:** thiếu dữ liệu thì hiện dấu "–". Mẫu nhỏ thì cảnh báo. Không bịa con số.
- **Cảm giác mượt:** chuyển trang có hiệu ứng (View Transitions), tải trước trang khi rê chuột, khung chờ (skeleton) giữ bố cục, tôn trọng chế độ giảm chuyển động.
- **Không cần bước build:** thư viện được vendor sẵn trong repo, có kiểm hash, chạy offline.
- **Một lệnh để chạy:** `start.bat`. Lần đầu tự cài, các lần sau mở ngay.

**Sang tài chính:** các màn hình tương ứng:

| Swish | Tài chính |
|---|---|
| Slate (các trận hôm nay) | Watchlist / bảng tín hiệu hôm nay |
| Game (phân tích một trận) | Phân tích một mã: định giá, tín hiệu, rủi ro, vì sao model nói vậy |
| Recap (diễn biến trận) | Nhìn lại một lệnh hoặc một phiên: giá trong ngày, điểm vào/ra, đóng góp vào lãi lỗ |
| Ratings | Xếp hạng mã theo điểm nhân tố hoặc sức mạnh tương đối, kèm độ bất định |
| Accuracy | Theo dõi tín hiệu: calibration, hit rate, Sharpe theo thời gian |
| News (gắn cờ "ảnh hưởng ai ra sân") | Tin doanh nghiệp gắn cờ sự kiện: KQKD, cổ tức, phát hành, giao dịch nội bộ, M&A |
| Bets (Kelly có giới hạn) | Sizing vị thế: Kelly phân số, giới hạn tối đa mỗi mã và mỗi ngành |

---

## 7. Chất lượng kỹ thuật

- **Test offline**, khoảng 300 test, chạy trong 10 giây. Không test nào được gọi mạng: có fixture chặn sẵn. Lý do là trên CI một API treo đã làm test mất 120 giây.
- **CI** chạy trên Python 3.11–3.13 mỗi lần push. Chỉ push khi test trên máy đã xanh.
- **Bí mật nằm trong `.env`**, không bao giờ trả ra trình duyệt, và được che trong log (thư viện `requests` có in cả URL chứa key khi gặp lỗi).
- **Thư viện nặng chỉ nạp khi cần** (scikit-learn, xgboost), nên app khởi động nhanh gấp đôi.
- **Không sửa đè lịch sử git.** Mỗi thay đổi đi qua một PR, và CI phải xanh trước khi merge.

---

## 8. Cách làm việc với AI để dựng hệ thống

Quy trình đã dùng cho Swish, áp dụng lại được nguyên vẹn:

1. **Dựng thước đo trước tiên** (benchmark và baseline), rồi mới thử thuật toán.
2. **Thử nhiều hướng song song.** Mỗi agent làm một hướng, tinh chỉnh trên dev, chạy holdout một lần.
3. **Mỗi kết quả tốt đều có một agent khác audit**, với yêu cầu "giả định là kết quả giả cho tới khi chứng minh được là thật".
4. **Tổng hợp thành menu** để người quyết định chọn, thay vì để agent tự quyết. Bạn đã chọn "chỉ A", và đó là quyết định đúng: các hướng phức tạp hơn đều không thắng.
5. **Kiểm soát chi phí:** dừng nhánh không cần, tự làm trực tiếp những phần nhỏ, chỉ dùng workflow cho phần nghiên cứu lớn.

---

## 9. Khung repo đề xuất cho app tài chính

```
finance_app/
  config.py                lịch giao dịch, phí, thuế, giới hạn vị thế
  data/
    prices.py              giá và khối lượng (gốc + hệ số điều chỉnh)
    fundamentals.py        BCTC theo ngày công bố
    news.py                tin doanh nghiệp, gắn cờ sự kiện
    calendar.py            ngày giao dịch, phiên, ngày lễ (như clock.py)
    cache.py, atomic.py    giữ nguyên từ Swish
  engine/
    signals.py             tín hiệu (momentum, value, Kalman trend…)
    risk.py                biến động, tương quan, drawdown
    sizing.py              Kelly phân số có giới hạn
    portfolio.py           dựng lại danh mục từ nhật ký lệnh
  research/
    benchmark.py           walk-forward + purged CV + phí giao dịch
    variants/              mỗi ý tưởng một file, như Swish
  server/                  giữ nguyên khung FastAPI, jobs, SSE, bảo mật
  web/                     giữ nguyên khung giao diện; đổi các trang theo bảng ở mục 6
  tests/                   test "lật dữ liệu" cho mọi feature
  start.py / start.bat     giữ nguyên
```

**Thứ tự làm, mỗi bước có tiêu chí xong:**

1. **Dữ liệu giá và lịch giao dịch đúng.** Xong khi test "tại thời điểm đó" pass.
2. **Benchmark và baseline** (mua và giữ, momentum). Xong khi ra con số Sharpe và drawdown sau phí.
3. **Một model đơn giản.** Xong khi thắng baseline vượt ngưỡng −2 SE trên holdout và có audit độc lập.
4. **Server, job nền, trang Watchlist và trang Phân tích mã.** Xong khi chạy được bằng `start.bat`.
5. **Tin tức gắn cờ sự kiện và trang Accuracy.**
6. **Sizing và nhật ký lệnh** (tương đương trang Bets). Làm sau cùng, khi tín hiệu đã chứng minh được giá trị.

**Lưu ý cuối:** app tài chính dùng tiền thật, nên phải có cảnh báo rõ ràng, mặc định chỉ giao dịch giấy (paper trading), và không bao giờ tự đặt lệnh khi chưa có xác nhận của người dùng.
