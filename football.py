import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# --- 1. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU ---


@st.cache_data
def load_and_prep_data():
    """
    Tải dữ liệu trực tiếp từ kho open-data của giải Ngoại hạng Anh.
    Lấy dữ liệu của mùa giải 23/24 và 24/25.
    """
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    ]

    dfs = []
    for url in urls:
        try:
            df = pd.read_csv(url)
            dfs.append(df)
        except:
            continue

    if not dfs:
        return None

    data = pd.concat(dfs, ignore_index=True)

    # Chỉ giữ lại các cột quan trọng
    # FTHG: Bàn thắng Đội nhà | FTAG: Bàn thắng Đội khách | FTR: Kết quả (H, D, A)
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    data = data[cols].dropna()

    # Chuẩn hóa định dạng ngày tháng
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    data = data.dropna(subset=['Date']).sort_values(
        'Date').reset_index(drop=True)

    return data


def get_team_form(data, team_name, current_date, n_games=5):
    """
    Tính phong độ: Hiệu số bàn thắng bại trung bình của n trận gần nhất
    trước một thời điểm cụ thể (không phân biệt sân nhà/khách).
    """
    # Lấy lịch sử thi đấu của đội
    mask = ((data['HomeTeam'] == team_name) | (
        data['AwayTeam'] == team_name)) & (data['Date'] < current_date)
    past_games = data[mask].tail(n_games)

    if len(past_games) == 0:
        return 0.0

    goal_diff = 0
    for _, row in past_games.iterrows():
        if row['HomeTeam'] == team_name:
            goal_diff += (row['FTHG'] - row['FTAG'])  # Đá sân nhà
        else:
            goal_diff += (row['FTAG'] - row['FTHG'])  # Đá sân khách

    return round(goal_diff / len(past_games), 2)


def build_training_dataset(data):
    """
    Kỹ thuật trích xuất đặc trưng (Feature Engineering) cho toàn bộ tập dữ liệu
    """
    features = []

    for i, row in data.iterrows():
        # Bỏ qua những trận đấu ở giai đoạn đầu mùa (chưa đủ 5 trận để tính phong độ)
        if i < 50:
            continue

        h_form = get_team_form(data, row['HomeTeam'], row['Date'])
        a_form = get_team_form(data, row['AwayTeam'], row['Date'])

        # Mã hóa kết quả: Home Win (2), Draw (1), Away Win (0)
        target = 2 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)

        features.append({
            'HomeForm': h_form,
            'AwayForm': a_form,
            'FormDiff': h_form - a_form,  # Chênh lệch phong độ
            'Result': target
        })

    return pd.DataFrame(features)


# --- 2. GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Premier League AI Predictor",
                   page_icon="⚽", layout="wide")

st.title("⚽ Premier League AI Predictor")
st.markdown("Hệ thống phân tích và dự đoán bóng đá giải Ngoại hạng Anh dựa trên chênh lệch **Hiệu số bàn thắng bại (Goal Difference)**.")

# Khởi tạo dữ liệu và mô hình
data = load_and_prep_data()

if data is not None:
    # Lấy danh sách các đội đang thi đấu
    teams = sorted(data['HomeTeam'].unique())

    # Giao diện chính
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Chọn trận đấu")
        home_team = st.selectbox("Đội Chủ Nhà", teams, index=teams.index(
            "Arsenal") if "Arsenal" in teams else 0)
        away_team = st.selectbox("Đội Khách", teams, index=teams.index(
            "Chelsea") if "Chelsea" in teams else 1)

        if home_team == away_team:
            st.error("Đội chủ nhà và đội khách không thể giống nhau!")

    with col2:
        st.subheader("Trạng thái Hệ thống")
        st.success(
            f"✅ Đã tải thành công cơ sở dữ liệu: **{len(data)} trận đấu**.")
        with st.spinner("Đang huấn luyện bộ não AI..."):
            # Chuẩn bị dữ liệu học
            train_df = build_training_dataset(data)

            # Huấn luyện mô hình Random Forest
            X = train_df[['HomeForm', 'AwayForm', 'FormDiff']]
            y = train_df['Result']
            model = RandomForestClassifier(
                n_estimators=200, random_state=42, max_depth=5)
            model.fit(X, y)
        st.success("✅ Mô hình đã sẵn sàng dự đoán!")

    st.divider()

    # Xử lý dự đoán
    if st.button("🚀 Phân tích & Dự đoán Tỷ lệ", type="primary", use_container_width=True):
        if home_team != away_team:
            # Lấy phong độ hiện tại nhất của 2 đội (giả định ngày hiện tại là ngày mai)
            current_date = pd.to_datetime('today')

            h_form = get_team_form(data, home_team, current_date)
            a_form = get_team_form(data, away_team, current_date)
            form_diff = h_form - a_form

            st.markdown(f"### Phân tích phong độ (5 trận gần nhất)")
            st.info(
                f"🏟️ **{home_team} (Sân nhà):** Hiệu số bàn thắng trung bình là **{h_form}** bàn/trận.")
            st.info(
                f"✈️ **{away_team} (Sân khách):** Hiệu số bàn thắng trung bình là **{a_form}** bàn/trận.")

            # Dự đoán
            input_features = pd.DataFrame([[h_form, a_form, form_diff]], columns=[
                                          'HomeForm', 'AwayForm', 'FormDiff'])
            probabilities = model.predict_proba(input_features)[0]

            # probabilities chứa mảng xác suất theo thứ tự class: [0: Away Win, 1: Draw, 2: Home Win]
            prob_away = probabilities[0] * 100
            prob_draw = probabilities[1] * 100
            prob_home = probabilities[2] * 100

            st.markdown("### Kết quả Phân tích Xác suất")

            # Hiển thị UI dạng cột
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(label=f"🟢 {home_team} Thắng",
                            value=f"{prob_home:.1f}%")
            res_col2.metric(label="⚪ Hai đội Hòa", value=f"{prob_draw:.1f}%")
            res_col3.metric(label=f"🔴 {away_team} Thắng",
                            value=f"{prob_away:.1f}%")

            # Kết luận bằng văn bản
            max_prob = max(prob_home, prob_draw, prob_away)
            if max_prob == prob_home:
                st.success(
                    f"💡 **Nhận định:** Đội chủ nhà **{home_team}** nắm giữ lợi thế lớn nhất.")
            elif max_prob == prob_away:
                st.warning(
                    f"💡 **Nhận định:** Khả năng có bất ngờ, đội khách **{away_team}** được đánh giá cao hơn.")
            else:
                st.info(
                    f"💡 **Nhận định:** Trận đấu rất cân bằng, khả năng cao sẽ kết thúc với **Tỷ số Hòa**.")
else:
    st.error("Không thể kết nối đến máy chủ dữ liệu. Kiểm tra lại kết nối mạng.")
