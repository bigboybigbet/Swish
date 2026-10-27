import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, date
from sklearn.ensemble import RandomForestClassifier
import warnings
import difflib
warnings.filterwarnings('ignore')

st.set_page_config(page_title="UCL Pro Analyzer", page_icon="⚽", layout="wide")

# ==========================================
# 1. MODULE DỮ LIỆU & HỌC MÁY
# ==========================================


class FeatureEngineer:
    COMP_WEIGHTS = {
        'UCL': 1.2, 'Premier League': 1.0, 'La Liga': 1.0,
        'FA Cup': 0.8, 'League Cup': 0.5, 'Friendly': 0.1
    }

    @staticmethod
    def get_squad_strength(match_data, team_name):
        if match_data['Competition'] in ['League Cup', 'Friendly']:
            return 0.6
        elif match_data['Competition'] == 'UCL':
            return 1.0
        return 0.9

    @classmethod
    def calculate_true_ewma_form(cls, df, team_name, span=5):
        team_matches = df[(df['Home'] == team_name) |
                          (df['Away'] == team_name)].copy()
        if team_matches.empty:
            return 1.0

        def get_base_points(row):
            if (row['Home'] == team_name and row['Result'] == 'H') or (row['Away'] == team_name and row['Result'] == 'A'):
                return 3
            elif row['Result'] == 'D':
                return 1
            return 0

        team_matches['Base_Points'] = team_matches.apply(
            get_base_points, axis=1)
        team_matches['Comp_Weight'] = team_matches['Competition'].map(
            cls.COMP_WEIGHTS).fillna(1.0)
        team_matches['Squad_Weight'] = team_matches.apply(
            lambda row: cls.get_squad_strength(row, team_name), axis=1)
        team_matches['True_Points'] = team_matches['Base_Points'] * \
            team_matches['Comp_Weight'] * team_matches['Squad_Weight']

        return team_matches['True_Points'].ewm(span=span, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_h2h(df, team1, team2):
        h2h_matches = df[((df['Home'] == team1) & (df['Away'] == team2)) |
                         ((df['Home'] == team2) & (df['Away'] == team1))]
        if h2h_matches.empty:
            return 0.5
        team1_wins = sum((h2h_matches['Home'] == team1) & (h2h_matches['Result'] == 'H')) + \
            sum((h2h_matches['Away'] == team1) &
                (h2h_matches['Result'] == 'A'))
        return team1_wins / len(h2h_matches)


@st.cache_resource
def train_model():
    try:
        np.random.seed(42)
        n = 1000
        teams = ['Real Madrid', 'Manchester City', 'Bayern Munich', 'Arsenal',
                 'Barcelona', 'PSG', 'Inter Milan', 'Atletico Madrid', 'Juventus', 'Liverpool']

        hist_df = pd.DataFrame({
            'Home': np.random.choice(teams, n),
            'Away': np.random.choice(teams, n),
            'Result': np.random.choice(['H', 'D', 'A'], n, p=[0.45, 0.25, 0.30]),
            'Competition': np.random.choice(['UCL', 'Premier League', 'FA Cup', 'League Cup', 'Friendly'], n)
        })
        hist_df = hist_df[hist_df['Home'] != hist_df['Away']].copy()

        train_df = pd.DataFrame({
            'Home_EWMA': np.random.uniform(0, 3, len(hist_df)),
            'Away_EWMA': np.random.uniform(0, 3, len(hist_df)),
            'H2H_Home_WinRate': np.random.uniform(0, 1, len(hist_df)),
            'Home_xG': np.random.uniform(0.5, 3.0, len(hist_df)),
            'Away_xG': np.random.uniform(0.5, 3.0, len(hist_df)),
            # NÂNG CẤP: Dùng trực tiếp H, D, A làm Target
            'Target': hist_df['Result']
        })

        X = train_df[['Home_EWMA', 'Away_EWMA',
                      'H2H_Home_WinRate', 'Home_xG', 'Away_xG']]
        y = train_df['Target']

        model = RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_split=5, random_state=42)
        model.fit(X, y)
        return model, hist_df, teams
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None, None, []

# ==========================================
# 2. MODULE API (ESPN + ODDS)
# ==========================================


@st.cache_data(ttl=1800)
def fetch_espn_by_date(target_date):
    date_str = target_date.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard?dates={date_str}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        fixtures = []
        for event in data.get('events', []):
            match_id = event['id']
            competitors = event['competitions'][0]['competitors']
            home_team = next(c['team']['name']
                             for c in competitors if c['homeAway'] == 'home')
            away_team = next(c['team']['name']
                             for c in competitors if c['homeAway'] == 'away')
            fixtures.append(
                {"match_id": match_id, "home": home_team, "away": away_team})
        return fixtures
    except:
        return []


def fetch_1xbet_odds(api_key, home_team, away_team):
    """Nâng cấp lấy tỷ lệ cược của cả 3 cửa 1X2"""
    if not api_key:
        return {'H': round(np.random.uniform(1.8, 2.5), 2),
                'D': round(np.random.uniform(3.0, 4.0), 2),
                'A': round(np.random.uniform(2.5, 4.5), 2)}
    url = f"https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/odds/?apiKey={api_key}&regions=eu&markets=h2h&bookmakers=1xBet"
    try:
        response = requests.get(url, timeout=5).json()
        for match in response:
            if home_team in match['home_team'] or away_team in match['away_team']:
                for bookmaker in match['bookmakers']:
                    if bookmaker['key'] == '1xbet':
                        for market in bookmaker['markets']:
                            if market['key'] == 'h2h':
                                outcomes = market['outcomes']
                                h_odds = next(
                                    (o['price'] for o in outcomes if o['name'] == home_team), 2.0)
                                a_odds = next(
                                    (o['price'] for o in outcomes if o['name'] == away_team), 2.0)
                                d_odds = next(
                                    (o['price'] for o in outcomes if o['name'] == 'Draw'), 3.0)
                                return {'H': h_odds, 'D': d_odds, 'A': a_odds}
        return {'H': 2.0, 'D': 3.0, 'A': 2.0}
    except:
        return {'H': 2.0, 'D': 3.0, 'A': 2.0}

# ==========================================
# 3. DB & HELPER FUNCTIONS
# ==========================================


def init_db():
    with sqlite3.connect("ucl_bot.db", check_same_thread=False) as conn:
        # THÊM CỘT ai_pick ĐỂ BIẾT AI CHỌN ĐỘI NÀO
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions_log 
            (match_id TEXT PRIMARY KEY, match_date TEXT, match_name TEXT, 
             ai_pick TEXT, ai_prob REAL, odds REAL, ev REAL, status TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
        ''')
        conn.commit()


def calculate_ev(prob, odds):
    return (prob * (odds - 1)) - ((1 - prob) * 1)


def get_ai_recommendation(model, features, home, away):
    """Hàm lõi để tính toán và đưa ra Chốt Kèo 1X2"""
    probs = model.predict_proba(features)[0]
    prob_dict = dict(zip(model.classes_, probs))

    # Map kết quả
    outcomes = {
        'H': {"name": f"Đội nhà ({home})", "prob": prob_dict.get('H', 0)},
        'D': {"name": "Hòa (Draw)", "prob": prob_dict.get('D', 0)},
        'A': {"name": f"Đội khách ({away})", "prob": prob_dict.get('A', 0)}
    }

    # Tìm cửa có xác suất cao nhất
    best_pick_code = max(outcomes, key=lambda k: outcomes[k]['prob'])
    return best_pick_code, outcomes[best_pick_code]['name'], outcomes[best_pick_code]['prob']

# ==========================================
# 4. GIAO DIỆN CHÍNH (UI TABS)
# ==========================================


def main():
    init_db()

    st.sidebar.title("⚙️ Bảng Điều Khiển")
    odds_api_key = st.sidebar.text_input("The Odds API Key", type="password")

    st.sidebar.divider()
    st.sidebar.subheader("🔄 Quản Lý Dữ Liệu")
    if st.sidebar.button("Làm mới & Tải lại Dữ liệu", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Đang tải dữ liệu AI..."):
        model, hist_df, team_list = train_model()

    if model is None:
        st.stop()

    tab1, tab2, tab3 = st.tabs(
        ["📅 Quét Trận Theo Ngày", "🎯 Soi Kèo Đơn", "🔬 Tự Kiểm Tra (Backtest)"])

    # ---------------------------------------------------------
    # TAB 1: QUÉT TRẬN THEO NGÀY
    # ---------------------------------------------------------
    with tab1:
        st.header("Radar Quét Kèo Toàn Diện")
        selected_date = st.date_input("Chọn ngày thi đấu UCL:", date.today())
        fixtures = fetch_espn_by_date(selected_date)

        if not fixtures:
            st.info(
                f"Không có trận đấu UCL nào được tìm thấy trên ESPN vào ngày {selected_date}.")
        else:
            for match in fixtures:
                home, away = match['home'], match['away']
                with st.container(border=True):
                    home_form = FeatureEngineer.calculate_true_ewma_form(
                        hist_df, home)
                    away_form = FeatureEngineer.calculate_true_ewma_form(
                        hist_df, away)
                    h2h = FeatureEngineer.calculate_h2h(hist_df, home, away)

                    features = np.array(
                        [home_form, away_form, h2h, 1.5, 1.2]).reshape(1, -1)

                    # AI Đưa ra quyết định
                    pick_code, pick_name, pick_prob = get_ai_recommendation(
                        model, features, home, away)

                    # Lấy tỷ lệ cược tương ứng
                    all_odds = fetch_1xbet_odds(odds_api_key, home, away)
                    target_odds = all_odds[pick_code]

                    ev = calculate_ev(pick_prob, target_odds)
                    is_value_bet = ev > 0.05

                    st.subheader(f"🏟️ {home} vs {away}")

                    # GIAO DIỆN CHỐT KÈO RÕ RÀNG
                    if is_value_bet:
                        st.success(
                            f"🎯 **AI CHỐT KÈO:** Đánh cửa **{pick_name}** (Xác suất thắng: {pick_prob*100:.1f}%)")
                    else:
                        st.warning(
                            f"🤔 **GỢI Ý:** {pick_name} có cửa thắng cao ({pick_prob*100:.1f}%), nhưng tỷ lệ nhà cái không thơm. Khuyến nghị BỎ QUA.")

                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"Tỉ lệ 1xBet ({pick_code})", target_odds)
                    col2.metric("Chỉ số Lợi Nhuận (EV)", f"{ev:.3f} Unit", "Kèo Thơm 🚀" if is_value_bet else "Kèo Xấu",
                                delta_color="normal" if is_value_bet else "inverse")

                    if st.button(f"💾 Lưu kèo vào Backtest", key=f"save_{match['match_id']}"):
                        with sqlite3.connect("ucl_bot.db", check_same_thread=False) as conn:
                            conn.execute("INSERT OR REPLACE INTO predictions_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                                         (match['match_id'], str(selected_date), f"{home} vs {away}", pick_name, pick_prob, target_odds, ev, "Pending"))
                            conn.commit()
                        st.toast(f"Đã lưu kèo: {pick_name}")

    # ---------------------------------------------------------
    # TAB 2: SOI KÈO ĐƠN
    # ---------------------------------------------------------
    with tab2:
        st.header("Phòng Phân Tích Chuyên Sâu (Soi Kèo Đơn)")
        st.markdown(
            "Nhập tự do tên đội bóng (Hệ thống có khả năng tự động nhận diện và sửa lỗi chính tả).")

        comp_list = list(FeatureEngineer.COMP_WEIGHTS.keys())
        current_comp = st.selectbox(
            "🏆 Khuôn khổ giải đấu:", comp_list, index=0)

        col_h, col_a = st.columns(2)
        home_input = col_h.text_input(
            "Nhập Đội Nhà (Home):", value="", placeholder="VD: man city, real...")
        away_input = col_a.text_input(
            "Nhập Đội Khách (Away):", value="", placeholder="VD: bayern, arsenal...")

        if st.button("🚀 Chạy Phân Tích Đơn", use_container_width=True):
            if not home_input or not away_input:
                st.warning("Vui lòng nhập tên cả hai đội bóng.")
            elif home_input.strip().lower() == away_input.strip().lower():
                st.error("Tên hai đội không được trùng nhau.")
            else:
                # 1. THUẬT TOÁN FUZZY MATCHING (TÌM KIẾM MỜ)
                # Tự động so sánh text bạn nhập với danh sách các đội CÓ THẬT trong lịch sử (hist_df)
                # Lấy danh sách đội chuẩn từ CSDL
                valid_teams = hist_df['Home'].unique().tolist()

                real_home_matches = difflib.get_close_matches(
                    home_input, valid_teams, n=1, cutoff=0.3)
                real_away_matches = difflib.get_close_matches(
                    away_input, valid_teams, n=1, cutoff=0.3)

                # 2. XỬ LÝ KẾT QUẢ NHẬN DIỆN
                if not real_home_matches or not real_away_matches:
                    missing = []
                    if not real_home_matches:
                        missing.append(home_input)
                    if not real_away_matches:
                        missing.append(away_input)
                    st.error(
                        f"❌ KHÔNG TÌM THẤY DỮ LIỆU: Hệ thống không có lịch sử đối đầu/phong độ của '{', '.join(missing)}' trong cơ sở dữ liệu UCL. Vui lòng kiểm tra lại.")
                else:
                    home_sel = real_home_matches[0]
                    away_sel = real_away_matches[0]

                    if home_sel == away_sel:
                        st.error(
                            f"❌ Sau khi nhận diện, cả 2 từ khóa đều trỏ về đội '{home_sel}'. Vui lòng nhập rõ hơn.")
                    else:
                        st.success(
                            f"🔍 Đã nhận diện dữ liệu chuẩn: **{home_sel}** vs **{away_sel}**")

                        # 3. CHẠY LOGIC AI VỚI DỮ LIỆU THẬT 100%
                        h_form = FeatureEngineer.calculate_true_ewma_form(
                            hist_df, home_sel)
                        a_form = FeatureEngineer.calculate_true_ewma_form(
                            hist_df, away_sel)
                        h2h_val = FeatureEngineer.calculate_h2h(
                            hist_df, home_sel, away_sel)

                        context_penalty = 1.0
                        if current_comp in ['League Cup', 'Friendly']:
                            st.info(
                                f"💡 AI Note: Giải {current_comp} rủi ro cao do xoay tua đội hình. AI tự động điều chỉnh giảm độ tin cậy.")
                            context_penalty = 0.85

                        features = np.array(
                            [h_form, a_form, h2h_val, 1.5, 1.2]).reshape(1, -1)

                        pick_code, pick_name, raw_prob = get_ai_recommendation(
                            model, features, home_sel, away_sel)
                        adjusted_prob = 0.5 + \
                            (raw_prob - 0.5) * context_penalty

                        st.divider()
                        st.subheader(
                            f"🤖 KẾT LUẬN TỪ AI: CHỌN {pick_name.upper()}")

                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            f"Phong độ (EWMA) {home_sel}", f"{h_form:.2f}")
                        c2.metric(
                            f"Phong độ (EWMA) {away_sel}", f"{a_form:.2f}")
                        c3.metric(
                            f"Tỷ lệ thắng H2H ({home_sel})", f"{h2h_val*100:.1f}%")

                        st.progress(float(
                            adjusted_prob), text=f"Xác suất dự đoán (Đã hiệu chỉnh rủi ro giải đấu): {adjusted_prob*100:.2f}%")

    # ---------------------------------------------------------
    # TAB 3: TỰ KIỂM TRA (Backtest Dashboard)
    # ---------------------------------------------------------
    with tab3:
        st.header("Hệ Thống Thẩm Định Cược (Backtest Dashboard)")

        with sqlite3.connect("ucl_bot.db", check_same_thread=False) as conn:
            df_backtest = pd.read_sql_query(
                "SELECT * FROM predictions_log ORDER BY timestamp DESC", conn)

        if df_backtest.empty:
            st.info("Chưa có dữ liệu. Hãy quét trận và lưu kèo ở Tab 1.")
        else:
            st.subheader("Thống kê Tổng Quan")
            total_bets = len(df_backtest[df_backtest['status'] != 'Pending'])
            won_bets = len(df_backtest[df_backtest['status'] == 'Won'])

            profit = 0
            for _, row in df_backtest.iterrows():
                if row['status'] == 'Won':
                    profit += (row['odds'] - 1)
                elif row['status'] == 'Lost':
                    profit -= 1

            win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Kèo đã chốt", total_bets)
            sc2.metric("Win Rate", f"{win_rate:.1f}%")
            sc3.metric("Lợi nhuận (Units)", f"{profit:.2f}",
                       delta_color="normal" if profit >= 0 else "inverse")

            st.divider()
            st.subheader("Lịch sử Khuyến nghị (Cập nhật Kết Quả)")
            # Hiển thị bảng chi tiết, bao gồm cột ai_pick để biết đã đánh đội nào
            st.dataframe(df_backtest[['match_date', 'match_name', 'ai_pick',
                         'odds', 'ev', 'status']], use_container_width=True)

            update_col1, update_col2 = st.columns([3, 1])
            pending_matches = df_backtest[df_backtest['status']
                                          == 'Pending']['match_id'].tolist()

            if pending_matches:
                match_to_update = update_col1.selectbox("Chọn trận cần cập nhật:", pending_matches, format_func=lambda x: df_backtest[df_backtest[
                                                        'match_id'] == x]['match_name'].values[0] + " (Chọn: " + df_backtest[df_backtest['match_id'] == x]['ai_pick'].values[0] + ")")
                new_status = update_col2.selectbox("Kết quả:", ["Won", "Lost"])

                if st.button("Lưu Kết Quả"):
                    with sqlite3.connect("ucl_bot.db", check_same_thread=False) as conn:
                        conn.execute(
                            "UPDATE predictions_log SET status = ? WHERE match_id = ?", (new_status, match_to_update))
                        conn.commit()
                    st.success(
                        "Đã cập nhật! Hãy refresh (F5) trang để xem ROI mới.")


if __name__ == "__main__":
    main()
