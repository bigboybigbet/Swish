import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import os
from datetime import datetime, timedelta
import time
from nba_api.stats.static import teams
import requests

# ==========================================
# CONFIGURATION & SESSION STATE
# ==========================================
MODEL_FILE = "nba_brain.joblib"
LOG_FILE = "prediction_history.csv"

st.set_page_config(page_title="NBA AI Master", page_icon="🏀", layout="wide")

if 'schedule_data' not in st.session_state:
    st.session_state.schedule_data = None

# ==========================================
# 1. DATA ACQUISITION & TRAINING SYSTEM
# ==========================================


def get_real_time_form(team_name, target_date=None):
    """
    Retrieves Plus/Minus and average points for the last 5 games.
    Includes 'target_date' logic to prevent data leakage during backtesting.
    """
    try:
        nba_teams = teams.get_teams()
        team_info = [t for t in nba_teams if team_name.lower(
        ) in t['full_name'].lower() or t['full_name'].lower() in team_name.lower()]

        if not team_info:
            return None

        team_id = team_info[0]['id']
        full_team_name = team_info[0]['full_name']

        game_finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id)
        games = game_finder.get_data_frames()[0]

        # ANTI-DATA LEAKAGE: Only take games that occurred BEFORE the target_date
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        if target_date:
            games = games[games['GAME_DATE'] < pd.to_datetime(target_date)]

        games = games.dropna(subset=['PLUS_MINUS', 'PTS'])
        if games.empty:
            return None

        recent_5_games = games.head(5)

        avg_form = recent_5_games['PLUS_MINUS'].mean()
        avg_pts = recent_5_games['PTS'].mean()
        avg_opponent_pts = avg_pts - avg_form

        return {
            "name": full_team_name,
            "form": avg_form,
            "pts_scored": avg_pts,
            "pts_allowed": avg_opponent_pts
        }
    except:
        return None


def get_h2h_winrate(home_team, away_team, target_date=None):
    """Calculates win rate of Home Team vs Away Team in the last 5 matchups."""
    try:
        nba_teams = teams.get_teams()
        h_info = [t for t in nba_teams if home_team.lower()
                  in t['full_name'].lower()]
        a_info = [t for t in nba_teams if away_team.lower()
                  in t['full_name'].lower()]

        if not h_info or not a_info:
            return 0.5

        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=h_info[0]['id'],
            vs_team_id_nullable=a_info[0]['id']
        )
        games = finder.get_data_frames()[0]

        # ANTI-DATA LEAKAGE
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        if target_date:
            games = games[games['GAME_DATE'] < pd.to_datetime(target_date)]

        if games.empty:
            return 0.5

        recent_matchups = games.head(5)
        home_wins = len(recent_matchups[recent_matchups['WL'] == 'W'])
        return home_wins / len(recent_matchups)
    except:
        return 0.5


def get_multi_season_data():
    """Fetches historical data for multiple seasons to train the model."""
    seasons = ['2024-25', '2025-26']
    all_games = []
    for season in seasons:
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, league_id_nullable='00')
            all_games.append(finder.get_data_frames()[0])
            time.sleep(1)  # Rate limit protection
        except:
            pass

    if all_games:
        df = pd.concat(all_games, ignore_index=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        df['RESULT'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
        df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

        # Calculate form (average plus/minus of previous 5 games)
        df['FORM'] = df.groupby('TEAM_ID')['PLUS_MINUS'].transform(
            lambda x: x.shift(1).rolling(5).mean())
        return df.dropna(subset=['FORM', 'WL'])
    return None


def train_model(df):
    """Trains the Random Forest Classifier."""
    X = df[['IS_HOME', 'FORM']]
    y = df['RESULT']
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    dump(model, MODEL_FILE)
    return model

# ==========================================
# 2. ODDS & SCHEDULE SYSTEM
# ==========================================


def get_schedule_by_date(selected_date):
    date_str = selected_date.strftime('%Y%m%d')
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        schedule = []
        for event in data.get('events', []):
            try:
                competitors = event['competitions'][0]['competitors']
                home_team = next(c['team']['displayName']
                                 for c in competitors if c['homeAway'] == 'home')
                away_team = next(c['team']['displayName']
                                 for c in competitors if c['homeAway'] == 'away')
                schedule.append({
                    "Date": selected_date.strftime('%Y-%m-%d'),
                    "Home Team": home_team, "Away Team": away_team
                })
            except:
                continue
        return pd.DataFrame(schedule)
    except:
        return None


def get_past_results_espn(selected_date):
    """Retrieves finalized game results for backtesting."""
    date_str = selected_date.strftime('%Y%m%d')
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    try:
        data = requests.get(url, timeout=10).json()
        results = []
        for event in data.get('events', []):
            if event['status']['type']['name'] == 'STATUS_FINAL':
                comps = event['competitions'][0]['competitors']
                home_c = next(c for c in comps if c['homeAway'] == 'home')
                away_c = next(c for c in comps if c['homeAway'] == 'away')

                h_score = int(home_c['score'])
                a_score = int(away_c['score'])

                results.append({
                    "Home Team": home_c['team']['displayName'],
                    "Away Team": away_c['team']['displayName'],
                    "H_Score": h_score,
                    "A_Score": a_score,
                    "Actual_Winner": 1 if h_score > a_score else 0
                })
        return results
    except:
        return []


def get_live_odds(api_key):
    """Fetches real-time Moneyline odds."""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={api_key}&regions=eu,us&markets=h2h"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        odds_dict = {}
        for match in data:
            home = match['home_team']
            target_bookie = next(
                (b for b in match['bookmakers'] if b['key'].lower() == 'onexbet'), None)
            if not target_bookie and len(match['bookmakers']) > 0:
                target_bookie = match['bookmakers'][0]
            if target_bookie:
                h_odd, a_odd = 0.0, 0.0
                for outcome in target_bookie['markets'][0]['outcomes']:
                    if outcome['name'] == home:
                        h_odd = outcome['price']
                    else:
                        a_odd = outcome['price']
                odds_dict[home] = {
                    "bookie": target_bookie['title'], "home_odd": h_odd, "away_odd": a_odd}
        return odds_dict
    except:
        return None


def generate_ai_explanation(pred_res, act_res, h_score, a_score):
    """Post-game analysis logic for backtesting."""
    diff = abs(h_score - a_score)
    if pred_res == act_res:
        return "✅ AI predicted correctly. Team form and H2H metrics were accurate indicators."
    if diff > 15:
        return f"❌ Blowout ({diff} pts). Unexpected performance surge or drop-off outside historical norms."
    if diff <= 5:
        return f"❌ Clutch Loss ({diff} pts). Game was decided in the final minutes; statistical noise."
    return "❌ Out of rhythm. Favored team underperformed. Model needs recent form update."


# ==========================================
# MAIN INTERFACE (UI)
# ==========================================
try:
    st.title("🏀 NBA AI INVESTMENT SYSTEM")

    with st.sidebar:
        st.header("⚙️ AI Engine")
        if st.button("🔄 Update Data & Train Model", use_container_width=True):
            with st.spinner("Fetching data from NBA stats..."):
                data = get_multi_season_data()
                if data is not None and not data.empty:
                    train_model(data)
                    st.success("✅ Model Trained successfully!")
                else:
                    st.error("❌ Connection failed.")

    tab1, tab2, tab3 = st.tabs(
        ["🎯 Single Prediction", "💰 Value Bet Radar", "🔬 Automated Backtest"])

    # --- TAB 1: SINGLE MATCH ANALYSIS ---
    with tab1:
        st.write("---")
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Match Analysis")
            ht = st.text_input("Home Team", "Lakers")
            at = st.text_input("Away Team", "Warriors")
            if st.button("🚀 RUN AI PREDICTION", type="primary"):
                if os.path.exists(MODEL_FILE):
                    model = load(MODEL_FILE)
                    hd, ad = get_real_time_form(ht), get_real_time_form(at)
                    if hd and ad:
                        base_prob = model.predict_proba(pd.DataFrame(
                            [[1, hd['form'] - ad['form']]], columns=['IS_HOME', 'FORM']))[0][1]
                        final_prob = (base_prob * 0.8) + \
                            (get_h2h_winrate(hd['name'], ad['name']) * 0.2)
                        st.metric(label=f"{ht} Win Probability",
                                  value=f"{final_prob*100:.1f}%")
                        if final_prob > 0.5:
                            st.success(f"🏆 AI Selection: **{ht}**")
                        else:
                            st.warning(f"⚠️ AI Selection: **{at}**")
                    else:
                        st.error("Invalid team name or no data found.")
                else:
                    st.error("AI Model not trained yet!")

    # --- TAB 2: EV RADAR ---
    with tab2:
        st.write("---")
        st.subheader("Expected Value (EV) Radar")
        api_key_input = st.text_input(
            "🔑 Enter The-Odds-API Key:", type="password")
        target_date = st.date_input("🗓️ Select Date", datetime.today())

        if st.button("🔍 Scan Schedule & Odds", use_container_width=True):
            with st.spinner("Connecting to ESPN..."):
                df_res = get_schedule_by_date(target_date)
                if df_res is not None and not df_res.empty:
                    if not os.path.exists(MODEL_FILE):
                        st.error("⚠️ AI Model not trained.")
                    else:
                        model = load(MODEL_FILE)
                        live_odds = get_live_odds(
                            api_key_input) if api_key_input else None
                        recs, progress_bar = [], st.progress(0)

                        for i, row in df_res.iterrows():
                            hn, an = row['Home Team'], row['Away Team']
                            hd, ad = get_real_time_form(
                                hn), get_real_time_form(an)
                            if hd and ad:
                                base_prob = model.predict_proba(pd.DataFrame(
                                    [[1, hd['form'] - ad['form']]], columns=['IS_HOME', 'FORM']))[0][1]
                                prob_h = (base_prob * 0.8) + \
                                    (get_h2h_winrate(hn, an) * 0.2)
                                prob_a = 1 - prob_h

                                if live_odds:
                                    match_odds = next(
                                        (v for k, v in live_odds.items() if k in hn or hn in k), None)
                                    if match_odds:
                                        ev_h = (
                                            prob_h * match_odds['home_odd']) - 1
                                        ev_a = (
                                            prob_a * match_odds['away_odd']) - 1
                                        if ev_h > 0.03:
                                            recs.append(
                                                {"Match": f"{hn} vs {an}", "Pick": f"🏠 {hn}", "Odds": match_odds['home_odd'], "EV": f"+{ev_h*100:.1f}%"})
                                        elif ev_a > 0.03:
                                            recs.append({"Match": f"{hn} vs {an}", "Pick": f"✈️ {an}",
                                                        "Odds": match_odds['away_odd'], "EV": f"+{ev_a*100:.1f}%"})
                                else:
                                    recs.append(
                                        {"Match": f"{hn} vs {an}", "AI Prediction": f"🏠 {hn} ({prob_h*100:.1f}%)" if prob_h > 0.5 else f"✈️ {an} ({prob_a*100:.1f}%)"})
                            progress_bar.progress((i + 1) / len(df_res))

                        if recs:
                            st.dataframe(pd.DataFrame(
                                recs), hide_index=True, use_container_width=True)
                        else:
                            st.warning("No value bets found.")
                else:
                    st.warning("No games found for this date.")

    # --- TAB 3: BACKTESTING ---
    with tab3:
        st.write("---")
        st.subheader("🔬 Performance Audit & Validation")
        st.info("This module simulates AI predictions for past dates and compares them with actual results to audit accuracy.")

        backtest_date = st.date_input(
            "🗓️ Select Past Date for Audit", datetime.today() - timedelta(days=1))

        if st.button("🚀 START BACKTEST SIMULATION", type="primary", use_container_width=True):
            if not os.path.exists(MODEL_FILE):
                st.error("⚠️ Model not trained.")
            else:
                with st.spinner("Processing historical data and AI simulation..."):
                    model = load(MODEL_FILE)
                    past_games = get_past_results_espn(backtest_date)

                    if not past_games:
                        st.warning("No finalized games found for this date.")
                    else:
                        eval_data = []
                        correct_count = 0
                        progress_bar = st.progress(0)

                        for i, game in enumerate(past_games):
                            hn, an = game['Home Team'], game['Away Team']
                            act_winner = game['Actual_Winner']

                            # Use target_date to get form AS IT WAS on that day
                            hd = get_real_time_form(
                                hn, target_date=backtest_date)
                            ad = get_real_time_form(
                                an, target_date=backtest_date)

                            if hd and ad:
                                base_prob = model.predict_proba(pd.DataFrame(
                                    [[1, hd['form'] - ad['form']]], columns=['IS_HOME', 'FORM']))[0][1]
                                h2h = get_h2h_winrate(
                                    hn, an, target_date=backtest_date)
                                final_prob_h = (base_prob * 0.8) + (h2h * 0.2)

                                pred_winner = 1 if final_prob_h > 0.5 else 0
                                is_correct = (pred_winner == act_winner)
                                if is_correct:
                                    correct_count += 1

                                explanation = generate_ai_explanation(
                                    pred_winner, act_winner, game['H_Score'], game['A_Score'])

                                eval_data.append({
                                    "Game": f"{hn} vs {an}",
                                    "Actual Score": f"{game['H_Score']} - {game['A_Score']}",
                                    "AI Prediction": f"🏠 {hn}" if pred_winner == 1 else f"✈️ {an}",
                                    "Status": "✅ WIN" if is_correct else "❌ LOSS",
                                    "Bot Analysis": explanation
                                })
                            progress_bar.progress((i + 1) / len(past_games))

                        accuracy = (correct_count / len(past_games)) * 100
                        st.markdown(
                            f"### 🎯 Simulation Accuracy: **{accuracy:.1f}%** ({correct_count}/{len(past_games)} games)")
                        st.dataframe(pd.DataFrame(eval_data),
                                     hide_index=True, use_container_width=True)

except Exception as e:
    st.error(f"🔥 SYSTEM ERROR: {e}")
