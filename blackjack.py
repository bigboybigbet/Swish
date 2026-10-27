import streamlit as st
import random
import time

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Pro Blackjack",
                   page_icon="🃏", layout="centered")

# --- CSS TÙY CHỈNH CHO THẺ BÀI ---
st.markdown("""
<style>
    .card {
        display: inline-block;
        width: 80px; height: 120px;
        border-radius: 8px;
        border: 2px solid #ddd;
        background-color: white;
        text-align: center;
        line-height: 110px;
        font-size: 24px; font-weight: bold;
        margin: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .red-card { color: #d9534f; }
    .black-card { color: #292b2c; }
    .hidden-card {
        background-color: #0275d8;
        background-image: repeating-linear-gradient(45deg, transparent, transparent 10px, #025aa5 10px, #025aa5 20px);
        color: transparent; border: 2px solid white;
    }
</style>
""", unsafe_allow_html=True)

# --- DỮ LIỆU CƠ BẢN ---
SUIT_SYMBOLS = {'Hearts': '♥', 'Diamonds': '♦', 'Spades': '♠', 'Clubs': '♣'}
RANKS = {'Two': '2', 'Three': '3', 'Four': '4', 'Five': '5', 'Six': '6', 'Seven': '7',
         'Eight': '8', 'Nine': '9', 'Ten': '10', 'Jack': 'J', 'Queen': 'Q', 'King': 'K', 'Ace': 'A'}
VALUES = {'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6, 'Seven': 7,
          'Eight': 8, 'Nine': 9, 'Ten': 10, 'Jack': 10, 'Queen': 10, 'King': 10, 'Ace': 11}

# --- KHỞI TẠO STATE (LƯU TRỮ DỮ LIỆU) ---
if 'balance' not in st.session_state:
    st.session_state.balance = 1000
if 'deck' not in st.session_state or len(st.session_state.deck) < 20:
    st.session_state.deck = [f"{r} of {s}" for _ in range(
        6) for s in SUIT_SYMBOLS.keys() for r in VALUES.keys()]
    random.shuffle(st.session_state.deck)
if 'player_hand' not in st.session_state:
    st.session_state.player_hand = []
if 'dealer_hand' not in st.session_state:
    st.session_state.dealer_hand = []
if 'game_active' not in st.session_state:
    st.session_state.game_active = False
if 'message' not in st.session_state:
    st.session_state.message = ""

# --- HÀM LOGIC ---


def deal_card():
    return st.session_state.deck.pop()


def calc_score(hand):
    score = sum(VALUES[card.split()[0]] for card in hand)
    aces = sum(1 for card in hand if "Ace" in card)
    while score > 21 and aces:
        score -= 10
        aces -= 1
    return score


def render_html_hand(hand, hide_first=False):
    """Tạo mã HTML để vẽ các lá bài có màu sắc"""
    html = ""
    for i, card in enumerate(hand):
        if hide_first and i == 0:
            html += '<div class="card hidden-card">?</div>'
        else:
            rank = RANKS[card.split()[0]]
            suit = card.split()[2]
            symbol = SUIT_SYMBOLS[suit]
            color_class = "red-card" if suit in ['Hearts',
                                                 'Diamonds'] else "black-card"
            html += f'<div class="card {color_class}">{rank}{symbol}</div>'
    return html


# --- GIAO DIỆN CHÍNH ---
st.title("🎰 Blackjack Pro - Web Edition")

# Dashboard thông tin
col1, col2, col3 = st.columns(3)
col1.metric("💰 Bankroll", f"${st.session_state.balance}")
col2.metric("🎴 Cards in Shoe", len(st.session_state.deck))

# Khung hiển thị thông báo kết quả
if st.session_state.message:
    st.info(st.session_state.message)

st.divider()

# --- PHẦN 1: ĐẶT CƯỢC & BẮT ĐẦU ---
if not st.session_state.game_active:
    st.subheader("Mời đặt cược!")
    bet_amount = st.number_input(
        "Số tiền cược:", min_value=10, max_value=st.session_state.balance, value=100, step=10)

    if st.button("🚀 Deal (Chia bài)", use_container_width=True):
        st.session_state.bet = bet_amount
        st.session_state.player_hand = [deal_card(), deal_card()]
        st.session_state.dealer_hand = [deal_card(), deal_card()]
        st.session_state.game_active = True
        st.session_state.message = ""
        st.rerun()

# --- PHẦN 2: BÀN CHƠI (Khi game đang chạy) ---
else:
    # 1. Khu vực Nhà cái (Dealer)
    st.subheader("Nhà Cái (Dealer)")
    dealer_placeholder = st.empty()  # Khung trống để làm animation
    dealer_placeholder.markdown(render_html_hand(
        st.session_state.dealer_hand, hide_first=True), unsafe_allow_html=True)

    st.write("")  # Spacer

    # 2. Khu vực Người chơi (Player)
    st.subheader(
        f"Người Chơi (Điểm: {calc_score(st.session_state.player_hand)})")
    st.markdown(render_html_hand(st.session_state.player_hand),
                unsafe_allow_html=True)

    st.write("")

    # Kểm tra nếu người chơi Quắc (Bust)
    if calc_score(st.session_state.player_hand) > 21:
        st.error("💥 Bạn đã BUST (Quá 21 điểm)! Nhà cái thắng.")
        st.session_state.balance -= st.session_state.bet
        st.session_state.game_active = False
        if st.button("Chơi ván mới"):
            st.rerun()

    else:
        # Nút Hành động
        col_h, col_s, col_d = st.columns(3)

        if col_h.button("👉 Rút bài (Hit)", use_container_width=True):
            st.session_state.player_hand.append(deal_card())
            st.rerun()

        if col_d.button("💰 Gấp đôi (Double)", disabled=(st.session_state.balance < st.session_state.bet * 2), use_container_width=True):
            st.session_state.bet *= 2
            st.session_state.player_hand.append(deal_card())
            # Tự động Stand sau khi Double
            st.session_state.action_stand = True
            st.rerun()

        if col_s.button("🛑 Dừng (Stand)", use_container_width=True) or st.session_state.get('action_stand', False):
            st.session_state.action_stand = False  # Reset flag

            # ANIMATION: Lật bài của nhà cái
            st.session_state.message = "Dealer đang mở bài..."
            dealer_placeholder.markdown(render_html_hand(
                st.session_state.dealer_hand, hide_first=False), unsafe_allow_html=True)
            time.sleep(1)  # Delay tạo kịch tính

            # Lượt nhà cái rút bài (kèm animation)
            while calc_score(st.session_state.dealer_hand) < 17:
                st.session_state.dealer_hand.append(deal_card())
                dealer_placeholder.markdown(render_html_hand(
                    st.session_state.dealer_hand, hide_first=False), unsafe_allow_html=True)
                time.sleep(0.8)  # Delay từng lá bài được rút ra

            # Tính toán kết quả
            p_score = calc_score(st.session_state.player_hand)
            d_score = calc_score(st.session_state.dealer_hand)

            if d_score > 21:
                st.session_state.message = f"🎉 Nhà cái BUST ({d_score}). BẠN THẮNG ${st.session_state.bet}!"
                st.session_state.balance += st.session_state.bet
            elif p_score > d_score:
                st.session_state.message = f"🎉 BẠN THẮNG! ({p_score} vs {d_score})"
                st.session_state.balance += st.session_state.bet
            elif p_score < d_score:
                st.session_state.message = f"💸 BẠN THUA! ({p_score} vs {d_score})"
                st.session_state.balance -= st.session_state.bet
            else:
                st.session_state.message = f"🤝 HÒA (PUSH)! ({p_score} vs {d_score})"

            st.session_state.game_active = False
            st.rerun()

# Nút Deposit nếu hết tiền
if st.session_state.balance <= 0 and not st.session_state.game_active:
    st.error("Bạn đã hết tiền!")
    if st.button("💳 Nạp thêm $1000"):
        st.session_state.balance += 1000
        st.rerun()
