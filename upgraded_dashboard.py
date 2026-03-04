import streamlit as st
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

st.set_page_config(page_title="AI Forex Intelligence Terminal", layout="wide", page_icon="📈")

# ========================= SECRETS =========================
TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# ========================= 32 PAIRS (ALL MAJOR PLAYERS) =========================
ALL_PAIRS = {
    "EURUSD": ["EUR/USD", ["EUR", "USD", "ECB", "Fed", "Euro", "Dollar"]],
    "GBPUSD": ["GBP/USD", ["GBP", "USD", "BOE", "Fed", "Pound", "Dollar"]],
    "USDJPY": ["USD/JPY", ["USD", "JPY", "BOJ", "Fed", "Dollar", "Yen"]],
    "USDCAD": ["USD/CAD", ["USD", "CAD", "BOC", "Fed", "Loonie"]],
    "AUDUSD": ["AUD/USD", ["AUD", "USD", "RBA", "Fed", "Aussie"]],
    "NZDUSD": ["NZD/USD", ["NZD", "USD", "RBNZ", "Fed", "Kiwi"]],
    "USDCHF": ["USD/CHF", ["USD", "CHF", "SNB", "Fed", "Franc"]],
    "EURGBP": ["EUR/GBP", ["EUR", "GBP", "ECB", "BOE"]],
    "EURJPY": ["EUR/JPY", ["EUR", "JPY", "ECB", "BOJ"]],
    "GBPJPY": ["GBP/JPY", ["GBP", "JPY", "BOE", "BOJ"]],
    "EURCHF": ["EUR/CHF", ["EUR", "CHF", "ECB", "SNB"]],
    "EURAUD": ["EUR/AUD", ["EUR", "AUD", "ECB", "RBA"]],
    "GBPAUD": ["GBP/AUD", ["GBP", "AUD", "BOE", "RBA"]],
    "AUDJPY": ["AUD/JPY", ["AUD", "JPY", "RBA", "BOJ"]],
    "USDCNH": ["USD/CNH", ["USD", "CNH", "PBOC", "Fed"]],
    "XAUUSD": ["XAU/USD", ["Gold", "USD", "Fed", "Treasury"]],
    "XAGUSD": ["XAG/USD", ["Silver", "USD", "Fed"]],
    "EURCAD": ["EUR/CAD", ["EUR", "CAD", "ECB", "BOC"]],
    "GBPCAD": ["GBP/CAD", ["GBP", "CAD", "BOE", "BOC"]],
    "AUDCAD": ["AUD/CAD", ["AUD", "CAD", "RBA", "BOC"]],
    "USDTRY": ["USD/TRY", ["USD", "TRY", "Turkey", "Lira"]],
    "USDMXN": ["USD/MXN", ["USD", "MXN", "Mexico", "Peso"]],
    "USDZAR": ["USD/ZAR", ["USD", "ZAR", "South Africa", "Rand"]],
    "USDBRL": ["USD/BRL", ["USD", "BRL", "Brazil", "Real"]],
    "EURTRY": ["EUR/TRY", ["EUR", "TRY", "ECB", "Turkey"]],
    "GBPTRY": ["GBP/TRY", ["GBP", "TRY", "BOE", "Turkey"]],
    "NZDJPY": ["NZD/JPY", ["NZD", "JPY", "RBNZ", "BOJ"]],
    "CADJPY": ["CAD/JPY", ["CAD", "JPY", "BOC", "BOJ"]],
    "AUDNZD": ["AUD/NZD", ["AUD", "NZD", "RBA", "RBNZ"]],
    "CHFJPY": ["CHF/JPY", ["CHF", "JPY", "SNB", "BOJ"]],
    "GBPNZD": ["GBP/NZD", ["GBP", "NZD", "BOE", "RBNZ"]],
    "EURNZD": ["EUR/NZD", ["EUR", "NZD", "ECB", "RBNZ"]],
}

# ========================= CACHED HELPERS (same as before) =========================
@st.cache_data(ttl=90)
def get_price(symbol: str): 
    try:
        r = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}", timeout=10).json()
        return float(r.get("price", 0)) if "price" in r else None
    except: return None

@st.cache_data(ttl=180)
def get_time_series_tf(symbol: str, interval: str, size: int = 60):
    try:
        r = requests.get(f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={size}&apikey={TWELVE_KEY}", timeout=10).json()
        if "values" not in r: return None
        df = pd.DataFrame(r["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        return df
    except: return None

def calculate_technical_score(df):
    if df is None or len(df) < 15: return 50, ["⚠️ Limited data"]
    closes = df["close"]
    ema12 = closes.ewm(span=12).mean().iloc[-1]
    ema26 = closes.ewm(span=26).mean().iloc[-1]
    rsi = calculate_rsi(closes)
    score = 50
    reasons = []
    if ema12 > ema26: score += 25; reasons.append("EMA12 > EMA26 → Bullish")
    else: score -= 25; reasons.append("EMA12 < EMA26 → Bearish")
    if rsi > 70: score -= 12; reasons.append(f"RSI {rsi:.1f} Overbought")
    elif rsi < 30: score += 12; reasons.append(f"RSI {rsi:.1f} Oversold")
    return max(15, min(85, int(score))), reasons

def calculate_rsi(prices, period=14):
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return float(100 - (100 / (1 + rs)).iloc[-1])
    except: return 50.0

# (RSS, FinBERT, news_analysis functions stay exactly the same as last version — I kept them short for space)

@st.cache_data(ttl=300)
def get_rss_headlines():
    feeds = ["https://feeds.reuters.com/reuters/businessNews", "https://www.fxstreet.com/rss/news", "https://www.marketwatch.com/rss/topstories"]
    headlines = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:15]:
                if title := item.find("title").text:
                    headlines.append(title.strip())
        except: continue
    return headlines

def finbert(text):
    try:
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        result = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=15).json()
        if isinstance(result, list) and result:
            scores = result[0]
            pos = next((s["score"] for s in scores if s["label"] == "positive"), 0)
            neg = next((s["score"] for s in scores if s["label"] == "negative"), 0)
            return int(50 + (pos - neg) * 50)
        return 50
    except: return 50

@st.cache_data(ttl=300)
def news_analysis(keywords):
    all_headlines = get_rss_headlines()
    filtered = [h for h in all_headlines if any(k.lower() in h.lower() for k in keywords)]
    if not filtered: return 50, ["No specific news"]
    score = finbert(" | ".join(filtered[:6]))
    return score, filtered[:6]

@st.cache_data(ttl=600)
def get_tf_trend(symbol: str, interval: str) -> str:
    df = get_time_series_tf(symbol, interval, 40)
    if df is None or len(df) < 20: return "Neutral"
    ema50 = df["close"].ewm(span=50).mean().iloc[-1]
    return "Bullish" if df["close"].iloc[-1] > ema50 else "Bearish"

# ========================= CORE ANALYSIS WITH NEUTRAL FIX =========================
def analyze_pair(pair_name: str, symbol: str, keywords: list):
    price = get_price(symbol)
    df_1h = get_time_series_tf(symbol, "1h", 60)
    
    tech_score, tech_reasons = calculate_technical_score(df_1h)
    news_score, headlines = news_analysis(keywords)
    
    final_score = int(0.65 * tech_score + 0.35 * news_score)
    final_score = max(15, min(85, final_score))
    
    direction = "Bullish" if final_score >= 68 else "Bearish" if final_score <= 38 else "Neutral"
    
    # === NEW NEUTRAL RISK RULE ===
    if direction == "Neutral":
        risk_percent = 0.5
        risk_color = "⚪"
        aligned_count = 0
        tf_status = "Neutral market — conservative"
        weekly_trend = daily_trend = h4_trend = "—"
    else:
        weekly_trend = get_tf_trend(symbol, "1week")
        daily_trend = get_tf_trend(symbol, "1day")
        h4_trend = get_tf_trend(symbol, "4h")
        tf_trends = [weekly_trend, daily_trend, h4_trend]
        aligned_count = sum(1 for t in tf_trends if t == direction)
        
        if aligned_count == 3:
            risk_percent = 3.0
            risk_color = "🟢"
            tf_status = "All 3 TFs aligned"
        elif aligned_count == 2:
            risk_percent = 1.5
            risk_color = "🟡"
            tf_status = "2/3 TFs aligned"
        else:
            risk_percent = 0.5
            risk_color = "🔴"
            tf_status = "Mixed structure"

    reasons = tech_reasons + headlines
    return price, final_score, direction, risk_percent, risk_color, tf_status, reasons, df_1h, weekly_trend, daily_trend, h4_trend

# ========================= UI =========================
st.title("🚀 AI Forex Intelligence Terminal")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC • All 32 pairs available")

if st.button("🔄 Refresh All Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# ====================== SIDEBAR ======================
st.sidebar.header("🎛️ Manage Dashboard")
st.sidebar.caption("Search • Add • Remove • All pairs visible by default")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(ALL_PAIRS.keys())  # ← ALL 32 shown by default!

selected = st.sidebar.multiselect(
    "Pairs on Dashboard",
    options=list(ALL_PAIRS.keys()),
    default=st.session_state.watchlist
)

if selected != st.session_state.watchlist:
    st.session_state.watchlist = selected
    st.rerun()

compact_mode = st.sidebar.toggle("Compact Mode (for 15+ pairs)", value=len(st.session_state.watchlist) > 10)

st.sidebar.divider()
st.sidebar.success(f"Showing **{len(st.session_state.watchlist)}** pairs")

# ====================== MAIN DASHBOARD (ALL visible) ======================
cols = st.columns(2) if compact_mode else st.columns(3)
col_idx = 0

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, score, direction, risk_percent, risk_color, tf_status, reasons, df, w, d, h4 = analyze_pair(pair, symbol, keywords)

    with cols[col_idx % len(cols)]:
        st.markdown(f"<div style='background:#1f2937;padding:18px;border-radius:12px;margin:8px;'>", unsafe_allow_html=True)
        
        st.subheader(f"**{pair}**")
        if price:
            st.metric("Live Price", f"{price:.5f}")
        
        st.write(f"**Direction:** {direction} ({score}%)")
        st.progress(score / 100)
        
        st.markdown(f"**{risk_color} Recommended Risk:** **{risk_percent}%**")
        st.caption(tf_status)
        
        if not compact_mode:
            with st.expander("📊 Market Structure", expanded=False):
                st.write(f"Weekly: **{w}** | Daily: **{d}** | 4H: **{h4}**")
            with st.expander("📋 Key Drivers", expanded=False):
                for r in reasons[:8]: st.write(f"• {r}")
            with st.expander("📈 1H Chart", expanded=False):
                if df is not None:
                    st.line_chart(df.set_index("datetime")["close"])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    col_idx += 1

st.caption("⚠️ Educational tool only • Not financial advice • Always do your own analysis")
