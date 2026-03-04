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

# ========================= ALL 20 PAIRS =========================
ALL_PAIRS = {
    "EURUSD": ["EUR/USD", ["EUR", "USD", "ECB", "Fed", "Euro", "Dollar", "inflation"]],
    "GBPUSD": ["GBP/USD", ["GBP", "USD", "BOE", "Fed", "Pound", "Dollar", "UK"]],
    "USDJPY": ["USD/JPY", ["USD", "JPY", "BOJ", "Fed", "Dollar", "Yen", "Japan"]],
    "USDCAD": ["USD/CAD", ["USD", "CAD", "BOC", "Fed", "Loonie", "Canada"]],
    "AUDUSD": ["AUD/USD", ["AUD", "USD", "RBA", "Fed", "Aussie", "Australia"]],
    "NZDUSD": ["NZD/USD", ["NZD", "USD", "RBNZ", "Fed", "Kiwi", "New Zealand"]],
    "USDCHF": ["USD/CHF", ["USD", "CHF", "SNB", "Fed", "Franc", "Switzerland"]],
    "EURGBP": ["EUR/GBP", ["EUR", "GBP", "ECB", "BOE", "Euro", "Pound"]],
    "EURJPY": ["EUR/JPY", ["EUR", "JPY", "ECB", "BOJ", "Euro", "Yen"]],
    "GBPJPY": ["GBP/JPY", ["GBP", "JPY", "BOE", "BOJ", "Pound", "Yen"]],
    "EURCHF": ["EUR/CHF", ["EUR", "CHF", "ECB", "SNB", "Euro", "Franc"]],
    "EURAUD": ["EUR/AUD", ["EUR", "AUD", "ECB", "RBA", "Euro", "Aussie"]],
    "GBPAUD": ["GBP/AUD", ["GBP", "AUD", "BOE", "RBA", "Pound", "Aussie"]],
    "AUDJPY": ["AUD/JPY", ["AUD", "JPY", "RBA", "BOJ", "Aussie", "Yen"]],
    "USDCNH": ["USD/CNH", ["USD", "CNH", "PBOC", "Fed", "China", "Yuan"]],
    "XAUUSD": ["XAU/USD", ["Gold", "USD", "Dollar", "Treasury", "Fed", "inflation"]],
    "XAGUSD": ["XAG/USD", ["Silver", "USD", "Dollar", "Fed", "inflation"]],
    "EURCAD": ["EUR/CAD", ["EUR", "CAD", "ECB", "BOC", "Euro", "Loonie"]],
    "GBPCAD": ["GBP/CAD", ["GBP", "CAD", "BOE", "BOC", "Pound", "Loonie"]],
    "AUDCAD": ["AUD/CAD", ["AUD", "CAD", "RBA", "BOC", "Aussie", "Loonie"]],
}

# ========================= CACHED HELPERS =========================
@st.cache_data(ttl=90)
def get_price(symbol: str):
    try:
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}"
        r = requests.get(url, timeout=10).json()
        return float(r.get("price", 0)) if "price" in r else None
    except:
        return None

@st.cache_data(ttl=180)
def get_time_series(symbol: str):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=60&apikey={TWELVE_KEY}"
        r = requests.get(url, timeout=10).json()
        if "values" not in r:
            return None
        df = pd.DataFrame(r["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        return df
    except:
        return None

def calculate_technical_score(df):
    if df is None or len(df) < 15:
        return 50, ["⚠️ Technical data unavailable (rate limit)"]
    
    closes = df["close"]
    ema12 = closes.ewm(span=12).mean().iloc[-1]
    ema26 = closes.ewm(span=26).mean().iloc[-1]
    rsi = calculate_rsi(closes)
    
    score = 50
    reasons = []
    
    # EMA Trend
    if ema12 > ema26:
        score += 25
        reasons.append("EMA12 > EMA26 → Bullish trend")
    else:
        score -= 25
        reasons.append("EMA12 < EMA26 → Bearish trend")
    
    # RSI
    if rsi > 70:
        score -= 12
        reasons.append(f"RSI {rsi:.1f} → Overbought")
    elif rsi < 30:
        score += 12
        reasons.append(f"RSI {rsi:.1f} → Oversold")
    else:
        reasons.append(f"RSI {rsi:.1f} → Neutral")
    
    # Momentum (last 5 candles)
    momentum = (closes.iloc[-1] / closes.iloc[-6] - 1) * 100 if len(closes) >= 6 else 0
    if momentum > 0.3:
        score += 8
        reasons.append(f"Momentum +{momentum:.2f}%")
    elif momentum < -0.3:
        score -= 8
        reasons.append(f"Momentum {momentum:.2f}%")
    
    return max(10, min(90, int(score))), reasons

def calculate_rsi(prices, period=14):
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return float(100 - (100 / (1 + rs)).iloc[-1])
    except:
        return 50.0

@st.cache_data(ttl=300)
def get_rss_headlines():
    # (same 4 feeds as before)
    feeds = ["https://feeds.reuters.com/reuters/businessNews", "https://www.fxstreet.com/rss/news",
             "https://www.marketwatch.com/rss/topstories", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"]
    headlines = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:15]:
                title = item.find("title").text
                if title: headlines.append(title.strip())
        except:
            continue
    return headlines

def finbert(text):
    # (same as before)
    try:
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=15)
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            scores = result[0]
            pos = next((s["score"] for s in scores if s["label"] == "positive"), 0)
            neg = next((s["score"] for s in scores if s["label"] == "negative"), 0)
            return int(50 + (pos - neg) * 50)
        return 50
    except:
        return 50

@st.cache_data(ttl=300)
def news_analysis(keywords):
    all_headlines = get_rss_headlines()
    filtered = [h for h in all_headlines if any(k.lower() in h.lower() for k in keywords)]
    if not filtered:
        return 50, ["No specific news found (neutral)"]
    combined = " | ".join(filtered[:6])
    score = finbert(combined)
    return score, filtered[:6]

# ========================= CORE ANALYSIS =========================
def analyze_pair(pair_name: str, symbol: str, keywords: list):
    price = get_price(symbol)
    df = get_time_series(symbol)
    
    tech_score, tech_reasons = calculate_technical_score(df)
    news_score, headlines = news_analysis(keywords)
    
    final_score = int(0.65 * tech_score + 0.35 * news_score)
    final_score = max(10, min(90, final_score))
    
    direction = "Bullish" if final_score >= 68 else "Bearish" if final_score <= 38 else "Neutral"
    risk = "High" if final_score >= 80 or final_score <= 20 else "Medium"
    
    reasons = tech_reasons + headlines
    return price, final_score, direction, risk, reasons, df, tech_score, news_score

# ========================= UI =========================
st.title("🚀 AI Forex Intelligence Terminal")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC • Free APIs")

if st.button("🔄 Refresh All Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# ====================== SIDEBAR WATCHLIST ======================
st.sidebar.header("📋 Your Watchlist")
st.sidebar.caption("Search & select pairs to show on dashboard")

# Default selection (only 5 to avoid rate limit on first load)
default_pairs = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "GBPJPY"]

if "watchlist" not in st.session_state:
    st.session_state.watchlist = default_pairs

selected = st.sidebar.multiselect(
    "Active Pairs (searchable)",
    options=list(ALL_PAIRS.keys()),
    default=st.session_state.watchlist,
    help="Search by typing (e.g. EUR, Gold, JPY)"
)

if selected != st.session_state.watchlist:
    st.session_state.watchlist = selected
    st.rerun()

st.sidebar.divider()
st.sidebar.warning("💡 Free tier tip: Keep ≤8 pairs for instant refresh. Wait 60s between full refreshes.")

# ====================== MAIN DASHBOARD ======================
cols = st.columns(3)
col_idx = 0
all_scores = []

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, score, direction, risk, reasons, df, tech, news = analyze_pair(pair, symbol, keywords)
    all_scores.append(score)

    with cols[col_idx % 3]:
        st.markdown(f"<div style='background:#1f2937;padding:18px;border-radius:12px;margin:8px;'>", unsafe_allow_html=True)
        
        st.subheader(f"**{pair}**")
        if price:
            st.metric("Live Price", f"{price:.5f}")
        else:
            st.error("Price unavailable")

        dir_color = "#22c55e" if direction == "Bullish" else "#ef4444" if direction == "Bearish" else "#eab308"
        st.markdown(f"**Direction:** <span style='color:{dir_color};font-weight:bold'>{direction} ({score}%)</span>", unsafe_allow_html=True)
        
        st.progress(score / 100)
        st.caption(f"**Tech:** {tech}% | **News:** {news}%")

        if risk == "High":
            st.error(f"⚠️ Risk: {risk}")
        else:
            st.warning(f"Risk: {risk}")

        with st.expander("📋 Key Drivers", expanded=False):
            for r in reasons[:8]:
                st.write(f"• {r}")

        with st.expander("📊 1H Chart", expanded=False):
            if df is not None:
                st.line_chart(df.set_index("datetime")["close"], use_container_width=True)
            else:
                st.write("Chart unavailable")

        st.markdown("</div>", unsafe_allow_html=True)
    
    col_idx += 1

# ====================== OVERALL SENTIMENT ======================
st.markdown("---")
avg_score = sum(all_scores) / len(all_scores) if all_scores else 50
sentiment = "🟢 STRONG BULLISH" if avg_score > 60 else "🔴 STRONG BEARISH" if avg_score < 40 else "⚪ NEUTRAL"
st.metric("OVERALL MARKET SENTIMENT", f"{avg_score:.1f}%", sentiment)

st.caption("⚠️ Educational tool only — not financial advice. Markets are volatile. Always verify with your own analysis.")
