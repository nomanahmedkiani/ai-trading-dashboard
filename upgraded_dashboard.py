import streamlit as st
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

st.set_page_config(page_title="AI Forex Intelligence Terminal", layout="wide", page_icon="📈")

# ========================= SECRETS =========================
TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# ========================= PAIRS =========================
ALL_PAIRS = {
    "EURUSD": ["EUR/USD", ["EUR", "USD", "ECB", "Fed"]],
    "GBPUSD": ["GBP/USD", ["GBP", "USD", "BOE", "Fed"]],
    "USDJPY": ["USD/JPY", ["USD", "JPY", "BOJ", "Fed"]],
    "USDCAD": ["USD/CAD", ["USD", "CAD", "BOC", "Fed"]],
}

# ========================= PRICE =========================
@st.cache_data(ttl=60)
def get_price(symbol):
    try:
        r = requests.get(
            f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}",
            timeout=10
        ).json()
        return float(r["price"]) if "price" in r else None
    except:
        return None

# ========================= BASE DATA (1H ONLY) =========================
@st.cache_data(ttl=180)
def get_base_data(symbol):
    try:
        r = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=1000&apikey={TWELVE_KEY}",
            timeout=15
        ).json()

        if "values" not in r:
            return None

        df = pd.DataFrame(r["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df["close"] = df["close"].astype(float)
        df.set_index("datetime", inplace=True)

        return df
    except:
        return None

# ========================= MARKET STRUCTURE (FIXED) =========================
def get_tf_structure(symbol, tf):
    df = get_base_data(symbol)

    if df is None or len(df) < 300:
        return "Neutral (insufficient data)"

    if tf == "4h":
        data = df["close"].resample("4H").last()
    elif tf == "1day":
        data = df["close"].resample("1D").last()
    elif tf == "1week":
        data = df["close"].resample("1W").last()
    else:
        return "Neutral"

    data = data.dropna()

    if len(data) < 50:
        return "Neutral (insufficient data)"

    sma20 = data.rolling(20).mean().iloc[-1]
    sma50 = data.rolling(50).mean().iloc[-1]
    last = data.iloc[-1]

    if last > sma20 > sma50:
        return "Bullish Structure"
    elif last < sma20 < sma50:
        return "Bearish Structure"
    else:
        return "Sideways / Transition"

# ========================= NEWS =========================
@st.cache_data(ttl=180)
def get_news(symbol):
    try:
        r = requests.get(
            f"https://api.twelvedata.com/news?symbol={symbol}&apikey={TWELVE_KEY}&limit=10",
            timeout=10
        ).json()

        news = []
        for item in r.get("data", []):
            news.append(item.get("title", ""))

        return news
    except:
        return []

# ========================= FINBERT =========================
def finbert(text):
    try:
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        result = requests.post(API_URL, headers=headers, json={"inputs": text[:1200]}, timeout=15).json()

        if isinstance(result, list) and result:
            scores = result[0]
            pos = next((s["score"] for s in scores if s["label"] == "positive"), 0)
            neg = next((s["score"] for s in scores if s["label"] == "negative"), 0)
            score = int(50 + (pos - neg) * 100)
            return max(1, min(100, score))
        return 50
    except:
        return 50

# ========================= ANALYSIS =========================
def analyze(pair, symbol, keywords):
    price = get_price(symbol)

    news = get_news(symbol)
    filtered = [n for n in news if any(k.lower() in n.lower() for k in keywords)]

    score = finbert(" | ".join(filtered)) if filtered else 50

    direction = "Bullish" if score > 55 else "Bearish" if score < 45 else "Neutral"
    confidence = score if direction == "Bullish" else 100 - score

    weekly = get_tf_structure(symbol, "1week")
    daily = get_tf_structure(symbol, "1day")
    h4 = get_tf_structure(symbol, "4h")

    return price, direction, confidence, weekly, daily, h4, filtered[:5]

# ========================= UI =========================
st.title("🚀 AI Forex Intelligence Terminal")
st.caption(f"Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

cols = st.columns(2)

i = 0
for pair in ALL_PAIRS:
    symbol, keywords = ALL_PAIRS[pair]

    price, direction, confidence, weekly, daily, h4, reasons = analyze(pair, symbol, keywords)

    with cols[i % 2]:
        st.markdown("----")
        st.subheader(pair)

        if price:
            st.metric("Live Price", f"{price:.5f}")

        st.write(f"**Bias:** {direction}")
        st.write(f"**AI Confidence:** {confidence}%")

        with st.expander("📈 Market Structure"):
            st.write(f"Weekly: {weekly}")
            st.write(f"Daily: {daily}")
            st.write(f"4H: {h4}")

        with st.expander("📰 Key Headlines"):
            for r in reasons:
                st.write("•", r)

    i += 1

st.caption("Educational Tool Only — Not Financial Advice")
