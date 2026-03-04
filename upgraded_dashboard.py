import streamlit as st
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

st.set_page_config(layout="wide")

TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]

pairs = {
    "EURUSD": ["EUR/USD", ["EUR", "USD", "ECB", "Fed", "Euro", "Dollar"]],
    "USDJPY": ["USD/JPY", ["USD", "JPY", "BOJ", "Fed", "Dollar", "Yen"]],
    "GBPUSD": ["GBP/USD", ["GBP", "USD", "BOE", "Fed", "Pound", "Dollar"]],
    "XAUUSD": ["XAU/USD", ["Gold", "USD", "Dollar", "Treasury", "Fed"]],
}

# =========================
# PRICE
# =========================
def get_price(symbol):
    try:
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}"
        r = requests.get(url, timeout=10).json()
        if "price" in r:
            return float(r["price"])
        return None
    except:
        return None

# =========================
# TECHNICAL
# =========================
def technical_score(symbol):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=200&apikey={TWELVE_KEY}"
        r = requests.get(url, timeout=10).json()

        if "values" not in r:
            return 50, ["No technical data"]

        df = pd.DataFrame(r["values"])
        df["close"] = df["close"].astype(float)
        df = df.sort_index(ascending=False)

        ema50 = df["close"].ewm(span=50).mean().iloc[-1]
        ema200 = df["close"].ewm(span=200).mean().iloc[-1]

        score = 50
        reasons = []

        if ema50 > ema200:
            score += 20
            reasons.append("EMA50 above EMA200 (Uptrend)")
        else:
            score -= 20
            reasons.append("EMA50 below EMA200 (Downtrend)")

        return max(1, min(100, int(score))), reasons

    except:
        return 50, ["Technical calculation error"]

# =========================
# FINBERT
# =========================
def finbert(text):
    try:
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=15)
        result = response.json()

        if isinstance(result, list):
            scores = result[0]
            pos = 0
            neg = 0
            for s in scores:
                if s["label"] == "positive":
                    pos = s["score"]
                if s["label"] == "negative":
                    neg = s["score"]
            return int(50 + (pos - neg) * 50)

        return 50
    except:
        return 50

# =========================
# RSS NEWS ENGINE
# =========================
def get_rss_headlines():
    feeds = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.fxstreet.com/rss/news",
        "https://www.marketwatch.com/rss/topstories"
    ]

    headlines = []

    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            root = ET.fromstring(r.content)

            for item in root.findall(".//item")[:10]:
                title = item.find("title").text
                headlines.append(title)
        except:
            continue

    return headlines

def news_analysis(keywords):
    try:
        all_headlines = get_rss_headlines()
        filtered = []

        for h in all_headlines:
            for k in keywords:
                if k.lower() in h.lower():
                    filtered.append(h)
                    break

        if not filtered:
            return 50, ["No relevant financial news"]

        combined = " ".join(filtered[:5])
        score = finbert(combined)

        return score, filtered[:5]

    except:
        return 50, ["News processing error"]

# =========================
# FINAL ENGINE
# =========================
def analyze(pair_name, symbol, keywords):
    price = get_price(symbol)
    tech_score, tech_reason = technical_score(symbol)
    news_score, headlines = news_analysis(keywords)

    final = int((0.6 * tech_score) + (0.4 * news_score))
    final = max(1, min(100, final))

    if final > 55:
        direction = "Bullish"
    elif final < 45:
        direction = "Bearish"
    else:
        direction = "Neutral"

    risk = "Medium"
    if final > 70 or final < 30:
        risk = "High"

    reasons = tech_reason + headlines

    return price, final, direction, risk, reasons

# =========================
# UI
# =========================
st.title("AI Forex Intelligence Terminal")

cols = st.columns(2)
i = 0

for pair, data in pairs.items():
    symbol = data[0]
    keywords = data[1]

    with cols[i % 2]:
        price, score, direction, risk, reasons = analyze(pair, symbol, keywords)

        st.subheader(pair)

        if price:
            st.metric("Live Price", round(price, 5))
        else:
            st.error("Price unavailable")

        st.write(f"Direction: **{direction} ({score}%)**")
        st.progress(score)

        if risk == "High":
            st.error(f"Risk: {risk}")
        else:
            st.warning(f"Risk: {risk}")

        st.markdown("**Drivers:**")
        for r in reasons:
            st.write("-", r)

    i += 1
