import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
NEWS_KEY = st.secrets["NEWS_API_KEY"]
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]

pairs = {
    "EURUSD": "EUR/USD",
    "USDJPY": "USD/JPY",
    "GBPUSD": "GBP/USD",
    "EURJPY": "EUR/JPY",
    "XAUUSD": "XAU/USD",
    "EURGBP": "EUR/GBP"
}

# ----------------------------
# PRICE
# ----------------------------
def get_price(symbol):
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}"
    r = requests.get(url).json()
    if "price" in r:
        return float(r["price"])
    return None

# ----------------------------
# TECHNICAL ENGINE
# ----------------------------
def technical_score(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=200&apikey={TWELVE_KEY}"
    r = requests.get(url).json()
    if "values" not in r:
        return 50, "No technical data"

    df = pd.DataFrame(r["values"])
    df["close"] = df["close"].astype(float)
    df = df.sort_index(ascending=False)

    ema50 = df["close"].ewm(span=50).mean().iloc[-1]
    ema200 = df["close"].ewm(span=200).mean().iloc[-1]

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    rsi = 100 - (100 / (1 + rs))
    rsi_now = rsi.iloc[-1]

    score = 50
    reasons = []

    if ema50 > ema200:
        score += 20
        reasons.append("EMA50 above EMA200 (Uptrend)")
    else:
        score -= 20
        reasons.append("EMA50 below EMA200 (Downtrend)")

    if rsi_now < 30:
        score += 15
        reasons.append("RSI oversold")
    elif rsi_now > 70:
        score -= 15
        reasons.append("RSI overbought")

    return max(1, min(100, score)), reasons

# ----------------------------
# FINBERT SENTIMENT
# ----------------------------
def finbert_sentiment(text):
    API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {HF_KEY}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    result = response.json()

    if isinstance(result, list):
        scores = result[0]
        for item in scores:
            if item["label"] == "positive":
                return 50 + item["score"] * 50
            if item["label"] == "negative":
                return 50 - item["score"] * 50
    return 50

def news_sentiment(pair):
    base = pair[:3]
    url = f"https://newsapi.org/v2/everything?q={base}&language=en&apiKey={NEWS_KEY}"
    r = requests.get(url).json()

    if "articles" not in r or len(r["articles"]) == 0:
        return 50, ["No recent financial news"]

    articles = r["articles"][:3]
    headlines = [a["title"] for a in articles]

    combined = " ".join(headlines)
    score = finbert_sentiment(combined)

    return max(1, min(100, int(score))), headlines

# ----------------------------
# USD STRENGTH PROXY
# ----------------------------
def usd_macro_bias():
    pairs_check = ["EUR/USD", "GBP/USD", "USD/JPY"]
    values = []

    for p in pairs_check:
        url = f"https://api.twelvedata.com/price?symbol={p}&apikey={TWELVE_KEY}"
        r = requests.get(url).json()
        if "price" in r:
            values.append(float(r["price"]))

    if len(values) < 3:
        return 50

    return 60 if values[0] < 1 else 40  # simple bias proxy

# ----------------------------
# FINAL ENGINE
# ----------------------------
def analyze(pair, symbol):
    price = get_price(symbol)
    tech_score, tech_reasons = technical_score(symbol)
    news_score, headlines = news_sentiment(pair)
    macro = usd_macro_bias()

    final = int((0.45 * tech_score) + (0.35 * news_score) + (0.20 * macro))

    direction = "Neutral"
    if final > 55:
        direction = "Bullish"
    elif final < 45:
        direction = "Bearish"

    risk = "Medium"
    if final > 70 or final < 30:
        risk = "High"
    elif 45 <= final <= 55:
        risk = "Low"

    reasons = tech_reasons + headlines

    return price, final, direction, risk, reasons

# ----------------------------
# UI
# ----------------------------
st.title("Professional AI Forex Intelligence Terminal")

cols = st.columns(3)
i = 0

for pair, symbol in pairs.items():
    with cols[i % 3]:
        price, score, direction, risk, reasons = analyze(pair, symbol)

        st.subheader(pair)

        if price:
            st.metric("Live Price", round(price, 5))
        else:
            st.error("Price unavailable")

        st.write(f"Direction: **{direction} ({score}%)**")
        st.progress(score)

        if risk == "Low":
            st.success(f"Risk: {risk}")
        elif risk == "Medium":
            st.warning(f"Risk: {risk}")
        else:
            st.error(f"Risk: {risk}")

        st.markdown("**Drivers:**")
        for r in reasons:
            st.write("-", r)

    i += 1
