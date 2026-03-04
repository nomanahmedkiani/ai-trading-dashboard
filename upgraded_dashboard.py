import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# ==============================
# API KEYS
# ==============================
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

# ==============================
# SAFE PRICE FETCH
# ==============================
def get_price(symbol):
    try:
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}"
        r = requests.get(url, timeout=10).json()

        if "price" in r:
            return float(r["price"])
        return None
    except:
        return None


# ==============================
# TECHNICAL ENGINE
# ==============================
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

        return max(1, min(100, int(score))), reasons

    except:
        return 50, ["Technical calculation error"]


# ==============================
# SAFE FINBERT
# ==============================
def finbert_sentiment(text):
    try:
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {HF_KEY}"}

        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=15
        )

        result = response.json()

        if isinstance(result, dict) and "error" in result:
            return 50

        if isinstance(result, list):
            if isinstance(result[0], list):
                scores = result[0]
            else:
                scores = result

            positive = 0
            negative = 0

            for item in scores:
                label = item["label"].lower()
                if label == "positive":
                    positive = item["score"]
                if label == "negative":
                    negative = item["score"]

            return int(50 + (positive - negative) * 50)

        return 50

    except:
        return 50


# ==============================
# SAFE NEWS
# ==============================
def news_sentiment(pair):
    try:
        base = pair[:3]
        url = f"https://newsapi.org/v2/everything?q={base}&language=en&apiKey={NEWS_KEY}"
        r = requests.get(url, timeout=10).json()

        if not isinstance(r, dict):
            return 50, ["News API invalid response"]

        if r.get("status") != "ok":
            return 50, [f"News API error"]

        articles = r.get("articles", [])

        if not articles:
            return 50, ["No recent financial news"]

        headlines = []
        for a in articles[:3]:
            title = a.get("title")
            if title:
                headlines.append(title)

        if not headlines:
            return 50, ["Headlines unavailable"]

        combined_text = " ".join(headlines)
        score = finbert_sentiment(combined_text)

        return int(score), headlines

    except:
        return 50, ["News processing error"]


# ==============================
# SIMPLE USD MACRO PROXY
# ==============================
def usd_macro_bias():
    try:
        url = f"https://api.twelvedata.com/price?symbol=EUR/USD&apikey={TWELVE_KEY}"
        r = requests.get(url).json()

        if "price" not in r:
            return 50

        price = float(r["price"])

        if price < 1.08:
            return 60
        else:
            return 40

    except:
        return 50


# ==============================
# FINAL ENGINE
# ==============================
def analyze(pair, symbol):
    price = get_price(symbol)
    tech_score, tech_reasons = technical_score(symbol)
    news_score, headlines = news_sentiment(pair)
    macro = usd_macro_bias()

    final_score = int(
        (0.45 * tech_score) +
        (0.35 * news_score) +
        (0.20 * macro)
    )

    final_score = max(1, min(100, final_score))

    if final_score > 55:
        direction = "Bullish"
    elif final_score < 45:
        direction = "Bearish"
    else:
        direction = "Neutral"

    if final_score > 70 or final_score < 30:
        risk = "High"
    elif 45 <= final_score <= 55:
        risk = "Low"
    else:
        risk = "Medium"

    reasons = tech_reasons + headlines

    return price, final_score, direction, risk, reasons


# ==============================
# UI
# ==============================
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
