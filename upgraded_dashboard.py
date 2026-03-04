import streamlit as st
import requests
import pandas as pd
import numpy as np
from textblob import TextBlob

st.set_page_config(layout="wide")

TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
NEWS_KEY = st.secrets["NEWS_API_KEY"]

pairs = {
    "EURUSD": "EUR/USD",
    "USDJPY": "USD/JPY",
    "GBPUSD": "GBP/USD",
    "EURJPY": "EUR/JPY",
    "XAUUSD": "XAU/USD",
    "EURGBP": "EUR/GBP"
}

# -----------------------------
# PRICE FETCHING
# -----------------------------
def get_price(symbol):
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}"
    r = requests.get(url).json()
    if "price" in r:
        return float(r["price"])
    return None


# -----------------------------
# TECHNICAL ANALYSIS
# -----------------------------
def get_technical_score(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=100&apikey={TWELVE_KEY}"
    r = requests.get(url).json()

    if "values" not in r:
        return 50

    df = pd.DataFrame(r["values"])
    df["close"] = df["close"].astype(float)
    df = df.sort_index(ascending=False)

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # EMA Trend
    ema50 = df["close"].ewm(span=50).mean().iloc[-1]
    ema200 = df["close"].ewm(span=200).mean().iloc[-1]

    score = 50

    if current_rsi < 30:
        score += 15
    elif current_rsi > 70:
        score -= 15

    if ema50 > ema200:
        score += 20
    else:
        score -= 20

    return max(1, min(100, score))


# -----------------------------
# NEWS SENTIMENT
# -----------------------------
def get_news_sentiment(pair):
    base = pair[:3]
    url = f"https://newsapi.org/v2/everything?q={base}&language=en&apiKey={NEWS_KEY}"
    r = requests.get(url).json()

    if "articles" not in r:
        return 50, "No recent news data."

    sentiments = []
    reasons = []

    for article in r["articles"][:5]:
        title = article["title"]
        blob = TextBlob(title)
        polarity = blob.sentiment.polarity
        sentiments.append(polarity)
        reasons.append(f"- {title}")

    if not sentiments:
        return 50, "No sentiment detected."

    avg_sentiment = np.mean(sentiments)

    score = 50 + (avg_sentiment * 50)
    score = max(1, min(100, score))

    reason_text = "\n".join(reasons[:3])

    return score, reason_text


# -----------------------------
# COMBINED SCORING ENGINE
# -----------------------------
def analyze_pair(pair, symbol):
    price = get_price(symbol)
    technical = get_technical_score(symbol)
    sentiment, reason = get_news_sentiment(pair)

    final_score = int((0.6 * technical) + (0.4 * sentiment))

    if final_score > 55:
        direction = "Bullish"
    elif final_score < 45:
        direction = "Bearish"
    else:
        direction = "Neutral"

    risk = "Low"
    if 40 <= final_score <= 60:
        risk = "Medium"
    else:
        risk = "High"

    return price, final_score, direction, risk, reason


# -----------------------------
# UI
# -----------------------------
st.title("AI Forex Intelligence Dashboard")

cols = st.columns(3)

i = 0
for pair, symbol in pairs.items():
    price, score, direction, risk, reason = analyze_pair(pair, symbol)

    with cols[i % 3]:
        st.subheader(pair)

        if price:
            st.metric("Price", round(price, 5))
        else:
            st.error("Live price unavailable")

        st.write(f"Direction: **{direction} ({score}%)**")
        st.progress(score)

        if risk == "Low":
            st.success(f"Risk: {risk}")
        elif risk == "Medium":
            st.warning(f"Risk: {risk}")
        else:
            st.error(f"Risk: {risk}")

        st.markdown("**Reasoning:**")
        st.markdown(reason)

    i += 1
