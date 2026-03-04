# upgraded_dashboard.py
import streamlit as st
import requests
from textblob import TextBlob

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📊 AI Trading Dashboard - Real-time Market Analysis")

# Main pairs always visible
main_pairs = ["EURUSD", "USDJPY", "GBPUSD", "EURJPY", "XAUUSD", "US30", "US100", "EURGBP"]
all_pairs = ["AUDUSD", "USDCAD", "NZDUSD", "GBPJPY", "EURCHF"] + main_pairs

# Read API keys from Streamlit secrets
TWELVE_DATA_API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

# -----------------------
# Helper Functions
# -----------------------
def get_price(pair):
    try:
        if pair in ["US30", "US100"]:
            symbol = "^DJI" if pair=="US30" else "^IXIC"
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
            data = requests.get(url).json()
            return data["quoteResponse"]["result"][0]["regularMarketPrice"]
        else:
            url = f"https://api.twelvedata.com/price?symbol={pair}&apikey={TWELVE_DATA_API_KEY}"
            data = requests.get(url).json()
            return float(data.get("price", 0))
    except:
        return None

def get_news(pair):
    try:
        url = f"https://newsapi.org/v2/everything?q={pair}&pageSize=3&apiKey={NEWS_API_KEY}"
        data = requests.get(url).json()
        return [a["title"] for a in data.get("articles", [])]
    except:
        return []

def analyze_sentiment(headlines):
    if not headlines:
        return "Neutral"
    score = sum(TextBlob(h).sentiment.polarity for h in headlines)
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def ai_probability(price, sentiment):
    base = 50
    if sentiment == "Positive":
        base += 15
    elif sentiment == "Negative":
        base -= 15
    return min(max(base, 0), 100)

def risk_factor(sentiment):
    if sentiment == "Positive":
        return "Low"
    elif sentiment == "Neutral":
        return "Medium"
    else:
        return "High"

def risk_color(risk):
    return {"Low":"#2ECC71","Medium":"#F1C40F","High":"#E74C3C"}.get(risk, "#BDC3C7")

# -----------------------
# Session State
# -----------------------
if "extra_pairs" not in st.session_state:
    st.session_state.extra_pairs = []

extra_pair = st.selectbox("Add More Pair", [p for p in all_pairs if p not in main_pairs + st.session_state.extra_pairs])
if st.button("Add Pair") and extra_pair:
    st.session_state.extra_pairs.append(extra_pair)

display_pairs = main_pairs + st.session_state.extra_pairs

# -----------------------
# Dashboard Grid
# -----------------------
cols_per_row = 4
rows = (len(display_pairs) + cols_per_row - 1) // cols_per_row

for i in range(rows):
    cols = st.columns(cols_per_row)
    for j, pair in enumerate(display_pairs[i*cols_per_row:(i+1)*cols_per_row]):
        price = get_price(pair)
        news = get_news(pair)
        sentiment = analyze_sentiment(news)
        prob = ai_probability(price, sentiment)
        risk = risk_factor(sentiment)
        color = risk_color(risk)

        with cols[j]:
            st.markdown(f"### {pair}")
            st.metric("Price", f"{price}")
            st.write(f"**Direction:** {'Bullish' if prob>=50 else 'Bearish'} ({prob}%)")
            st.progress(prob/100)
            st.markdown(f"<div style='background-color:{color};padding:5px;border-radius:5px;text-align:center;color:white'>Risk: {risk}</div>", unsafe_allow_html=True)
            st.write(f"**Sentiment:** {sentiment}")
            if news:
                st.write("**Top Headlines:**")
                for n in news:
                    st.write(f"- {n[:60]}{'...' if len(n)>60 else ''}")

st.experimental_rerun()
