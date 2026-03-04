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

# ========================= PAIRS (All Major Players) =========================
pairs = {
    "EURUSD": ["EUR/USD", ["EUR", "USD", "ECB", "Fed", "Euro", "Dollar", "inflation"]],
    "GBPUSD": ["GBP/USD", ["GBP", "USD", "BOE", "Fed", "Pound", "Dollar", "UK"]],
    "USDJPY": ["USD/JPY", ["USD", "JPY", "BOJ", "Fed", "Dollar", "Yen", "Japan"]],
    "USDCAD": ["USD/CAD", ["USD", "CAD", "BOC", "Fed", "Loonie", "Canada"]],
    "AUDUSD": ["AUD/USD", ["AUD", "USD", "RBA", "Fed", "Aussie", "Australia"]],
    "NZDUSD": ["NZD/USD", ["NZD", "USD", "RBNZ", "Fed", "Kiwi", "New Zealand"]],
    "USDCHF": ["USD/CHF", ["USD", "CHF", "SNB", "Fed", "Franc", "Switzerland"]],
    "XAUUSD": ["XAU/USD", ["Gold", "USD", "Dollar", "Treasury", "Fed", "inflation"]],
}

# ========================= CACHED HELPERS =========================
@st.cache_data(ttl=60)
def get_price(symbol: str):
    try:
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}"
        r = requests.get(url, timeout=10).json()
        return float(r.get("price", 0)) if "price" in r else None
    except:
        return None

@st.cache_data(ttl=60)
def get_time_series(symbol: str):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=100&apikey={TWELVE_KEY}"
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

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except:
        return 50.0

@st.cache_data(ttl=300)
def get_rss_headlines() -> list:
    feeds = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.fxstreet.com/rss/news",
        "https://www.marketwatch.com/rss/topstories",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    ]
    headlines = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:12]:
                title = item.find("title").text
                if title:
                    headlines.append(title.strip())
        except:
            continue
    return headlines

def finbert(text: str) -> int:
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
def news_analysis(keywords: list) -> tuple:
    all_headlines = get_rss_headlines()
    filtered = [h for h in all_headlines if any(k.lower() in h.lower() for k in keywords)]
    if not filtered:
        return 50, ["No relevant news found"]
    combined = " | ".join(filtered[:6])
    score = finbert(combined)
    return score, filtered[:6]

# ========================= CORE ANALYSIS ENGINE =========================
def analyze_pair(pair_name: str, symbol: str, keywords: list):
    price = get_price(symbol)
    df = get_time_series(symbol)

    # Technical Score
    tech_score = 50
    tech_reasons = []
    if df is not None and len(df) > 30:
        closes = df["close"]
        ema50 = closes.ewm(span=50).mean().iloc[-1]
        ema200 = closes.ewm(span=200).mean().iloc[-1]
        rsi = calculate_rsi(closes)

        if ema50 > ema200:
            tech_score += 25
            tech_reasons.append("EMA50 > EMA200 → Strong uptrend")
        else:
            tech_score -= 25
            tech_reasons.append("EMA50 < EMA200 → Downtrend")

        if rsi > 70:
            tech_score -= 15
            tech_reasons.append(f"RSI {rsi:.1f} → Overbought")
        elif rsi < 30:
            tech_score += 15
            tech_reasons.append(f"RSI {rsi:.1f} → Oversold")
        else:
            tech_reasons.append(f"RSI {rsi:.1f} → Neutral")

    # News Score
    news_score, headlines = news_analysis(keywords)

    # Final AI Score
    final_score = int(0.65 * tech_score + 0.35 * news_score)
    final_score = max(10, min(90, final_score))

    if final_score >= 68:
        direction = "Bullish"
        risk = "High" if final_score >= 80 else "Medium"
    elif final_score <= 38:
        direction = "Bearish"
        risk = "High" if final_score <= 20 else "Medium"
    else:
        direction = "Neutral"
        risk = "Medium"

    reasons = tech_reasons + headlines
    return price, final_score, direction, risk, reasons, df

# ========================= UI =========================
st.title("🚀 AI Forex Intelligence Terminal")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC • Powered by free APIs")

if st.button("🔄 Refresh All Data Now", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Overall Market Sentiment
all_scores = []
summary_data = []

cols = st.columns(3)
col_idx = 0

for pair, (symbol, keywords) in pairs.items():
    price, score, direction, risk, reasons, df = analyze_pair(pair, symbol, keywords)
    all_scores.append(score)
    summary_data.append({"Pair": pair, "Price": round(price, 5) if price else "—", 
                         "AI Score": score, "Direction": direction, "Risk": risk})

    with cols[col_idx % 3]:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        
        st.subheader(f"**{pair}**")
        if price:
            st.metric("Live Price", f"{price:.5f}")
        else:
            st.error("Price unavailable")

        dir_color = "bullish" if direction == "Bullish" else "bearish" if direction == "Bearish" else "neutral"
        st.markdown(f"**Direction:** <span class='{dir_color}'>{direction} ({score}%)</span>", unsafe_allow_html=True)
        
        st.progress(score / 100)
        
        if risk == "High":
            st.error(f"⚠️ Risk: {risk}")
        else:
            st.warning(f"Risk: {risk}")

        with st.expander("📋 Key Drivers & News", expanded=False):
            for r in reasons[:8]:
                st.write(f"• {r}")

        with st.expander("📊 1H Price Chart", expanded=False):
            if df is not None:
                chart_df = df.set_index("datetime")[["close"]]
                st.line_chart(chart_df, use_container_width=True)
            else:
                st.write("Chart data unavailable")

        st.markdown("</div>", unsafe_allow_html=True)
    
    col_idx += 1

# Summary Table + Overall Sentiment
st.markdown("---")
st.subheader("📊 Market Overview")

avg_score = sum(all_scores) / len(all_scores)
sentiment = "🟢 BULLISH" if avg_score > 55 else "🔴 BEARISH" if avg_score < 45 else "⚪ NEUTRAL"

st.metric("OVERALL MARKET SENTIMENT", f"{avg_score:.1f}%", sentiment)

summary_df = pd.DataFrame(summary_data)
st.dataframe(
    summary_df,
    use_container_width=True,
    column_config={
        "AI Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100),
        "Direction": st.column_config.TextColumn("Direction")
    },
    hide_index=True
)

st.markdown("---")
st.caption("⚠️ **Disclaimer:** This dashboard is for educational and entertainment purposes only. It is NOT financial advice. Past performance does not guarantee future results. Always do your own research and consult a licensed advisor before trading. Markets are highly volatile.")
