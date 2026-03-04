import streamlit as st
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="AI Macro Forex Desk", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

# ── Secrets ────────────────────────────────────────────────────────────────
TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# ── All pairs (you can add/remove as needed) ───────────────────────────────
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
    "XAUUSD": ["XAU/USD", ["Gold", "USD", "Fed", "Treasury"]],
    # ... add the rest of your pairs here if you want more ...
}

# ── Price & Time Series ────────────────────────────────────────────────────
@st.cache_data(ttl=90)
def get_price(symbol: str):
    try:
        r = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}", timeout=10).json()
        return float(r.get("price", 0)) if "price" in r else None
    except:
        return None

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
    except:
        return None

# ── Technical scoring (simplified version) ─────────────────────────────────
def calculate_technical_score(df):
    if df is None or len(df) < 15:
        return 50, ["Limited data"]
    closes = df["close"]
    ema12 = closes.ewm(span=12).mean().iloc[-1]
    ema26 = closes.ewm(span=26).mean().iloc[-1]
    score = 50
    reasons = []
    if ema12 > ema26:
        score += 25
        reasons.append("EMA12 > EMA26 → Bullish")
    else:
        score -= 25
        reasons.append("EMA12 < EMA26 → Bearish")
    return max(15, min(85, int(score))), reasons

# ── RSS + FinBERT (simplified) ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_rss_headlines():
    feeds = [
        "https://www.forexlive.com/feed",
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
                title = item.find("title")
                if title is not None and title.text:
                    headlines.append(title.text.strip())
        except:
            continue
    return headlines

def finbert(text):
    try:
        url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        response = requests.post(url, headers=headers, json={"inputs": text}, timeout=15)
        result = response.json()
        if isinstance(result, list) and result:
            scores = result[0]
            pos = next((s["score"] for s in scores if s["label"] == "positive"), 0)
            neg = next((s["score"] for s in scores if s["label"] == "negative"), 0)
            return int(50 + (pos - neg) * 50)
        return 50
    except:
        return 50

@st.cache_data(ttl=300)
def news_analysis(keywords):
    headlines = get_rss_headlines()
    filtered = [h for h in headlines if any(k.lower() in h.lower() for k in keywords)]
    if not filtered:
        return 50, ["No relevant news"]
    combined = " | ".join(filtered[:5])
    score = finbert(combined)
    return score, filtered[:5]

# ── Multi-timeframe trend ──────────────────────────────────────────────────
@st.cache_data(ttl=600)
def get_tf_trend(symbol: str, interval: str) -> str:
    df = get_time_series_tf(symbol, interval, 40)
    if df is None or len(df) < 20:
        return "Neutral"
    ema50 = df["close"].ewm(span=50).mean().iloc[-1]
    return "Bullish" if df["close"].iloc[-1] > ema50 else "Bearish"

# ── Core analysis function ─────────────────────────────────────────────────
def analyze_pair(pair, symbol, keywords):
    price = get_price(symbol)
    df_1h = get_time_series_tf(symbol, "1h")
    tech_score, tech_reasons = calculate_technical_score(df_1h)
    news_score, headlines = news_analysis(keywords)
    final_score = int(0.65 * tech_score + 0.35 * news_score)
    final_score = max(15, min(85, final_score))
    direction = "Bullish" if final_score >= 68 else "Bearish" if final_score <= 38 else "Neutral"

    if direction == "Neutral":
        risk_percent = 0.5
        risk_color = "⚪"
        tf_status = "Neutral market — low risk"
        w = d = h4 = "—"
    else:
        w = get_tf_trend(symbol, "1week")
        d = get_tf_trend(symbol, "1day")
        h4 = get_tf_trend(symbol, "4h")
        aligned = sum(1 for t in [w, d, h4] if t == direction)
        if aligned == 3:
            risk_percent, risk_color = 3.0, "🟢"
        elif aligned == 2:
            risk_percent, risk_color = 1.5, "🟡"
        else:
            risk_percent, risk_color = 0.5, "🔴"
        tf_status = f"{aligned}/3 timeframes aligned"

    reasons = tech_reasons + headlines
    return price, final_score, direction, risk_percent, risk_color, tf_status, reasons, df_1h, w, d, h4

# ── MAIN APP STARTS HERE ───────────────────────────────────────────────────
st.title("🚀 AI Macro Forex Desk")
st.caption(f"Live • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")

if st.button("🔄 Refresh All", type="primary"):
    st.cache_data.clear()
    st.rerun()

# ── Watchlist initialization (THIS IS THE SAFE PLACE) ──────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(ALL_PAIRS.keys())[:12]   # start with 12 pairs

st.sidebar.header("🎛️ Watchlist")
selected = st.sidebar.multiselect(
    "Visible pairs",
    options=list(ALL_PAIRS.keys()),
    default=st.session_state.watchlist
)

if selected != st.session_state.watchlist:
    st.session_state.watchlist = selected
    st.rerun()

# ── Dashboard layout ───────────────────────────────────────────────────────
cols = st.columns(3)
col_idx = 0

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, score, direction, risk_percent, risk_color, tf_status, reasons, df, w, d, h4 = analyze_pair(pair, symbol, keywords)

    with cols[col_idx % 3]:
        st.markdown(f"""
        <div style="background:#1e293b; border-radius:12px; padding:16px; margin:8px; border:1px solid #334155;">
            <h3 style="margin:0; color:#cbd5e1;">{pair}</h3>
            <div style="font-size:2.4rem; font-weight:700; color:#60a5fa;">{f"{price:.5f}" if price else "—"}</div>
            <span style="background:{'rgba(34,197,94,0.2)' if direction=='Bullish' else 'rgba(239,68,68,0.2)' if direction=='Bearish' else 'rgba(234,179,8,0.2)'}; 
                         color:{' #22c55e' if direction=='Bullish' else ' #ef4444' if direction=='Bearish' else ' #eab308'}; 
                         padding:4px 12px; border-radius:999px; font-weight:600;">
                {direction} ({score}%)
            </span>
            <div style="margin:12px 0; font-size:1.1rem;">
                <strong>{risk_color} Risk:</strong> {risk_percent}% • {tf_status}
            </div>
            <small style="color:#94a3b8;">{chr(10).join([f"• {r}" for r in reasons[:3]])}</small>
        </div>
        """, unsafe_allow_html=True)

    col_idx += 1

st.caption("Educational tool only – not financial advice")
