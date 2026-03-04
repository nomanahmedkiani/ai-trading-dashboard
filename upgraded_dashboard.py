import streamlit as st
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="AI Macro Forex Desk", layout="wide", page_icon="📈")

# Secrets
TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# Pairs (expanded list - add more if needed)
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
    "XAUUSD": ["XAU/USD", ["Gold", "USD", "Fed", "Treasury", "inflation"]],
    "XAGUSD": ["XAG/USD", ["Silver", "USD", "Fed"]],
    # Add more pairs here...
}

# ── Cached Data Fetchers ───────────────────────────────────────────────────
@st.cache_data(ttl=90)
def get_price(symbol):
    try:
        r = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}", timeout=10).json()
        return float(r.get("price")) if "price" in r else None
    except:
        return None

@st.cache_data(ttl=180)
def get_time_series_tf(symbol, interval, size=60):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={size}&apikey={TWELVE_KEY}"
        r = requests.get(url, timeout=10).json()
        if "values" not in r: return None
        df = pd.DataFrame(r["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        return df
    except:
        return None

# ── Technical Score ────────────────────────────────────────────────────────
def calculate_technical_score(df):
    if df is None or len(df) < 15:
        return 50, ["Limited technical data"]
    closes = df["close"]
    ema12 = closes.ewm(span=12).mean().iloc[-1]
    ema26 = closes.ewm(span=26).mean().iloc[-1]
    score = 50
    reasons = []
    if ema12 > ema26:
        score += 25
        reasons.append("EMA12 > EMA26 → Uptrend")
    else:
        score -= 25
        reasons.append("EMA12 < EMA26 → Downtrend")
    return max(15, min(85, int(score))), reasons

# ── News & Sentiment ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_rss_headlines():
    feeds = [
        "https://www.forexlive.com/feed",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.fxstreet.com/rss/news",
        "https://www.marketwatch.com/rss/topstories",
    ]
    headlines = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:12]:
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
        result = requests.post(url, headers=headers, json={"inputs": text}, timeout=15).json()
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
    all_headlines = get_rss_headlines()
    filtered = [h for h in all_headlines if any(k.lower() in h.lower() for k in keywords)]
    if not filtered:
        return 50, ["No matching news found"]
    combined = " | ".join(filtered[:6])
    score = finbert(combined)
    return score, filtered[:6]

# ── Risk & Multi-TF ────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def get_tf_trend(symbol, interval):
    df = get_time_series_tf(symbol, interval, 40)
    if df is None or len(df) < 20:
        return "Neutral"
    ema50 = df["close"].ewm(span=50).mean().iloc[-1]
    return "Bullish" if df["close"].iloc[-1] > ema50 else "Bearish"

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
        tf_status = "Neutral – stay conservative"
        w = d = h4 = "—"
    else:
        w = get_tf_trend(symbol, "1week")
        d = get_tf_trend(symbol, "1day")
        h4 = get_tf_trend(symbol, "4h")
        aligned = sum(1 for t in [w, d, h4] if t == direction)
        if aligned == 3:
            risk_percent, risk_color = 3.0, "🟢 High conviction"
        elif aligned == 2:
            risk_percent, risk_color = 1.5, "🟡 Moderate"
        else:
            risk_percent, risk_color = 0.5, "🔴 Low"
        tf_status = f"{aligned}/3 TFs aligned"

    reasons = tech_reasons + headlines
    return price, final_score, direction, risk_percent, risk_color, tf_status, reasons, df_1h, w, d, h4

# ── AI Overview Text ───────────────────────────────────────────────────────
def generate_overview(direction, score, reasons):
    bias = "bullish momentum" if direction == "Bullish" else "bearish pressure" if direction == "Bearish" else "balanced/neutral conditions"
    return f"Current AI sentiment leans **{bias}** ({score}% confidence). Key drivers include {reasons[0] if reasons else 'market flows'}. Watch for continuation or reversal based on higher timeframe alignment."

# ── High-Contrast Dark Theme CSS ───────────────────────────────────────────
st.markdown("""
<style>
    body, .stApp { background: #0f172a !important; color: #e2e8f0 !important; }
    h1, h2, h3, h4 { color: #f1f5f9 !important; }
    .stMetric label { color: #94a3b8 !important; font-size: 1rem; }
    p, div, span, small { color: #cbd5e1 !important; }
    .card { 
        background: #1e293b !important; 
        border: 1px solid #334155 !important; 
        border-radius: 12px; 
        padding: 20px; 
        margin: 12px 0; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.4); 
    }
    .big-price { font-size: 2.8rem; font-weight: 700; color: #60a5fa; }
    .bullish { background: rgba(34,197,94,0.25); color: #22c55e; padding: 6px 14px; border-radius: 999px; font-weight: 600; }
    .bearish { background: rgba(239,68,68,0.25); color: #ef4444; padding: 6px 14px; border-radius: 999px; font-weight: 600; }
    .neutral { background: rgba(234,179,8,0.25); color: #eab308; padding: 6px 14px; border-radius: 999px; font-weight: 600; }
    .gauge { width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#22c55e 0% var(--pct), #334155 var(--pct) 100%); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: bold; color: #fff; margin: 12px auto; }
</style>
""", unsafe_allow_html=True)

# ── Main UI ────────────────────────────────────────────────────────────────
st.title("🚀 AI Macro Forex Desk")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} • High-contrast dark mode")

if st.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Safe watchlist init
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(ALL_PAIRS.keys())[:10]  # start with 10 for better performance

st.sidebar.header("Watchlist Manager")
selected = st.sidebar.multiselect(
    "Select pairs to display",
    options=list(ALL_PAIRS.keys()),
    default=st.session_state.watchlist
)
if selected != st.session_state.watchlist:
    st.session_state.watchlist = selected
    st.rerun()

# Dashboard
cols = st.columns(3)
col_idx = 0

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, score, direction, risk_percent, risk_color, tf_status, reasons, df, w, d, h4 = analyze_pair(pair, symbol, keywords)

    pct_str = f"{score}%"
    dir_class = "bullish" if direction == "Bullish" else "bearish" if direction == "Bearish" else "neutral"
    overview = generate_overview(direction, score, reasons)

    with cols[col_idx % 3]:
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        
        st.subheader(pair)
        st.markdown(f'<div class="big-price">{price:.5f if price else "—"}</div>', unsafe_allow_html=True)
        
        st.markdown(f'<span class="{dir_class}">{direction}</span> <small>({pct_str} AI Confidence)</small>', unsafe_allow_html=True)
        
        # Gauge
        st.markdown(f'<div class="gauge" style="--pct: {score}%;">{pct_str}</div>', unsafe_allow_html=True)
        
        st.markdown(f"**{risk_color} Risk:** {risk_percent}%  •  {tf_status}")
        
        st.markdown(f"**AI Overview:** {overview}")
        
        if df is not None:
            fig = go.Figure(go.Scatter(x=df["datetime"], y=df["close"], mode='lines', line=dict(color='#60a5fa')))
            fig.update_layout(height=140, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis_showgrid=False, yaxis_showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Market Structure & Drivers"):
            st.write(f"Weekly: **{w}** | Daily: **{d}** | 4H: **{h4}**")
            for r in reasons[:6]:
                st.write(f"• {r}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    col_idx += 1

st.caption("⚠️ Educational tool only – not trading advice. Markets involve risk.")
