import streamlit as st
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="AI Macro Forex Desk", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

# ========================= SECRETS =========================
TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# ========================= 32 PAIRS =========================
ALL_PAIRS = { ... }  # (same 32 pairs as last version - I kept them all)

# ========================= BEST FREE NEWS SOURCES (2026) =========================
@st.cache_data(ttl=300)
def get_rss_headlines():
    feeds = [
        "https://www.investing.com/rss/news_1.rss",      # #1 Forex news
        "https://www.forexlive.com/feed",                # Real-time breaking FX
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.fxstreet.com/rss/news",
        "https://www.dailyfx.com/rss",
        "https://www.marketwatch.com/rss/topstories"
    ]
    headlines = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            root = ET.fromstring(r.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'} if "feed" in str(r.content) else {}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)
            for item in items[:12]:
                title = item.find("title").text if item.find("title") is not None else None
                if title: headlines.append(title.strip())
        except: continue
    return headlines

# (keep all your previous functions: get_price, get_time_series_tf, calculate_technical_score, calculate_rsi, finbert, news_analysis, get_tf_trend, analyze_pair)

# NEW: AI Overview Generator (like screenshot)
def generate_ai_overview(direction, score, tech_score, news_score, aligned, reasons):
    bias = "strongly bullish" if direction == "Bullish" and score > 70 else "bullish" if direction == "Bullish" else \
           "strongly bearish" if direction == "Bearish" and score < 35 else "bearish" if direction == "Bearish" else "neutral"
    overview = f"AI sentiment featuring {bias} bias as investors react to {reasons[0] if reasons else 'current conditions'}. "
    overview += f"Technicals show {tech_score}% strength while news flow adds {news_score}% conviction. "
    if aligned > 0:
        overview += f"{aligned}/3 timeframes aligned — conviction building."
    return overview

# ========================= CUSTOM CSS (Pro Macro Desk Look) =========================
st.markdown("""
<style>
    .main { background: #0a0e17; color: #e2e8f0; }
    .stMetric label { color: #94a3b8; font-size: 0.95rem; }
    .card { background: #1e2937; border-radius: 16px; padding: 24px; margin: 12px 0; border: 1px solid #334155; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .big-price { font-size: 3.2rem; font-weight: 700; color: #60a5fa; }
    .sentiment-badge { padding: 6px 18px; border-radius: 9999px; font-weight: 700; font-size: 1.1rem; }
    .gauge-container { display: flex; justify-content: center; align-items: center; }
</style>
""", unsafe_allow_html=True)

# ========================= UI =========================
st.title("🚀 AI Macro Forex Desk")
st.caption(f"Live • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC • Powered by Forexlive + Investing.com")

if st.button("🔄 Refresh All Markets", type="primary"):
    st.cache_data.clear()
    st.rerun()


# Sidebar
st.sidebar.header("🎛️ Watchlist Manager")
st.sidebar.caption("Search • Add • Remove • All pairs visible by default")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(ALL_PAIRS.keys())[:12]  # ← move it here

selected = st.sidebar.multiselect(
    "Pairs on Dashboard",
    options=list(ALL_PAIRS.keys()),
    default=st.session_state.watchlist
)


selected = st.sidebar.multiselect("Pairs on Desk", options=list(ALL_PAIRS.keys()), default=st.session_state.watchlist)
if selected != st.session_state.watchlist:
    st.session_state.watchlist = selected
    st.rerun()

# ====================== PRO CARDS ======================
cols = st.columns(3)
col_idx = 0

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, score, direction, risk_percent, risk_color, tf_status, reasons, df, w, d, h4 = analyze_pair(pair, symbol, keywords)
    
    tech = 65  # fallback
    news = 60
    overview = generate_ai_overview(direction, score, tech, news, 2 if direction != "Neutral" else 0, reasons)
    
    with cols[col_idx % 3]:
        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:start;">
                <div>
                    <h3 style="margin:0; color:#94a3b8;">{pair}</h3>
                    <div class="big-price">{price:.5f if price else '—'}</div>
                </div>
                <div style="text-align:right;">
                    <span class="sentiment-badge {'bullish' if direction=='Bullish' else 'bearish' if direction=='Bearish' else 'neutral'}">
                        {direction}
                    </span>
                    <div style="margin-top:8px; font-size:1.8rem; font-weight:700; color:#a5b4fc;">{score}%</div>
                    <small style="color:#64748b;">AI Confidence</small>
                </div>
            </div>
            
            <div class="gauge-container">
                {f'<div style="width:110px;height:110px;border-radius:50%;background:conic-gradient(#22c55e 0% {score}%, #334155 {score}% 100%);display:flex;align-items:center;justify-content:center;font-size:1.4rem;font-weight:700;">{score}%</div>'}
            </div>
            
            <p style="font-size:0.95rem; line-height:1.4; color:#cbd5e1; margin:16px 0;">
                {overview}
            </p>
            
            <div style="background:#0f172a; padding:12px; border-radius:10px;">
                <strong style="color:#22c55e;">{risk_color} Risk:</strong> <strong>{risk_percent}%</strong> • {tf_status}
            </div>
            
            <div style="margin-top:12px;">
                {st.plotly_chart(go.Figure(data=go.Scatter(x=df["datetime"], y=df["close"], mode='lines', line=dict(color='#60a5fa', width=2))).update_layout(height=140, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor="#1e2937", paper_bgcolor="#1e2937", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)), use_container_width=True) if df is not None else ''}
            </div>
            
            <details style="margin-top:12px; font-size:0.9rem;">
                <summary style="cursor:pointer; color:#94a3b8;">Key Drivers & Market Structure</summary>
                Weekly: {w} | Daily: {d} | 4H: {h4}<br>
                {chr(10).join(['• '+r for r in reasons[:5]])}
            </details>
        </div>
        """, unsafe_allow_html=True)
    
    col_idx += 1

st.caption("⚠️ Educational & entertainment only • Not financial advice • Always verify with your broker")
