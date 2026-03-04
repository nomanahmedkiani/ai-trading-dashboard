import streamlit as st
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

st.set_page_config(page_title="AI Forex Intelligence Terminal", layout="wide", page_icon="📈")

# ========================= SECRETS =========================
TWELVE_KEY = st.secrets["TWELVE_DATA_API_KEY"]
HF_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# ========================= 32 PAIRS =========================
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
    "EURCHF": ["EUR/CHF", ["EUR", "CHF", "ECB", "SNB"]],
    "EURAUD": ["EUR/AUD", ["EUR", "AUD", "ECB", "RBA"]],
    "GBPAUD": ["GBP/AUD", ["GBP", "AUD", "BOE", "RBA"]],
    "AUDJPY": ["AUD/JPY", ["AUD", "JPY", "RBA", "BOJ"]],
    "USDCNH": ["USD/CNH", ["USD", "CNH", "PBOC", "Fed"]],
    "XAUUSD": ["XAU/USD", ["Gold", "USD", "Fed", "Treasury"]],
    "XAGUSD": ["XAG/USD", ["Silver", "USD", "Fed"]],
    "EURCAD": ["EUR/CAD", ["EUR", "CAD", "ECB", "BOC"]],
    "GBPCAD": ["GBP/CAD", ["GBP", "CAD", "BOE", "BOC"]],
    "AUDCAD": ["AUD/CAD", ["AUD", "CAD", "RBA", "BOC"]],
    "USDTRY": ["USD/TRY", ["USD", "TRY", "Turkey", "Lira"]],
    "USDMXN": ["USD/MXN", ["USD", "MXN", "Mexico", "Peso"]],
    "USDZAR": ["USD/ZAR", ["USD", "ZAR", "South Africa", "Rand"]],
    "USDBRL": ["USD/BRL", ["USD", "BRL", "Brazil", "Real"]],
    "EURTRY": ["EUR/TRY", ["EUR", "TRY", "ECB", "Turkey"]],
    "GBPTRY": ["GBP/TRY", ["GBP", "TRY", "BOE", "Turkey"]],
    "NZDJPY": ["NZD/JPY", ["NZD", "JPY", "RBNZ", "BOJ"]],
    "CADJPY": ["CAD/JPY", ["CAD", "JPY", "BOC", "BOJ"]],
    "AUDNZD": ["AUD/NZD", ["AUD", "NZD", "RBA", "RBNZ"]],
    "CHFJPY": ["CHF/JPY", ["CHF", "JPY", "SNB", "BOJ"]],
    "GBPNZD": ["GBP/NZD", ["GBP", "NZD", "BOE", "RBNZ"]],
    "EURNZD": ["EUR/NZD", ["EUR", "NZD", "ECB", "RBNZ"]],
}

# ========================= PURE AI NEWS + FUNDAMENTALS ENGINE =========================
@st.cache_data(ttl=180)
def get_pair_news(symbol: str):
    try:
        r = requests.get(f"https://api.twelvedata.com/news?symbol={symbol}&limit=15&language=en&apikey={TWELVE_KEY}", timeout=12).json()
        news = []
        for item in r.get("data", [])[:12]:
            title = item.get("title", "")
            desc = item.get("description", "")[:300]
            if title:
                news.append(title + ". " + desc)
        return news
    except:
        return []

@st.cache_data(ttl=240)
def get_rss_headlines():
    feeds = [
        "https://www.investing.com/rss/news_1.rss",
        "https://www.fxstreet.com/rss/news",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.feedburner.com/forexlive"
    ]
    headlines = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:20]:
                if title := item.find("title"):
                    headlines.append(title.text.strip())
        except:
            continue
    return headlines

def finbert(text: str) -> int:
    try:
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        result = requests.post(API_URL, headers=headers, json={"inputs": text[:1400]}, timeout=18).json()
        if isinstance(result, list) and result:
            scores = result[0]
            pos = next((s["score"] for s in scores if s["label"] == "positive"), 0)
            neg = next((s["score"] for s in scores if s["label"] == "negative"), 0)
            return int(50 + (pos - neg) * 105)   # VERY strong swing for clear Bullish/Bearish
        return 50
    except:
        return 50

@st.cache_data(ttl=300)
def get_dxy_bias():
    try:
        r = requests.get(f"https://api.twelvedata.com/price?symbol=DXY&apikey={TWELVE_KEY}", timeout=8).json()
        price = float(r.get("price", 0))
        return 25 if price > 103 else -25 if price < 99 else 0
    except:
        return 0

@st.cache_data(ttl=300)
def get_high_impact_events():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=10)
        root = ET.fromstring(r.content)
        events = []
        for ev in root.findall(".//event"):
            if ev.find("impact") is not None and "high" in (ev.find("impact").text or "").lower():
                cur = ev.find("currency").text or ""
                title = ev.find("title").text or ""
                events.append((cur, title))
        return events
    except:
        return []

# ========================= PURE AI ANALYSIS =========================
def analyze_pair(pair_name: str, symbol: str, keywords: list):
    price = get_price(symbol)
    news = get_pair_news(symbol) + get_rss_headlines()
    filtered = [h for h in news if any(k.lower() in h.lower() for k in keywords)]
    
    if not filtered:
        score = 50
        headlines = ["No major news today"]
    else:
        score = finbert(" | ".join(filtered[:12]))
        if len(filtered) >= 8:
            score = int(score * 1.15)   # extra boost when lots of news
    
    # Add high-impact events & USD bias
    events = get_high_impact_events()
    event_bonus = sum(35 if any(k in ev[0] for k in keywords) else -35 for ev in events[:8])
    score += event_bonus
    dxy = get_dxy_bias()
    if "USD" in pair_name or pair_name in ["XAUUSD", "XAGUSD"]:
        score += dxy
    
    final_score = max(12, min(88, score))
    direction = "Bullish" if final_score > 52 else "Bearish" if final_score < 48 else "Neutral"
    confidence = final_score if direction == "Bullish" else (100 - final_score)
    
    # AI Insight + Reasons
    top_headlines = filtered[:5]
    ai_insight = "Strong bullish momentum from positive central bank outlook and risk appetite." if final_score > 65 else \
                "Bearish pressure due to strong USD and risk-off sentiment." if final_score < 35 else \
                "Mixed sentiment with upcoming data releases likely to decide direction."
    
    reasons = [ai_insight] + top_headlines
    
    return price, final_score, direction, confidence, reasons

@st.cache_data(ttl=90)
def get_price(symbol: str):
    try:
        r = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}", timeout=10).json()
        return float(r.get("price", 0)) if "price" in r else None
    except:
        return None

# ========================= UI =========================
st.title("🚀 AI Forex Intelligence Terminal – Pure AI News & Fundamentals")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC • 100% AI + News + Events")

if st.button("🔄 Refresh All Data (Pure AI Analysis)", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.header("🎛️ Manage Dashboard")
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(ALL_PAIRS.keys())

selected = st.sidebar.multiselect("Pairs on Dashboard", options=list(ALL_PAIRS.keys()), default=st.session_state.watchlist)
if selected != st.session_state.watchlist:
    st.session_state.watchlist = selected
    st.rerun()

compact_mode = st.sidebar.toggle("Compact Mode", value=len(st.session_state.watchlist) > 10)

cols = st.columns(2) if compact_mode else st.columns(3)
col_idx = 0

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, score, direction, confidence, reasons = analyze_pair(pair, symbol, keywords)
    
    with cols[col_idx % len(cols)]:
        st.markdown("<div style='background:#1f2937;padding:20px;border-radius:14px;margin:8px;'>", unsafe_allow_html=True)
        
        st.subheader(f"**{pair}**")
        if price:
            st.metric("Live Price", f"{price:.5f}")
        
        color = "#22c55e" if direction == "Bullish" else "#ef4444"
        st.markdown(f"**{direction}** <span style='color:{color};font-size:2em'>**{confidence}%**</span>", unsafe_allow_html=True)
        st.progress(confidence / 100)
        
        with st.expander("📋 AI Analysis & Reasons", expanded=True):
            for r in reasons:
                st.write(f"• {r}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    col_idx += 1

st.caption("⚠️ Educational tool only • Powered by real-time news & AI sentiment • Not financial advice")
