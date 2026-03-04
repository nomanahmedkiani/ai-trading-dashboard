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

# ========================= REAL-TIME AI ENGINE (unchanged) =========================
# ... [All the previous functions: get_pair_news, get_rss_headlines, finbert, get_high_impact_events, get_dxy_bias, generate_ai_overview, generate_positioning, get_flow_status, get_technical_bias, get_price] ...
# (I kept them exactly as in the last version you liked — no changes here)

# ========================= FIXED MARKET STRUCTURE =========================
@st.cache_data(ttl=180)
def get_time_series_tf(symbol: str, interval: str):
    try:
        r = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=80&apikey={TWELVE_KEY}",
            timeout=12
        ).json()
        if "values" not in r:
            return None
        df = pd.DataFrame(r["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        return df
    except:
        return None

def get_tf_structure(symbol: str, interval: str) -> str:
    df = get_time_series_tf(symbol, interval)
    if df is None or len(df) < 15:
        return "Neutral"
    closes = df["close"]
    ema34 = closes.ewm(span=34, adjust=False).mean().iloc[-1]
    last_close = closes.iloc[-1]
    if pd.isna(ema34):
        return "Neutral"
    return "Bullish" if last_close > ema34 else "Bearish"

# ========================= ANALYSIS (only added Market Structure) =========================
def analyze_pair(pair_name: str, symbol: str, keywords: list):
    price = get_price(symbol)
    news = get_pair_news(symbol) + get_rss_headlines()
    filtered = [h for h in news if any(k.lower() in h.lower() for k in keywords)]
    
    score = finbert(" | ".join(filtered[:12])) if filtered else 50
    events = get_high_impact_events()
    event_bonus = sum(35 if any(k in ev[0] for k in keywords) else -35 for ev in events[:8])
    score += event_bonus
    if "USD" in pair_name or pair_name.startswith(("XAU","XAG")):
        score += 18 if get_dxy_bias() == "RISK-OFF" else -18
    
    final_score = max(12, min(88, int(score * 1.12)))
    direction = "Bullish" if final_score > 52 else "Bearish"
    confidence = final_score if direction == "Bullish" else (100 - final_score)
    
    mood = get_dxy_bias()
    ai_overview = generate_ai_overview(final_score, pair_name, mood)
    positioning = generate_positioning(final_score)
    flow_status = get_flow_status(final_score)
    tech_bias = get_technical_bias(final_score)
    
    # FIXED Market Structure
    weekly = get_tf_structure(symbol, "1week")
    daily = get_tf_structure(symbol, "1day")
    h4 = get_tf_structure(symbol, "4h")
    
    reasons = filtered[:6] if filtered else ["Limited real-time news flow – awaiting next major catalyst"]
    
    return price, direction, confidence, mood, ai_overview, positioning, flow_status, tech_bias, reasons, weekly, daily, h4

# ========================= UI (unchanged except Market Structure display) =========================
st.title("🚀 AI Forex Intelligence Terminal")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC • Real-time AI Bias • Current Market Mood: {get_dxy_bias()}")

if st.button("🔄 Refresh All Data (AI Re-Analysis)", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Sidebar and loop remain exactly the same as last version
# ... [your existing sidebar code] ...

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, direction, confidence, mood, ai_overview, positioning, flow_status, tech_bias, reasons, weekly, daily, h4 = analyze_pair(pair, symbol, keywords)
    
    with cols[col_idx % len(cols)]:
        st.markdown("<div style='background:#1f2937;padding:22px;border-radius:16px;margin:8px;box-shadow:0 4px 15px rgba(0,0,0,0.3);'>", unsafe_allow_html=True)
        
        st.subheader(f"**{pair}**")
        if price:
            st.metric("Live Price", f"{price:.5f}")
        
        # Main status row (unchanged)
        c1, c2, c3 = st.columns(3)
        with c1:
            color = "#ef4444" if direction == "Bearish" else "#22c55e"
            st.markdown(f"<div style='text-align:center;'><strong>Bias</strong><br><span style='color:{color};font-size:26px;font-weight:bold'>{direction}</span></div>", unsafe_allow_html=True)
        with c2:
            bias_color = "#ef4444" if tech_bias == "Sell" else "#22c55e" if tech_bias == "Buy" else "#eab308"
            st.markdown(f"<div style='text-align:center;'><strong>Technical</strong><br><span style='color:{bias_color};font-size:26px;font-weight:bold'>{tech_bias}</span></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='text-align:center;'><strong>AI Confidence</strong><br><span style='color:#00ff9d;font-size:30px;font-weight:bold'>{confidence}%</span></div>", unsafe_allow_html=True)
        
        st.markdown(f"**Market Mood**: <span style='color:orange;font-weight:bold'>{mood}</span> • **Flow**: <span style='font-weight:bold'>{flow_status}</span>", unsafe_allow_html=True)
        
        # FIXED Market Structure
        with st.expander("📈 Market Structure", expanded=False):
            st.write(f"**Weekly**: **{weekly}**")
            st.write(f"**Daily**: **{daily}**")
            st.write(f"**4H**: **{h4}**")
        
        # Rest of expanders (AI Overview, Investor Positioning, Key Reasons) unchanged
        with st.expander("📊 AI Overview", expanded=True):
            st.write(ai_overview)
        with st.expander("Investor Positioning"):
            st.write(positioning)
        with st.expander("Key Reasons & Headlines"):
            for r in reasons:
                st.write(f"• {r}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    col_idx += 1

st.caption("⚠️ Educational tool only • Powered by real-time AI sentiment & fundamentals • Not financial advice")
