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

# ========================= REAL-TIME AI ENGINE =========================
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
            return int(50 + (pos - neg) * 105)
        return 50
    except:
        return 50

@st.cache_data(ttl=300)
def get_high_impact_events():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=10)
        root = ET.fromstring(r.content)
        events = []
        for ev in root.findall(".//event"):
            if ev.find("impact") is not None and "high" in (ev.find("impact").text or "").lower():
                events.append((ev.find("currency").text or "", ev.find("title").text or ""))
        return events
    except:
        return []

@st.cache_data(ttl=300)
def get_dxy_bias():
    try:
        r = requests.get(f"https://api.twelvedata.com/price?symbol=DXY&apikey={TWELVE_KEY}", timeout=8).json()
        price = float(r.get("price", 0))
        return "RISK-OFF" if price > 102.5 else "RISK-ON"
    except:
        return "RISK-OFF"

# ========================= FIXED MARKET STRUCTURE (NOW SHOWS REAL BULLISH/BEARISH) =========================
@st.cache_data(ttl=180)
def get_time_series_tf(symbol: str, interval: str):
    try:
        r = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={TWELVE_KEY}",
            timeout=15
        ).json()
        if "values" not in r or len(r["values"]) < 40:
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
    if df is None or len(df) < 40:
        return "Neutral"
    
    closes = df["close"]
    # Use rolling high/low + SMA for clear structure
    recent = closes.tail(80)
    rolling_high = recent.rolling(5).max().iloc[-1]
    rolling_low = recent.rolling(5).min().iloc[-1]
    last_close = recent.iloc[-1]
    sma20 = recent.rolling(20).mean().iloc[-1]
    sma50 = recent.rolling(50).mean().iloc[-1]
    
    if last_close > rolling_high:
        return "Bullish Structure (Break of High)"
    elif last_close < rolling_low:
        return "Bearish Structure (Break of Low)"
    elif sma20 > sma50:
        return "Bullish Structure (Trend Holding)"
    elif sma20 < sma50:
        return "Bearish Structure (Trend Holding)"
    else:
        return "Sideways / Compression"

# ========================= HELPER FUNCTIONS =========================
def generate_ai_overview(score, pair, mood):
    if score >= 68:
        return f"Strong bullish conviction on {pair}. Risk appetite returning — central bank dovishness, positive economic surprises and safe-haven unwinding driving flows into the base currency."
    elif score <= 35:
        return f"Clear bearish dominance on {pair}. {mood} sentiment prevails — geopolitical tensions, strong safe-haven demand and risk aversion punishing risk-sensitive currencies."
    else:
        return f"Mixed but cautious bias on {pair}. Traders remain on the sidelines ahead of key data releases and central bank commentary."

def generate_positioning(score):
    if score >= 68:
        return "Risk-on dominance. Funds rotating into higher-yielding / growth currencies. Traders punishing underperforming shorts lacking immediate returns."
    elif score <= 35:
        return "Risk-off dominance. Use of funds under the microscope — traders punishing longs that do not offer immediate cash returns or defensive characteristics."
    else:
        return "Balanced / neutral positioning. Investors cautious — seeking safer assets until clearer directional catalyst emerges."

def get_flow_status(score):
    if score >= 68: return "HEALTHY FLOW"
    elif score <= 35: return "CHOPPY DOWN TREND"
    else: return "NORMAL PARTICIPATION"

def get_technical_bias(score):
    if score >= 68: return "Buy"
    elif score <= 35: return "Sell"
    else: return "Hold"

@st.cache_data(ttl=90)
def get_price(symbol: str):
    try:
        r = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}", timeout=10).json()
        return float(r.get("price", 0)) if "price" in r else None
    except:
        return None

# ========================= ANALYSIS FUNCTION =========================
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
   
    weekly = get_tf_structure(symbol, "1week")
    daily = get_tf_structure(symbol, "1day")
    h4 = get_tf_structure(symbol, "4h")
   
    reasons = filtered[:6] if filtered else ["Limited real-time news flow – awaiting next major catalyst"]
   
    return price, direction, confidence, mood, ai_overview, positioning, flow_status, tech_bias, reasons, weekly, daily, h4

# ========================= UI =========================
current_mood = get_dxy_bias()

st.title("🚀 AI Forex Intelligence Terminal")
st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC • "
    f"Real-time AI Bias • Current Market Mood: {current_mood}"
)

if st.button("🔄 Refresh All Data (AI Re-Analysis)", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.header("🎛️ Manage Dashboard")
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(ALL_PAIRS.keys())

selected = st.sidebar.multiselect(
    "Pairs on Dashboard",
    options=list(ALL_PAIRS.keys()),
    default=st.session_state.watchlist
)
if selected != st.session_state.watchlist:
    st.session_state.watchlist = selected
    st.rerun()

compact_mode = st.sidebar.toggle("Compact Mode", value=len(st.session_state.watchlist) > 8)

cols = st.columns(2) if compact_mode else st.columns(3)
col_idx = 0

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, direction, confidence, mood, ai_overview, positioning, flow_status, tech_bias, reasons, weekly, daily, h4 = analyze_pair(pair, symbol, keywords)
   
    with cols[col_idx % len(cols)]:
        st.markdown("<div style='background:#1f2937;padding:22px;border-radius:16px;margin:8px;box-shadow:0 4px 15px rgba(0,0,0,0.3);'>", unsafe_allow_html=True)
       
        st.subheader(f"**{pair}**")
        if price:
            st.metric("Live Price", f"{price:.5f}")
       
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
       
        with st.expander("📈 Market Structure", expanded=False):
            st.write(f"**Weekly**: **{weekly}**")
            st.write(f"**Daily**: **{daily}**")
            st.write(f"**4H**: **{h4}**")
       
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
