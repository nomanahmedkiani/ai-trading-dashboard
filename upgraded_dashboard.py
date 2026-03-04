import streamlit as st
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np

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

# ========================= CURRENCY STRENGTH ENGINE =========================
CURRENCY_KEYWORDS = {
    "EUR": ["EUR", "ECB", "Eurozone", "Lagarde", "Frankfurt"],
    "GBP": ["GBP", "BOE", "Pound", "UK", "Bailey"],
    "USD": ["USD", "Fed", "Dollar", "Powell", "FOMC", "Treasury"],
    "JPY": ["JPY", "BOJ", "Yen", "Ueda"],
    "CAD": ["CAD", "BOC", "Loonie", "Canada"],
    "AUD": ["AUD", "RBA", "Aussie"],
    "NZD": ["NZD", "RBNZ", "Kiwi"],
    "CHF": ["CHF", "SNB", "Franc"],
    "CNH": ["CNH", "PBOC", "Yuan", "China"],
    "XAU": ["Gold", "XAU", "bullion", "inflation", "rate cut"],
    "XAG": ["Silver", "XAG"],
    "TRY": ["TRY", "Turkey", "Lira", "Erdogan"],
    "MXN": ["MXN", "Mexico", "Peso", "Banxico"],
    "ZAR": ["ZAR", "Rand", "South Africa"],
    "BRL": ["BRL", "Brazil", "Real"],
}

@st.cache_data(ttl=240)
def get_currency_strength(currency: str):
    keywords = CURRENCY_KEYWORDS.get(currency, [currency])
    try:
        main_pair = f"{currency}USD" if currency != "USD" else "EURUSD"
        if currency in ["XAU", "XAG"]:
            main_pair = f"{currency}USD"
        twelve_news = get_twelve_news(main_pair)
        rss = get_rss_headlines()
        all_news = twelve_news + rss
        filtered = [h for h in all_news if any(k.lower() in h.lower() for k in keywords)]
        if not filtered:
            return 50
        combined = " | ".join(filtered[:12])  # more headlines
        score = finbert(combined)
        # Amplify swing + volume bonus
        score = int(score * 1.18)  # stronger reaction
        if len(filtered) >= 6:
            score += 12 if score > 50 else -12
        # High-impact events ×3 multiplier
        events = get_high_impact_events()
        event_bonus = sum(24 if currency in ev[0] else -24 for ev in events[:6])  # ×3 boost
        score += event_bonus
        return max(10, min(90, score))
    except:
        return 50

@st.cache_data(ttl=300)
def get_all_currency_strengths():
    return {cur: get_currency_strength(cur) for cur in CURRENCY_KEYWORDS}

@st.cache_data(ttl=180)
def get_twelve_news(symbol: str):
    try:
        url = f"https://api.twelvedata.com/news?symbol={symbol}&limit=12&language=en&apikey={TWELVE_KEY}"
        r = requests.get(url, timeout=12).json()
        news_list = []
        items = r.get("data", []) or r.get("news", []) or r.get("values", [])
        for item in items[:12]:
            title = item.get("title", "")
            desc = item.get("description", "")[:250]
            if title:
                news_list.append(title + ". " + desc)
        return news_list
    except:
        return []

@st.cache_data(ttl=240)
def get_rss_headlines():
    feeds = [
        "https://www.investing.com/rss/news_1.rss",
        "https://www.fxstreet.com/rss/news",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.marketwatch.com/rss/topstories",
        "https://feeds.feedburner.com/forexlive"
    ]
    headlines = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:20]:  # more items
                if title := item.find("title"):
                    headlines.append(title.text.strip())
        except:
            continue
    return headlines

def finbert(text: str) -> int:
    try:
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        result = requests.post(API_URL, headers=headers, json={"inputs": text[:1200]}, timeout=18).json()
        if isinstance(result, list) and result:
            scores = result[0]
            pos = next((s["score"] for s in scores if s["label"] == "positive"), 0)
            neg = next((s["score"] for s in scores if s["label"] == "negative"), 0)
            raw_diff = pos - neg
            return int(50 + raw_diff * 90)  # MUCH stronger swing → real variation
        return 50
    except:
        return 50

@st.cache_data(ttl=300)
def get_high_impact_events():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=10)
        root = ET.fromstring(r.content)
        events = []
        for event in root.findall(".//event"):
            impact = event.find("impact")
            if impact is not None and impact.text and "high" in impact.text.lower():
                currency = event.find("currency").text or ""
                title = event.find("title").text or ""
                events.append((currency, title))
        return events
    except:
        return []

@st.cache_data(ttl=90)
def get_price(symbol: str):
    try:
        r = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_KEY}", timeout=10).json()
        return float(r.get("price", 0)) if "price" in r else None
    except:
        return None

@st.cache_data(ttl=180)
def get_time_series_tf(symbol: str, interval: str, size: int = 70):
    try:
        r = requests.get(f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={size}&apikey={TWELVE_KEY}", timeout=10).json()
        if "values" not in r:
            return None
        df = pd.DataFrame(r["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        return df
    except:
        return None

def calculate_technical_score(df):
    if df is None or len(df) < 25:
        return 50, ["⚠️ Limited data"]
    closes = df["close"]
    reasons = []
    score = 50
    ema12 = closes.ewm(span=12).mean().iloc[-1]
    ema26 = closes.ewm(span=26).mean().iloc[-1]
    if ema12 > ema26:
        score += 26
        reasons.append("EMA Bullish")
    else:
        score -= 26
        reasons.append("EMA Bearish")
    ema12_full = closes.ewm(span=12).mean()
    ema26_full = closes.ewm(span=26).mean()
    macd = ema12_full - ema26_full
    signal = macd.ewm(span=9).mean()
    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] < signal.iloc[-2]:
        score += 22
        reasons.append("MACD Bullish Crossover")
    elif macd.iloc[-1] < signal.iloc[-1]:
        score -= 18
        reasons.append("MACD Bearish")
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.isna().iloc[-1] else 50.0
    if rsi < 30:
        score += 20
        reasons.append(f"RSI Oversold ({rsi:.0f})")
    elif rsi > 70:
        score -= 20
        reasons.append(f"RSI Overbought ({rsi:.0f})")
    return max(10, min(90, int(score))), reasons

def get_tf_trend(symbol: str, interval: str) -> str:
    df = get_time_series_tf(symbol, interval, 40)
    if df is None or len(df) < 15:
        return "Neutral"
    ema = df["close"].ewm(span=34).mean().iloc[-1]
    return "Bullish" if df["close"].iloc[-1] > ema else "Bearish"

# ========================= MAIN ANALYSIS =========================
def analyze_pair(pair_name: str, symbol: str, keywords: list):
    price = get_price(symbol)
    df_1h = get_time_series_tf(symbol, "1h", 80)
    tech_score, tech_reasons = calculate_technical_score(df_1h)
    strengths = get_all_currency_strengths()
    if pair_name.startswith(("XAU", "XAG")):
        base = pair_name[:3]
        quote = "USD"
    else:
        base = pair_name[:3]
        quote = pair_name[3:]
    base_strength = strengths.get(base, 50)
    quote_strength = strengths.get(quote, 50)
    diff = base_strength - quote_strength
    relative_score = int(50 + diff * 1.4)  # stronger separation
    # Higher technical weight for trending pairs
    final_score = int(0.52 * tech_score + 0.48 * relative_score)
    final_score = max(10, min(90, final_score))
    direction = "Bullish" if final_score > 50 else "Bearish"
    confidence = final_score if direction == "Bullish" else (100 - final_score)
    w = get_tf_trend(symbol, "1week")
    d = get_tf_trend(symbol, "1day")
    h4 = get_tf_trend(symbol, "4h")
    aligned = sum(1 for t in [w, d, h4] if t == direction)
    if aligned == 3:
        risk_percent = round(3.0 + confidence / 35, 1)
        risk_color = "🟢"
        tf_status = "ALL TFs aligned 🔥"
    elif aligned == 2:
        risk_percent = round(1.8 + confidence / 50, 1)
        risk_color = "🟡"
        tf_status = "Strong alignment"
    else:
        risk_percent = 0.9
        risk_color = "🔴"
        tf_status = "Mixed TFs"
    reasons = tech_reasons + [f"{base} strength: {base_strength}% | {quote} strength: {quote_strength}%"]
    return price, final_score, direction, confidence, risk_percent, risk_color, tf_status, reasons, df_1h, w, d, h4

# ========================= UI =========================
st.title("🚀 AI Forex Intelligence Terminal – Enhanced Direction")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC • Stronger AI sentiment & fundamentals")

if st.button("🔄 Refresh All Data (may take 30–90s)", type="primary"):
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

compact_mode = st.sidebar.toggle("Compact Mode", value=len(st.session_state.watchlist) > 10)

cols = st.columns(2) if compact_mode else st.columns(3)
col_idx = 0

for pair in st.session_state.watchlist:
    symbol, keywords = ALL_PAIRS[pair]
    price, final_score, direction, confidence, risk_percent, risk_color, tf_status, reasons, df, w, d, h4 = analyze_pair(pair, symbol, keywords)
    
    with cols[col_idx % len(cols)]:
        st.markdown(
            "<div style='background:#1f2937;padding:20px;border-radius:14px;margin:8px;'>",
            unsafe_allow_html=True
        )
        
        st.subheader(f"**{pair}**")
        if price:
            st.metric("Live Price", f"{price:.5f}")
        
        color = "#22c55e" if direction == "Bullish" else "#ef4444"
        st.markdown(
            f"**{direction}** <span style='color:{color};font-size:1.9em'>**{confidence}%**</span>",
            unsafe_allow_html=True
        )
        st.progress(confidence / 100)
        
        st.markdown(f"**{risk_color} Recommended Risk:** **{risk_percent}%**")
        st.caption(tf_status)
        
        if not compact_mode:
            with st.expander("Market Structure", expanded=False):
                st.write(f"Weekly: **{w}** | Daily: **{d}** | 4H: **{h4}**")
            with st.expander("Key Drivers (AI + News + Events)", expanded=False):
                for r in reasons[:10]:
                    st.write(f"• {r}")
            with st.expander("1H Chart", expanded=False):
                if df is not None:
                    st.line_chart(df.set_index("datetime")["close"])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    col_idx += 1

st.caption("⚠️ Educational tool only • Not financial advice • Results vary with real-time news & events • Refresh often")
