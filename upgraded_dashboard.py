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
ALL_PAIRS = { ... same as before ... }  # (keep your existing ALL_PAIRS dict unchanged)

# ========================= CURRENCY STRENGTH ENGINE (THE BIG FIX) =========================
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
        # Pair-specific news from Twelve Data (best source)
        main_pair = f"{currency}USD" if currency != "USD" else "EURUSD"
        if currency in ["XAU", "XAG"]: main_pair = f"{currency}USD"
        twelve_news = get_twelve_news(main_pair)
        
        # Global RSS
        rss = get_rss_headlines()
        all_news = twelve_news + rss
        filtered = [h for h in all_news if any(k.lower() in h.lower() for k in keywords)]
        
        if not filtered:
            return 50
        
        score = finbert(" | ".join(filtered[:10]))
        # Stronger swing + event bias
        score = int(score * 1.12)
        
        # Add Forex Factory high-impact events bias
        events = get_high_impact_events()
        event_bonus = sum(8 if currency in ev[0] else -8 for ev in events[:5])
        score += event_bonus
        
        return max(15, min(85, score))
    except:
        return 50

@st.cache_data(ttl=300)
def get_all_currency_strengths():
    return {cur: get_currency_strength(cur) for cur in CURRENCY_KEYWORDS}

# (Keep your existing get_twelve_news, get_rss_headlines, finbert, get_high_impact_events, get_price, get_time_series_tf, calculate_technical_score exactly as in my previous version)

# ========================= UPDATED ANALYSIS =========================
def analyze_pair(pair_name: str, symbol: str, keywords: list):
    price = get_price(symbol)
    df_1h = get_time_series_tf(symbol, "1h", 70)
    
    tech_score, tech_reasons = calculate_technical_score(df_1h)
    
    # === CURRENCY STRENGTH LOGIC ===
    strengths = get_all_currency_strengths()
    
    # Parse base & quote
    if pair_name.startswith(("XAU", "XAG")):
        base = pair_name[:3]
        quote = "USD"
    else:
        base = pair_name[:3]
        quote = pair_name[3:]
    
    base_strength = strengths.get(base, 50)
    quote_strength = strengths.get(quote, 50)
    
    relative_score = int(base_strength - quote_strength) + 50   # 0-100 range
    
    # Final confidence
    final_score = int(0.45 * tech_score + 0.55 * relative_score)
    final_score = max(15, min(88, final_score))
    
    direction = "Bullish" if final_score > 50 else "Bearish"
    confidence = final_score if direction == "Bullish" else (100 - final_score)
    
    # Risk & multi-timeframe (same as before)
    w = get_tf_trend(symbol, "1week")
    d = get_tf_trend(symbol, "1day")
    h4 = get_tf_trend(symbol, "4h")
    aligned = sum(1 for t in [w, d, h4] if t == direction)
    
    if aligned == 3:
        risk_percent = round(2.8 + confidence / 40, 1)
        risk_color = "🟢"
        tf_status = "ALL TFs aligned 🔥"
    elif aligned == 2:
        risk_percent = round(1.6 + confidence / 55, 1)
        risk_color = "🟡"
        tf_status = "Strong alignment"
    else:
        risk_percent = 0.8
        risk_color = "🔴"
        tf_status = "Mixed TFs"
    
    reasons = tech_reasons + [f"{base} strength: {base_strength}% | {quote} strength: {quote_strength}%"]
    
    return price, final_score, direction, confidence, risk_percent, risk_color, tf_status, reasons, df_1h, w, d, h4

# ========================= UI (same as before, just update the display) =========================
# ... (keep the entire UI section exactly the same as my previous version)

st.subheader(f"**{pair}**")
if price:
    st.metric("Live Price", f"{price:.5f}")

color = "#22c55e" if direction == "Bullish" else "#ef4444"
st.markdown(f"**{direction}** <span style='color:{color};font-size:1.9em'>**{confidence}%**</span>", unsafe_allow_html=True)
st.progress(confidence / 100)

st.markdown(f"**{risk_color} Recommended Risk:** **{risk_percent}%**")
st.caption(tf_status)
