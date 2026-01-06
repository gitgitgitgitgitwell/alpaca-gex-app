import streamlit as st
import pandas as pd

from dealer_flow_alpaca import run_scan  # <-- change filename if needed

st.set_page_config(page_title="Alpaca GEX Scanner", layout="wide")
st.title("Alpaca GEX Scanner")

# Defaults (match your config)
DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "GOOGL", "RMBS", "VRT", "MRVL", "MU", "CRDO", "APH", "ALAB",
                   "ANET", "PRIM", "LRCX", "MOD", "DOV", "AVGO", "SEI", "ABBNY", "VICR", "COHR",
                   "PRYMY", "SNDK", "AMZN", "BE", "WMB", "AR", "NVCR", "INTC", "RIVN", "POET", "DBRG"]

with st.sidebar:
    st.header("Config")

    tickers_text = st.text_area(
        "Tickers (comma-separated)",
        value=",".join(DEFAULT_TICKERS),
        height=140
    )

    # Optional: expose parameters later if you wire them into run_scan
    # max_days = st.number_input("MAX_DAYS_TO_EXPIRY", min_value=1, max_value=365, value=30)
    # use_real_oi = st.checkbox("USE_REAL_OPEN_INTEREST", value=True)

    run_btn = st.button("Run scan", type="primary")

tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

if run_btn:
    with st.spinner(f"Running scan for {len(tickers)} tickers..."):
        details, summary, narrative = run_scan(tickers)

    # If your run_scan returns DataFrames (your earlier type hint showed 3 DataFrames),
    # display them accordingly. If it returns dicts/strings, adjust.
    st.subheader("Details")
    if isinstance(details, pd.DataFrame):
        st.dataframe(details, use_container_width=True)
    else:
        st.json(details)

    st.subheader("Summary")
    if isinstance(summary, pd.DataFrame):
        st.dataframe(summary, use_container_width=True)
    else:
        st.json(summary)

    st.subheader("Narrative")
    if isinstance(narrative, pd.DataFrame):
        st.dataframe(narrative, use_container_width=True)
    else:
        st.write(narrative)
