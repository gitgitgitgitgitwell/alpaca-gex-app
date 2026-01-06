"""import streamlit as st
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
"""

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from dealer_flow_alpaca import run_scan  # <-- change filename if needed

st.set_page_config(page_title="Alpaca GEX Scanner", layout="wide")
st.title("Alpaca GEX Scanner")

# Defaults (match your config)
DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "GOOGL", "RMBS", "VRT", "MRVL", "MU", "CRDO", "APH", "ALAB",
                   "ANET", "PRIM", "LRCX", "MOD", "DOV", "AVGO", "SEI", "ABBNY", "VICR", "COHR",
                   "PRYMY", "SNDK", "AMZN", "BE", "WMB", "AR", "NVCR", "INTC", "RIVN", "POET", "DBRG"]

# ----------------------------
# Helpers
# ----------------------------
def _to_dt_series(s: pd.Series) -> pd.Series:
    # Your CSV has timezone-aware strings like "2026-01-06 00:00:00+00:00"
    # Run output might be already datetime; this handles both.
    return pd.to_datetime(s, errors="coerce", utc=True)

def build_gamma_by_strike(details_df: pd.DataFrame, underlying: str, expiry: str | None, strike_bin: float) -> pd.DataFrame:
    """
    Expects details columns:
      underlying, type (call/put), strike, expiration, gex, spot
    Returns: df aggregated by strike_bin with call_gex, put_gex, net_gex
    """
    df = details_df.copy()
    df["underlying"] = df["underlying"].astype(str).str.upper()
    df = df[df["underlying"] == underlying.upper()].copy()
    if df.empty:
        return df

    df["expiration"] = _to_dt_series(df["expiration"])
    if expiry and expiry != "All":
        # expiry is a date string like YYYY-MM-DD
        exp_day = pd.to_datetime(expiry, utc=True).date()
        df = df[df["expiration"].dt.date == exp_day].copy()

    if df.empty:
        return df

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["gex"] = pd.to_numeric(df["gex"], errors="coerce").fillna(0.0)
    df["type"] = df["type"].astype(str).str.lower()

    # Bin strikes for cleaner mobile chart
    strike_bin = float(strike_bin) if strike_bin and strike_bin > 0 else 0.0
    if strike_bin > 0:
        df["strike_bin"] = (df["strike"] / strike_bin).round() * strike_bin
    else:
        df["strike_bin"] = df["strike"]

    # IMPORTANT:
    # Your gex values appear positive for both calls & puts (no sign baked in).
    # For a "net" view, a common convention is: net = call_gex - put_gex.
    df["call_gex"] = np.where(df["type"].eq("call"), df["gex"], 0.0)
    df["put_gex"] = np.where(df["type"].eq("put"), df["gex"], 0.0)

    agg = (
        df.groupby("strike_bin", as_index=False)
          .agg(
              call_gex=("call_gex", "sum"),
              put_gex=("put_gex", "sum"),
              spot=("spot", "max"),
              n=("gex", "size"),
          )
          .sort_values("strike_bin")
    )
    agg["net_gex"] = agg["call_gex"] - agg["put_gex"]
    agg["abs_net"] = agg["net_gex"].abs()
    return agg

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Config")

    tickers_text = st.text_area(
        "Tickers (comma-separated)",
        value=",".join(DEFAULT_TICKERS),
        height=140
    )

    run_btn = st.button("Run scan", type="primary")

tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

# ----------------------------
# Run + store results (so tabs don't lose state)
# ----------------------------
if run_btn:
    with st.spinner(f"Running scan for {len(tickers)} tickers..."):
        details, summary, narrative = run_scan(tickers)

    st.session_state["details"] = details
    st.session_state["summary"] = summary
    st.session_state["narrative"] = narrative

details = st.session_state.get("details")
summary = st.session_state.get("summary")
narrative = st.session_state.get("narrative")

# ----------------------------
# UI
# ----------------------------
if details is None:
    st.info("Click **Run scan** to generate results.")
else:
    tabs = st.tabs(["Charts", "Details", "Summary", "Narrative"])

    # ========== Charts ==========
    with tabs[0]:
        st.subheader("Dealer Gamma by Strike")

        if not isinstance(details, pd.DataFrame) or details.empty:
            st.warning("No details dataframe to chart.")
        else:
            # Controls
            underlyings = sorted(details["underlying"].astype(str).str.upper().unique())
            c1, c2, c3, c4 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")

            with c1:
                u = st.selectbox("Underlying", underlyings)

            # Expiries available for the selected underlying
            tmp = details.copy()
            tmp["expiration"] = _to_dt_series(tmp["expiration"])
            tmp["underlying"] = tmp["underlying"].astype(str).str.upper()
            expiries = (
                tmp[tmp["underlying"] == u]["expiration"]
                .dropna()
                .dt.date
                .astype(str)
                .unique()
                .tolist()
            )
            expiries = sorted(expiries)

            with c2:
                expiry = st.selectbox("Expiry", ["All"] + expiries)

            with c3:
                strike_bin = st.number_input("Strike bin", min_value=0.0, value=5.0, step=1.0)

            with c4:
                top_n = st.slider("Show top N strikes (by |net|)", 20, 200, 80, step=10)

            show_stacked = st.toggle("Stack calls vs puts", value=True)
            center_around_spot = st.toggle("Center around spot (±20%)", value=True)

            agg = build_gamma_by_strike(details, u, expiry, strike_bin)

            if agg.empty:
                st.warning("No rows after filtering.")
            else:
                # Rank then re-sort for charting
                plot_df = agg.sort_values("abs_net", ascending=False).head(int(top_n)).sort_values("strike_bin")

                spot = float(plot_df["spot"].dropna().iloc[0]) if plot_df["spot"].notna().any() else None
                if center_around_spot and spot:
                    lo, hi = 0.8 * spot, 1.2 * spot
                    centered = plot_df[(plot_df["strike_bin"] >= lo) & (plot_df["strike_bin"] <= hi)]
                    if len(centered) >= 15:
                        plot_df = centered

                fig = go.Figure()

                x = plot_df["strike_bin"].astype(float)

                if show_stacked:
                    fig.add_bar(name="Calls GEX", x=x, y=plot_df["call_gex"])
                    # show puts as negative so bars visually oppose calls (easy read)
                    fig.add_bar(name="Puts GEX (shown negative)", x=x, y=-plot_df["put_gex"])
                    fig.update_layout(barmode="relative")
                    y_title = "GEX (calls positive, puts negative for display)"
                else:
                    fig.add_bar(name="Net GEX (calls - puts)", x=x, y=plot_df["net_gex"])
                    y_title = "Net GEX"

                if spot:
                    fig.add_vline(x=spot, line_dash="dash", line_width=2)

                fig.update_layout(
                    height=460,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                fig.update_xaxes(title="Strike")
                fig.update_yaxes(title=y_title)

                st.plotly_chart(fig, use_container_width=True)

                # Quick table: top walls
                st.markdown("**Top strikes by |net|**")
                top_walls = agg.sort_values("abs_net", ascending=False).head(12)[
                    ["strike_bin", "net_gex", "call_gex", "put_gex", "n"]
                ].rename(columns={"strike_bin": "strike", "n": "contracts"})
                st.dataframe(top_walls, use_container_width=True, hide_index=True)

                st.caption(
                    "Note: Your `gex` in details appears unsigned by option type. "
                    "For a net view we use: net = call_gex - put_gex. "
                    "If your internal convention differs, we can flip that with a toggle."
                )

    # ========== Details ==========
    with tabs[1]:
        st.subheader("Details")
        if isinstance(details, pd.DataFrame):
            st.dataframe(details, use_container_width=True)
        else:
            st.json(details)

    # ========== Summary ==========
    with tabs[2]:
        st.subheader("Summary")
        if isinstance(summary, pd.DataFrame):
            st.dataframe(summary, use_container_width=True)
        else:
            st.json(summary)

    # ========== Narrative ==========
    with tabs[3]:
        st.subheader("Narrative")
        if isinstance(narrative, pd.DataFrame):
            st.dataframe(narrative, use_container_width=True)
        else:
            st.write(narrative)
