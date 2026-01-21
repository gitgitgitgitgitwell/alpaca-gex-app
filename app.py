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
DEFAULT_TICKERS = ["SPY", "QQQ", "TSM", "NVDA", "GOOGL", "RMBS", "VRT", "MRVL", "MU", "CRDO", "APH", "ALAB",
                   "ANET", "PRIM", "LRCX", "AMBA", "DOV", "AVGO", "SEI", "ABBNY", "VICR", "COHR", "FLNC", "LGN", "GTLB",
                   "PRYMY", "SNDK", "AMZN", "BE", "WMB", "Q", "DIOD", "COHU", "ACMR", "LWLG", "WRD", "NVTS", "GSIT", "AR", "NVCR", "INTC", "POET", "DBRG"]

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
# Regime-aware metrics display helpers
# ----------------------------
def is_long_gamma_regime(regime: str) -> bool:
    if not isinstance(regime, str):
        return False
    r = regime.upper()
    return ("LONG GAMMA" in r) or ("MEAN-REVERT" in r) or ("PINNED" in r)

def _color_vwap_z(val):
    if pd.isna(val):
        return ""
    a = abs(float(val))
    if a < 0.5:
        return "background-color: rgba(0,255,0,0.12)"
    if a < 1.0:
        return "background-color: rgba(255,255,0,0.18)"
    if a < 1.5:
        return "background-color: rgba(255,165,0,0.22)"
    return "background-color: rgba(255,0,0,0.22)"

def _color_zg_dist(val):
    if pd.isna(val):
        return ""
    v = float(val)
    if v >= 1.5:
        return "background-color: rgba(0,255,0,0.12)"
    if v >= 1.0:
        return "background-color: rgba(255,255,0,0.18)"
    if v >= 0.7:
        return "background-color: rgba(255,165,0,0.22)"
    return "background-color: rgba(255,0,0,0.22)"

def _grey_inactive_vwap(row: pd.Series):
    active = bool(row.get("vwap_z_active", False))
    if not active:
        return ["color: rgba(255,255,255,0.45)" if c == "vwap_z" else "" for c in row.index]
    return ["" for _ in row.index]

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

        if isinstance(narrative, pd.DataFrame) and not narrative.empty:
            df = narrative.copy()

            # Back-compat: compute metrics if older scan doesn't include them
            if "zg_dev_pct" not in df.columns and {"spot","zero_gamma_strike"}.issubset(df.columns):
                df["zg_dev_pct"] = np.where(
                    (pd.to_numeric(df["spot"], errors="coerce") != 0),
                    (pd.to_numeric(df["spot"], errors="coerce") - pd.to_numeric(df["zero_gamma_strike"], errors="coerce"))
                    / pd.to_numeric(df["spot"], errors="coerce") * 100.0,
                    np.nan
                )

            if "zg_dist" not in df.columns and {"spot","put_wall_strike","zero_gamma_strike"}.issubset(df.columns):
                spot = pd.to_numeric(df["spot"], errors="coerce")
                zg = pd.to_numeric(df["zero_gamma_strike"], errors="coerce")
                pw = pd.to_numeric(df["put_wall_strike"], errors="coerce")
                denom = (pw - zg)
                df["zg_dist"] = np.where(denom != 0, (spot - zg) / denom, np.nan)

            # Regime-aware VWAP Z interpretation
            if "vwap_z" in df.columns:
                df["vwap_z_active"] = df.get("regime", "").apply(is_long_gamma_regime)
            else:
                df["vwap_z_active"] = False

            # Choose columns (only those that exist)
            preferred_cols = [
                "ticker","spot","regime","confidence",
                "session_vwap","vwap_sigma","vwap_z",
                "put_wall_strike","call_wall_strike","zero_gamma_strike",
                "zg_dev_pct","zg_dist",
                "net_gex","call_gex_total","put_gex_total",
                "oi_real_ratio","kept_contracts","moneyness_removed","oi_real_used","oi_proxy_used",
                "summary"
            ]
            cols = [c for c in preferred_cols if c in df.columns]
            view = df[cols].copy()

            styled = (
                view.style
                    .apply(_grey_inactive_vwap, axis=1)
                    .applymap(_color_vwap_z, subset=["vwap_z"] if "vwap_z" in view.columns else [])
                    .applymap(_color_zg_dist, subset=["zg_dist"] if "zg_dist" in view.columns else [])
                    .format({
                        "spot":"{:.2f}",
                        "session_vwap":"{:.2f}",
                        "vwap_sigma":"{:.2f}",
                        "vwap_z":"{:+.2f}",
                        "put_wall_strike":"{:.2f}",
                        "call_wall_strike":"{:.2f}",
                        "zero_gamma_strike":"{:.2f}",
                        "zg_dev_pct":"{:+.1f}%",
                        "zg_dist":"{:.2f}",
                        "net_gex":"{:+,.0f}",
                        "call_gex_total":"{:+,.0f}",
                        "put_gex_total":"{:+,.0f}",
                        "oi_real_ratio":"{:.0%}",
                    }, na_rep="—")
            )

            st.dataframe(styled, use_container_width=True, height=620)

            # Optional quick risk panel: lowest ZG distance
            if "zg_dist" in df.columns:
                with st.expander("Risk panel: lowest ZG distance (structural fragility)", expanded=False):
                    tmp = df[["ticker","zg_dist"]].copy()
                    tmp["zg_dist"] = pd.to_numeric(tmp["zg_dist"], errors="coerce")
                    tmp = tmp.dropna().sort_values("zg_dist").head(12)
                    if not tmp.empty:
                        fig = go.Figure()
                        fig.add_bar(x=tmp["zg_dist"], y=tmp["ticker"], orientation="h")
                        fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
                        fig.update_xaxes(title="ZG distance (normalized)")
                        fig.update_yaxes(title="")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("No ZG distance values available.")

        else:
            st.write(narrative)
