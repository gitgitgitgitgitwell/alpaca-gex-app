# CLAUDE.md

This file provides guidance for AI assistants working with the Alpaca GEX App codebase.

## Project Overview

Alpaca GEX Scanner is a **dealer gamma exposure (GEX) analysis tool** for options trading. It fetches live option chain data from the Alpaca Securities API, calculates gamma exposure per strike, identifies gamma walls and zero-gamma strikes, classifies market regimes, and presents results in an interactive Streamlit web app.

**Domain**: Options market microstructure, dealer hedging analysis.

## Repository Structure

```
alpaca-gex-app/
├── app.py                      # Streamlit web UI (charts, tables, narrative display)
├── dealer_flow_alpaca.py       # Core analysis engine (API calls, GEX math, regime logic)
├── requirements.txt            # Python dependencies (streamlit, pandas, alpaca-py, plotly)
├── .devcontainer/
│   └── devcontainer.json       # Dev Container config (Python 3.11, auto-install, port 8501)
└── .gitignore
```

There are only two source files:
- **`dealer_flow_alpaca.py`** (~920 lines) - All data fetching, parsing, GEX calculation, regime classification, and narrative generation.
- **`app.py`** (~360 lines) - Streamlit UI: sidebar config, charting with Plotly, styled tables, session state management.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Web framework | Streamlit |
| Data | pandas, numpy |
| Visualization | Plotly |
| API client | alpaca-py |
| Container | Dev Containers (VS Code / Codespaces) |

## Running the Application

### Dev Container (preferred)
Open in VS Code with Dev Containers extension or GitHub Codespaces. Dependencies install automatically and the app starts on port 8501.

### Manual
```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
pip install -r requirements.txt
streamlit run app.py
```

### Standalone script (CLI, outputs CSVs)
```bash
python dealer_flow_alpaca.py
```
Produces: `alpaca_dealer_details.csv`, `alpaca_dealer_summary.csv`, `alpaca_dealer_narrative.csv`

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALPACA_API_KEY` | (required) | Alpaca API key |
| `ALPACA_SECRET_KEY` | (required) | Alpaca API secret |
| `COMPUTE_VWAP_METRICS` | `"1"` | Enable/disable VWAP z-score computation |
| `VWAP_BARS_CHUNK_SIZE` | `50` | Batch size for intraday bar fetching |

## No Build/Lint/Test Infrastructure

There is no build step, linter config, formatter config, or test suite. The project runs directly as Python scripts. There is no CI/CD pipeline.

## Architecture & Data Flow

1. **Input**: List of ticker symbols (configurable in UI sidebar or `TICKERS` constant)
2. **API fetch**: For each ticker: spot price, option chain snapshot, open interest (paginated), intraday bars
3. **Processing**: Parse OCC symbols, apply moneyness filter (+-20%), compute GEX with expiry bucket weights
4. **Analysis**: Aggregate by strike, find call/put walls, zero-gamma strike, classify regime, generate narrative
5. **Output**: Three DataFrames (`details_df`, `summary_df`, `narrative_df`) rendered in Streamlit tabs

`run_scan(tickers)` in `dealer_flow_alpaca.py` is the main orchestration function.

## Key Domain Concepts

- **GEX** (Gamma Exposure): `gamma * open_interest * contract_multiplier * expiry_weight`. Positive for calls, negative for puts.
- **Net GEX**: `call_gex - put_gex`. Positive = dealers are long gamma (stabilizing); negative = short gamma (amplifying).
- **Call/Put Wall**: Strike with the highest absolute GEX concentration for calls/puts.
- **Zero-Gamma Strike**: Strike where cumulative net gamma crosses zero.
- **Regime**: LONG GAMMA, SHORT GAMMA, PINNED, NEUTRAL, MEAN-REVERT - classified based on net GEX, proximity to walls, and VWAP metrics.
- **VWAP Z-score** (`vwap_z`): `(spot - session_vwap) / vwap_sigma` - measures intraday price deviation.
- **ZG Distance** (`zg_dist`): Normalized distance from spot to zero-gamma strike.

## Key Configuration Constants (dealer_flow_alpaca.py)

```python
MAX_DAYS_TO_EXPIRY = 30
CONTRACT_MULTIPLIER = 100.0
USE_REAL_OPEN_INTEREST = True
PAPER_TRADING = True
USE_MONEYNESS_FILTER = True
MONEYNESS_PCT = 0.20              # +/- 20% around spot
USE_EXPIRY_BUCKET_WEIGHTS = True
EXPIRY_BUCKETS = [(0, 7, 1.0), (8, 21, 0.7), (22, 30, 0.4)]
```

## Data Models

Two dataclasses in `dealer_flow_alpaca.py`:

- **`OptionPoint`**: Individual option contract with symbol, strike, gamma, OI, computed GEX, DTE, moneyness, weight, spot.
- **`TickerSummary`**: Per-ticker aggregates: call/put/net GEX totals, wall strikes, zero-gamma strike.

## Code Conventions

- **Type hints**: Functions use `Optional[T]`, `List`, `Dict`, `Tuple` type annotations.
- **Defensive data handling**: `_safe_float()` for nullable API values; try/except around API calls with fallbacks (trade price -> quote midpoint).
- **Pandas patterns**: `.copy()` on slices, `.groupby().agg()`, numeric coercion with `errors="coerce"`.
- **Logging**: `print()` statements for progress tracking (OI pages fetched, contracts parsed, filters applied).
- **Section headers**: `# ========= SECTION NAME =========` dividers in source.
- **No abstractions**: Flat procedural style; no classes beyond dataclasses, no dependency injection.

## Important Gotchas

1. **Feed selection**: Stock bars must use `feed="iex"` to avoid SIP entitlement errors on free/paper accounts.
2. **OCC symbol parsing**: Option symbols follow OCC format (e.g., `SPY251121C00450000`). The parser in `parse_occ_symbol()` handles edge cases with multi-character underlying tickers.
3. **Timezone handling**: All datetime operations use UTC-aware timestamps. RTH (Regular Trading Hours) is 9:30-16:00 ET.
4. **API pagination**: Open interest fetching uses `next_page_token` loop; must handle exhaustion gracefully.
5. **GEX sign convention**: In `app.py`, raw GEX values are positive for both calls and puts; the sign flip (puts negative) is applied during chart aggregation, not in the core engine.
6. **Pinned threshold**: Differs for indices (0.1% of spot) vs. single stocks (0.6% of spot).

## Streamlit UI Structure (app.py)

- **Sidebar**: Ticker textarea, VWAP toggle, Run Scan button
- **Tab 1 - Charts**: Plotly bar chart of GEX by strike (binned), with expiry filter, strike bin selector, stacked/net toggle, spot line overlay
- **Tab 2 - Details**: Raw option contract data table
- **Tab 3 - Summary**: Per-ticker GEX aggregates
- **Tab 4 - Narrative**: Regime classifications with color-coded VWAP z-score and ZG distance metrics
- **Session state**: Results persist in `st.session_state` across Streamlit reruns

## Making Changes

When modifying this codebase:
- **GEX calculation changes** go in `dealer_flow_alpaca.py` (functions: `build_option_points_for_ticker`, `summarize_ticker`)
- **Regime logic changes** go in `classify_regime()` and `english_summary_for_ticker()` in `dealer_flow_alpaca.py`
- **UI/display changes** go in `app.py`
- **New tickers**: Update `TICKERS` list in `dealer_flow_alpaca.py` and `DEFAULT_TICKERS` in `app.py`
- **New dependencies**: Add to `requirements.txt` (no version pinning convention except plotly)
- Test changes by running `streamlit run app.py` locally with valid Alpaca API credentials
