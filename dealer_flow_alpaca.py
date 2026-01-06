'''"""
dealer_flow_alpaca.py (snapshot-only + real OI + moneyness/expiry filters)

- Pulls option chain snapshots for each underlying (SPY, QQQ, etc.)
- Parses OCC option symbols to get strike, expiry, call/put
- Uses gamma from greeks
- Pulls real open interest from the Trading API (contracts), with pagination
- Filters contracts by moneyness around spot
- Applies expiry-bucket weights when computing GEX
- Computes call/put GEX, "walls", and a rough zero-gamma strike

Prereqs:
    pip install alpaca-py

Env vars:
    ALPACA_API_KEY, ALPACA_SECRET_KEY
"""

import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockLatestQuoteRequest

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus


# ========= CONFIG =========

# Tickers you want to scan
TICKERS = ["SPY", "QQQ", "TSM", "NVDA", "GOOGL", "RMBS", "VRT", "MRVL", "MU", "CRDO", "APH", "ALAB", "ANET", "PRIM", "LRCX", "MOD", "DOV", "AVGO", "SEI", "ABBNY", "PRYMY", "SNDK", "AMZN", "BE", "WMB", "AR", "NVCR", "INTC", "XYL", "HNGE", "POET", "DBRG"]

# Only look at expiries within this many calendar days from today
MAX_DAYS_TO_EXPIRY = 30

# Contract multiplier for equity options
CONTRACT_MULTIPLIER = 100.0

# If True, use real open interest from Trading API when available
USE_REAL_OPEN_INTEREST = True

# Use paper trading endpoint for TradingClient (doesn't matter for data client)
PAPER_TRADING = True

# --- Moneyness filter settings ---

# If True, only keep strikes within +/- MONEYNESS_PCT of spot
USE_MONEYNESS_FILTER = True
MONEYNESS_PCT = 0.20  # 0.20 = +/- 20% around spot

# --- Expiry bucket weighting ---

# If True, weight nearer expiries more when computing GEX
USE_EXPIRY_BUCKET_WEIGHTS = True

# List of (min_dte, max_dte, weight) buckets
# e.g. 0-7D full weight, 8-21D medium, 22+ lighter
EXPIRY_BUCKETS: List[Tuple[int, int, float]] = [
    (0, 7, 1.0),
    (8, 21, 0.7),
    (22, MAX_DAYS_TO_EXPIRY, 0.4),
]


# ========= DATA MODELS =========

@dataclass
class OptionPoint:
    symbol: str
    underlying: str
    type: str        # "call" or "put"
    strike: float
    expiration: datetime
    gamma: float
    open_interest: float
    gex: float       # gamma exposure (weighted)
    dte: int
    moneyness: float
    expiry_weight: float
    spot: Optional[float]


@dataclass
class TickerSummary:
    ticker: str
    call_gex_total: float
    put_gex_total: float
    net_gex: float
    call_wall_strike: Optional[float]
    put_wall_strike: Optional[float]
    zero_gamma_strike: Optional[float]


# ========= HELPERS =========

def _get_keys() -> Tuple[str, str]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY env vars. "
            "Set them before running this script."
        )
    return api_key, secret_key


def get_data_client() -> OptionHistoricalDataClient:
    api_key, secret_key = _get_keys()
    return OptionHistoricalDataClient(api_key, secret_key)


def get_stock_client() -> StockHistoricalDataClient:
    api_key, secret_key = _get_keys()
    return StockHistoricalDataClient(api_key, secret_key)


def get_trading_client() -> TradingClient:
    api_key, secret_key = _get_keys()
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=PAPER_TRADING)


def parse_occ_symbol(symbol: str, underlying: str) -> Optional[Tuple[datetime, str, float]]:
    """
    Parse OCC-style option symbol:
      UNDERLY + YYMMDD + C/P + STRIKE(8 digits, x1000)

    Example: SPY251121C00450000 -> underlying=SPY, expiry=2025-11-21, type='call', strike=450.0
    We strip spaces in case of padded roots.
    """
    normalized = symbol.replace(" ", "")

    if not normalized.startswith(underlying):
        return None

    rest = normalized[len(underlying):]

    # Must at least have 6 (date) + 1 (cp) + 8 (strike) = 15 chars
    if len(rest) < 15:
        return None

    date_str = rest[0:6]
    cp_char = rest[6].upper()
    strike_str = rest[7:]

    try:
        yy = int(date_str[0:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        year = 2000 + yy
        expiry = datetime(year, mm, dd, tzinfo=timezone.utc)

        if cp_char == "C":
            opt_type = "call"
        elif cp_char == "P":
            opt_type = "put"
        else:
            return None

        strike = int(strike_str) / 1000.0
    except Exception:
        return None

    return expiry, opt_type, strike


def build_oi_map_for_ticker(
    ticker: str,
    trading_client: TradingClient,
    max_days_to_expiry: int = MAX_DAYS_TO_EXPIRY,
) -> Dict[Tuple[date, str, float], float]:
    """
    Use TradingClient.get_option_contracts() to get open_interest for ALL contracts
    on this underlying within the expiry window (handles pagination).

    Returns a map keyed by (expiry_date, opt_type, strike): open_interest.
    """
    today = date.today()
    min_exp = today
    max_exp = today + timedelta(days=max_days_to_expiry)

    base_req = GetOptionContractsRequest(
        underlying_symbols=[ticker],
        status=AssetStatus.ACTIVE,
        expiration_date_gte=min_exp,
        expiration_date_lte=max_exp,
    )

    oi_map: Dict[Tuple[date, str, float], float] = {}
    num_contracts_total = 0
    num_with_oi = 0
    num_parsed = 0
    page = 0

    req = base_req
    while True:
        page += 1
        resp = trading_client.get_option_contracts(req)
        contracts = getattr(resp, "option_contracts", resp)
        if not contracts:
            break

        for c in contracts:
            num_contracts_total += 1

            symbol = getattr(c, "symbol", None)
            oi_raw = getattr(c, "open_interest", None)
            if symbol is None or oi_raw is None:
                continue

            try:
                oi_val = float(oi_raw)
            except Exception:
                continue

            if oi_val <= 0:
                continue

            parsed = parse_occ_symbol(symbol, ticker)
            if parsed is None:
                continue

            exp_dt, opt_type, strike = parsed
            num_parsed += 1

            key = (exp_dt.date(), opt_type, float(strike))
            oi_map[key] = oi_map.get(key, 0.0) + oi_val
            num_with_oi += 1

        next_token = getattr(resp, "next_page_token", None)
        if not next_token:
            break

        req = GetOptionContractsRequest(
            underlying_symbols=[ticker],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=min_exp,
            expiration_date_lte=max_exp,
            page_token=next_token,
        )

    print(
        f"  [OI] pages={page}, contracts={num_contracts_total}, "
        f"parsed={num_parsed}, with_open_interest={num_with_oi}, "
        f"unique_keys={len(oi_map)}"
    )
    return oi_map


def get_expiry_weight(dte: int) -> float:
    if not USE_EXPIRY_BUCKET_WEIGHTS:
        return 1.0
    for dmin, dmax, w in EXPIRY_BUCKETS:
        if dmin <= dte <= dmax:
            return w
    # Outside defined buckets -> effectively ignore
    return 0.0


def get_spot_price(ticker: str, stock_client: StockHistoricalDataClient) -> Optional[float]:
    """
    Prefer last trade price instead of bid/ask midpoint.
    This avoids bad/stale quotes for illiquid names.
    """
    from alpaca.data.requests import StockLatestTradeRequest

    try:
        req = StockLatestTradeRequest(symbol_or_symbols=ticker)
        resp = stock_client.get_stock_latest_trade(req)
        trade = resp[ticker]
        if trade and trade.price:
            return float(trade.price)
    except Exception as e:
        print(f"[WARN] Failed to get trade spot for {ticker}: {e}")

    # fallback to quote midpoint
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        resp = stock_client.get_stock_latest_quote(req)
        q = resp[ticker]
        if q.ask_price and q.bid_price and q.ask_price > 0 and q.bid_price > 0:
            return float((q.ask_price + q.bid_price) / 2)
        if q.ask_price and q.ask_price > 0:
            return float(q.ask_price)
        if q.bid_price and q.bid_price > 0:
            return float(q.bid_price)
    except:
        pass

    return None



def build_option_points_for_ticker(
    ticker: str,
    data_client: OptionHistoricalDataClient,
    oi_map: Optional[Dict[Tuple[date, str, float], float]] = None,
    spot_price: Optional[float] = None,
) -> List[OptionPoint]:
    print(f"\n=== Processing {ticker} ===")

    # 1) Pull chain snapshots for the underlying
    req = OptionChainRequest(underlying_symbol=ticker)
    chain_snapshots: Dict[str, any] = data_client.get_option_chain(req)

    print(f"  Loaded {len(chain_snapshots)} snapshots from Alpaca")

    points: List[OptionPoint] = []
    now = datetime.now(timezone.utc)

    num_parsed = 0
    num_with_gamma = 0
    num_with_real_oi = 0
    num_proxy_oi = 0
    num_moneyness_filtered = 0
    num_weight_filtered = 0

    for symbol, snap in chain_snapshots.items():
        parsed = parse_occ_symbol(symbol, ticker)
        if parsed is None:
            continue

        exp_dt, opt_type, strike = parsed
        dte = (exp_dt - now).days
        if dte < 0 or dte > MAX_DAYS_TO_EXPIRY:
            continue

        # Moneyness filter
        moneyness = 0.0
        if USE_MONEYNESS_FILTER and spot_price is not None:
            moneyness = (strike / spot_price) - 1.0
            if abs(moneyness) > MONEYNESS_PCT:
                num_moneyness_filtered += 1
                continue

        num_parsed += 1

        # Require greeks + gamma
        if not getattr(snap, "greeks", None) or snap.greeks.gamma is None:
            continue

        gamma = float(snap.greeks.gamma)
        num_with_gamma += 1

        # Expiry weight
        expiry_weight = get_expiry_weight(dte)
        if expiry_weight <= 0.0:
            num_weight_filtered += 1
            continue

        # Open interest: lookup by (expiry_date, type, strike)
        if oi_map is not None:
            key = (exp_dt.date(), opt_type, float(strike))
            oi = float(oi_map.get(key, 0.0))
            if oi > 0:
                num_with_real_oi += 1
            else:
                oi = 1.0
                num_proxy_oi += 1
        else:
            oi = 1.0
            num_proxy_oi += 1

        # Weighted GEX: gamma * OI * contract multiplier * expiry_weight
        gex_val = gamma * oi * CONTRACT_MULTIPLIER * expiry_weight

        point = OptionPoint(
            symbol=symbol,
            underlying=ticker,
            type=opt_type,
            strike=strike,
            expiration=exp_dt,
            gamma=gamma,
            open_interest=oi,
            gex=gex_val,
            dte=dte,
            moneyness=moneyness,
            expiry_weight=expiry_weight,
            spot=spot_price,
        )
        points.append(point)

    print(
        f"  Parsed {num_parsed} symbols within {MAX_DAYS_TO_EXPIRY}D, "
        f"{num_with_gamma} with gamma, {len(points)} kept"
    )
    if USE_MONEYNESS_FILTER and spot_price is not None:
        print(f"  Moneyness filter removed {num_moneyness_filtered} contracts")
    if USE_EXPIRY_BUCKET_WEIGHTS:
        print(f"  Expiry weight filter removed {num_weight_filtered} contracts with 0 weight")

    if oi_map is not None:
        print(
            f"  OI usage: {num_with_real_oi} with real OI, "
            f"{num_proxy_oi} used proxy OI=1.0"
        )

    return points


def summarize_ticker(points: List[OptionPoint], ticker: str) -> TickerSummary:
    if not points:
        return TickerSummary(
            ticker=ticker,
            call_gex_total=0.0,
            put_gex_total=0.0,
            net_gex=0.0,
            call_wall_strike=None,
            put_wall_strike=None,
            zero_gamma_strike=None,
        )

    df = pd.DataFrame([p.__dict__ for p in points])

    # Totals
    call_gex_total = float(df.loc[df["type"] == "call", "gex"].sum())
    put_gex_total = float(df.loc[df["type"] == "put", "gex"].sum())
    net_gex = call_gex_total - put_gex_total

    # Walls – by total GEX per strike
    call_by_strike = (
        df[df["type"] == "call"].groupby("strike")["gex"].sum().sort_values(ascending=False)
    )
    put_by_strike = (
        df[df["type"] == "put"].groupby("strike")["gex"].sum().sort_values(ascending=False)
    )

    call_wall_strike = float(call_by_strike.index[0]) if len(call_by_strike) else None
    put_wall_strike = float(put_by_strike.index[0]) if len(put_by_strike) else None

    # Rough zero-gamma level (using weighted gex):
    net_by_strike = (
        df.groupby(["strike", "type"])["gex"]
        .sum()
        .unstack(fill_value=0.0)
        .assign(net=lambda x: x.get("call", 0.0) - x.get("put", 0.0))
        .reset_index()
        .sort_values("strike")
    )

    net_by_strike["cum_net_gex"] = net_by_strike["net"].cumsum()
    idx_min = net_by_strike["cum_net_gex"].abs().idxmin()
    zero_gamma_strike = float(net_by_strike.loc[idx_min, "strike"])

    return TickerSummary(
        ticker=ticker,
        call_gex_total=call_gex_total,
        put_gex_total=put_gex_total,
        net_gex=net_gex,
        call_wall_strike=call_wall_strike,
        put_wall_strike=put_wall_strike,
        zero_gamma_strike=zero_gamma_strike,
    )


def run_scan(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_client = get_data_client()
    stock_client = get_stock_client()
    trading_client: Optional[TradingClient] = None

    if USE_REAL_OPEN_INTEREST:
        print("Initializing TradingClient for real OI…")
        trading_client = get_trading_client()

    all_points: List[OptionPoint] = []
    summaries: List[TickerSummary] = []

    for t in tickers:
        spot = get_spot_price(t, stock_client) if USE_MONEYNESS_FILTER else None
        if spot is not None:
            print(f"\n>>> {t} spot ~ {spot:.2f}")
        else:
            print(f"\n>>> {t} spot unavailable - skipping moneyness filter for this ticker")

        oi_map = None
        if trading_client is not None:
            try:
                oi_map = build_oi_map_for_ticker(t, trading_client)
            except Exception as e:
                print(f"  [WARN] Failed to fetch OI for {t}: {e}")
                oi_map = None

        pts = build_option_points_for_ticker(
            t,
            data_client,
            oi_map=oi_map,
            spot_price=spot,
        )
        all_points.extend(pts)
        summaries.append(summarize_ticker(pts, t))

    details_df = pd.DataFrame([p.__dict__ for p in all_points])
    summary_df = pd.DataFrame([s.__dict__ for s in summaries])

    return details_df, summary_df


# ========= MAIN =========

if __name__ == "__main__":
    details, summary = run_scan(TICKERS)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", None)

    print("\n=== SUMMARY BY TICKER ===")
    print(summary.to_string(index=False))

    details.to_csv("alpaca_dealer_details.csv", index=False)
    summary.to_csv("alpaca_dealer_summary.csv", index=False)
    print("\nSaved: alpaca_dealer_details.csv, alpaca_dealer_summary.csv")
'''

"""
dealer_flow_alpaca.py (snapshot + real OI + moneyness/expiry filters + regime classifier + English summaries)

What it does:
- Pulls option chain snapshots for each underlying (Alpaca OptionHistoricalDataClient)
- Parses OCC option symbols to get strike, expiry, call/put
- Uses gamma from greeks
- Pulls real open interest from Alpaca Trading API (GetOptionContracts) with pagination
- Filters contracts by moneyness around spot (spot from latest trade, fallback to quote)
- Applies expiry-bucket weights to emphasize nearer expiries
- Computes call/put GEX, "walls" + rough zero-gamma strike
- Adds automatic regime classification + per-ticker English narrative

Prereqs:
  pip install alpaca-py pandas

Env vars:
  ALPACA_API_KEY, ALPACA_SECRET_KEY
"""

import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockLatestQuoteRequest, StockLatestTradeRequest

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus


# ========= CONFIG =========

TICKERS = ["SPY", "QQQ", "NVDA", "GOOGL", "RMBS", "VRT", "MRVL", "MU", "CRDO", "APH", "ALAB", "ANET", "PRIM", "LRCX", "MOD", "DOV", "AVGO", "SEI", "ABBNY", "VICR", "COHR", "PRYMY", "SNDK", "AMZN", "BE", "WMB", "AR", "NVCR", "INTC", "RIVN", "POET", "DBRG"]

MAX_DAYS_TO_EXPIRY = 30
CONTRACT_MULTIPLIER = 100.0

USE_REAL_OPEN_INTEREST = True
PAPER_TRADING = True

# --- Moneyness filter (around spot) ---
USE_MONEYNESS_FILTER = True
MONEYNESS_PCT = 0.20  # +/- 20%

# --- Expiry bucket weighting ---
USE_EXPIRY_BUCKET_WEIGHTS = True
EXPIRY_BUCKETS: List[Tuple[int, int, float]] = [
    (0, 7, 1.0),
    (8, 21, 0.7),
    (22, MAX_DAYS_TO_EXPIRY, 0.4),
]


# ========= DATA MODELS =========

@dataclass
class OptionPoint:
    symbol: str
    underlying: str
    type: str        # "call" or "put"
    strike: float
    expiration: datetime
    gamma: float
    open_interest: float
    gex: float       # gamma exposure (weighted)
    dte: int
    moneyness: float
    expiry_weight: float
    spot: Optional[float]


@dataclass
class TickerSummary:
    ticker: str
    call_gex_total: float
    put_gex_total: float
    net_gex: float
    call_wall_strike: Optional[float]
    put_wall_strike: Optional[float]
    zero_gamma_strike: Optional[float]


# ========= CORE CLIENTS =========

def _get_keys() -> Tuple[str, str]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY env vars. "
            "Set them before running this script."
        )
    return api_key, secret_key


def get_option_data_client() -> OptionHistoricalDataClient:
    api_key, secret_key = _get_keys()
    return OptionHistoricalDataClient(api_key, secret_key)


def get_stock_client() -> StockHistoricalDataClient:
    api_key, secret_key = _get_keys()
    return StockHistoricalDataClient(api_key, secret_key)


def get_trading_client() -> TradingClient:
    api_key, secret_key = _get_keys()
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=PAPER_TRADING)


# ========= SYMBOL PARSING / LOOKUPS =========

def parse_occ_symbol(symbol: str, underlying: str) -> Optional[Tuple[datetime, str, float]]:
    """
    Parse OCC-style option symbol:
      ROOT + YYMMDD + C/P + STRIKE(8 digits, x1000)

    Example: SPY251121C00450000 -> expiry=2025-11-21, type='call', strike=450.0

    NOTE: We strip spaces to handle padded roots.
    """
    normalized = symbol.replace(" ", "")
    if not normalized.startswith(underlying):
        return None

    rest = normalized[len(underlying):]
    if len(rest) < 15:  # 6 date + 1 cp + 8 strike
        return None

    date_str = rest[0:6]
    cp_char = rest[6].upper()
    strike_str = rest[7:]

    try:
        yy = int(date_str[0:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        year = 2000 + yy
        expiry = datetime(year, mm, dd, tzinfo=timezone.utc)

        if cp_char == "C":
            opt_type = "call"
        elif cp_char == "P":
            opt_type = "put"
        else:
            return None

        strike = int(strike_str) / 1000.0
    except Exception:
        return None

    return expiry, opt_type, strike


def get_expiry_weight(dte: int) -> float:
    if not USE_EXPIRY_BUCKET_WEIGHTS:
        return 1.0
    for dmin, dmax, w in EXPIRY_BUCKETS:
        if dmin <= dte <= dmax:
            return w
    return 0.0


def get_spot_price(ticker: str, stock_client: StockHistoricalDataClient) -> Optional[float]:
    """
    FIX #1: Prefer latest TRADE price (more reliable than quote mid on thin names),
    then fallback to quote midpoint/ask/bid.
    """
    # Prefer last trade
    try:
        t_req = StockLatestTradeRequest(symbol_or_symbols=ticker)
        t_resp = stock_client.get_stock_latest_trade(t_req)
        trade = t_resp[ticker]
        px = getattr(trade, "price", None)
        if px is not None and float(px) > 0:
            return float(px)
    except Exception:
        pass

    # Fallback to quote
    try:
        q_req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        q_resp = stock_client.get_stock_latest_quote(q_req)
        quote = q_resp[ticker]
        ask = getattr(quote, "ask_price", None)
        bid = getattr(quote, "bid_price", None)
        if ask is not None and bid is not None and float(ask) > 0 and float(bid) > 0:
            return float((float(ask) + float(bid)) / 2.0)
        if ask is not None and float(ask) > 0:
            return float(ask)
        if bid is not None and float(bid) > 0:
            return float(bid)
    except Exception:
        pass

    return None


# ========= OI FETCH (PAGINATED) =========

def build_oi_map_for_ticker(
    ticker: str,
    trading_client: TradingClient,
    max_days_to_expiry: int = MAX_DAYS_TO_EXPIRY,
) -> Dict[Tuple[date, str, float], float]:
    """
    Pull open_interest for ALL contracts on this underlying within expiry window (handles pagination).
    Keyed by (expiry_date, opt_type, strike) -> OI
    """
    today = date.today()
    min_exp = today
    max_exp = today + timedelta(days=max_days_to_expiry)

    base_req = GetOptionContractsRequest(
        underlying_symbols=[ticker],
        status=AssetStatus.ACTIVE,
        expiration_date_gte=min_exp,
        expiration_date_lte=max_exp,
    )

    oi_map: Dict[Tuple[date, str, float], float] = {}

    num_contracts_total = 0
    num_with_oi = 0
    num_parsed = 0
    page = 0

    req = base_req
    while True:
        page += 1
        resp = trading_client.get_option_contracts(req)
        contracts = getattr(resp, "option_contracts", resp)
        if not contracts:
            break

        for c in contracts:
            num_contracts_total += 1
            symbol = getattr(c, "symbol", None)
            oi_raw = getattr(c, "open_interest", None)
            if symbol is None or oi_raw is None:
                continue

            try:
                oi_val = float(oi_raw)
            except Exception:
                continue

            if oi_val <= 0:
                continue

            parsed = parse_occ_symbol(symbol, ticker)
            if parsed is None:
                continue

            exp_dt, opt_type, strike = parsed
            num_parsed += 1

            key = (exp_dt.date(), opt_type, float(strike))
            oi_map[key] = oi_map.get(key, 0.0) + oi_val
            num_with_oi += 1

        next_token = getattr(resp, "next_page_token", None)
        if not next_token:
            break

        req = GetOptionContractsRequest(
            underlying_symbols=[ticker],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=min_exp,
            expiration_date_lte=max_exp,
            page_token=next_token,
        )

    print(
        f"  [OI] pages={page}, contracts={num_contracts_total}, "
        f"parsed={num_parsed}, with_open_interest={num_with_oi}, "
        f"unique_keys={len(oi_map)}"
    )
    return oi_map


# ========= BUILD POINTS (SNAPSHOTS + FILTERS + WEIGHTS) =========

def build_option_points_for_ticker(
    ticker: str,
    data_client: OptionHistoricalDataClient,
    oi_map: Optional[Dict[Tuple[date, str, float], float]],
    spot_price: Optional[float],
) -> Tuple[List[OptionPoint], Dict[str, int]]:
    print(f"\n=== Processing {ticker} ===")

    req = OptionChainRequest(underlying_symbol=ticker)
    chain_snapshots: Dict[str, Any] = data_client.get_option_chain(req)
    print(f"  Loaded {len(chain_snapshots)} snapshots from Alpaca")

    points: List[OptionPoint] = []
    now = datetime.now(timezone.utc)

    num_parsed = 0
    num_with_gamma = 0
    num_with_real_oi = 0
    num_proxy_oi = 0
    num_moneyness_filtered = 0
    num_weight_filtered = 0

    for symbol, snap in chain_snapshots.items():
        parsed = parse_occ_symbol(symbol, ticker)
        if parsed is None:
            continue

        exp_dt, opt_type, strike = parsed
        dte = (exp_dt - now).days
        if dte < 0 or dte > MAX_DAYS_TO_EXPIRY:
            continue

        # Moneyness filter
        moneyness = 0.0
        if USE_MONEYNESS_FILTER and spot_price is not None:
            moneyness = (strike / spot_price) - 1.0
            if abs(moneyness) > MONEYNESS_PCT:
                num_moneyness_filtered += 1
                continue

        num_parsed += 1

        # Need gamma from greeks
        if not getattr(snap, "greeks", None) or getattr(snap.greeks, "gamma", None) is None:
            continue

        gamma = float(snap.greeks.gamma)
        num_with_gamma += 1

        # Expiry weighting
        expiry_weight = get_expiry_weight(dte)
        if expiry_weight <= 0.0:
            num_weight_filtered += 1
            continue

        # OI mapping
        if oi_map is not None:
            key = (exp_dt.date(), opt_type, float(strike))
            oi = float(oi_map.get(key, 0.0))
            if oi > 0:
                num_with_real_oi += 1
            else:
                oi = 1.0
                num_proxy_oi += 1
        else:
            oi = 1.0
            num_proxy_oi += 1

        gex_val = gamma * oi * CONTRACT_MULTIPLIER * expiry_weight

        points.append(
            OptionPoint(
                symbol=symbol,
                underlying=ticker,
                type=opt_type,
                strike=strike,
                expiration=exp_dt,
                gamma=gamma,
                open_interest=oi,
                gex=gex_val,
                dte=dte,
                moneyness=moneyness,
                expiry_weight=expiry_weight,
                spot=spot_price,
            )
        )

    print(
        f"  Parsed {num_parsed} symbols within {MAX_DAYS_TO_EXPIRY}D, "
        f"{num_with_gamma} with gamma, {len(points)} kept"
    )
    if USE_MONEYNESS_FILTER and spot_price is not None:
        print(f"  Moneyness filter removed {num_moneyness_filtered} contracts")
    if USE_EXPIRY_BUCKET_WEIGHTS:
        print(f"  Expiry weight filter removed {num_weight_filtered} contracts with 0 weight")
    if oi_map is not None:
        print(f"  OI usage: {num_with_real_oi} with real OI, {num_proxy_oi} used proxy OI=1.0")

    stats = {
        "kept_contracts": len(points),
        "moneyness_removed": num_moneyness_filtered,
        "oi_real_used": num_with_real_oi,
        "oi_proxy_used": num_proxy_oi,
    }
    return points, stats


# ========= SUMMARIZATION =========

def summarize_ticker(points: List[OptionPoint], ticker: str) -> TickerSummary:
    if not points:
        return TickerSummary(
            ticker=ticker,
            call_gex_total=0.0,
            put_gex_total=0.0,
            net_gex=0.0,
            call_wall_strike=None,
            put_wall_strike=None,
            zero_gamma_strike=None,
        )

    df = pd.DataFrame([p.__dict__ for p in points])

    call_gex_total = float(df.loc[df["type"] == "call", "gex"].sum())
    put_gex_total = float(df.loc[df["type"] == "put", "gex"].sum())
    net_gex = call_gex_total - put_gex_total

    call_by_strike = df[df["type"] == "call"].groupby("strike")["gex"].sum().sort_values(ascending=False)
    put_by_strike = df[df["type"] == "put"].groupby("strike")["gex"].sum().sort_values(ascending=False)

    call_wall_strike = float(call_by_strike.index[0]) if len(call_by_strike) else None
    put_wall_strike = float(put_by_strike.index[0]) if len(put_by_strike) else None

    net_by_strike = (
        df.groupby(["strike", "type"])["gex"]
        .sum()
        .unstack(fill_value=0.0)
        .assign(net=lambda x: x.get("call", 0.0) - x.get("put", 0.0))
        .reset_index()
        .sort_values("strike")
    )
    net_by_strike["cum_net_gex"] = net_by_strike["net"].cumsum()
    idx_min = net_by_strike["cum_net_gex"].abs().idxmin()
    zero_gamma_strike = float(net_by_strike.loc[idx_min, "strike"])

    return TickerSummary(
        ticker=ticker,
        call_gex_total=call_gex_total,
        put_gex_total=put_gex_total,
        net_gex=net_gex,
        call_wall_strike=call_wall_strike,
        put_wall_strike=put_wall_strike,
        zero_gamma_strike=zero_gamma_strike,
    )


# ========= REGIME CLASSIFIER + ENGLISH NARRATIVE =========

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def classify_regime(
    spot: Optional[float],
    net_gex: float,
    call_gex_total: float,
    put_gex_total: float,
    call_wall: Optional[float],
    put_wall: Optional[float],
    zero_gamma: Optional[float],
    kept_contracts: int,
    oi_real_ratio: Optional[float],
    ticker: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Returns (regime_label, confidence_label)

    Key improvements:
    - Uses normalized net share = net_gex / (call+put) for regime strength
    - Uses pin distance to nearest wall, with tighter thresholds for indices
    - Adds guardrails for thin chains / low OI coverage
    """

    # ===== Hard fail / incomplete =====
    if kept_contracts == 0 or spot is None:
        return ("UNRELIABLE / INCOMPLETE", "LOW")

    # We can still classify without walls/ZG, but we cap confidence.
    have_levels = (call_wall is not None and put_wall is not None)
    have_zg = (zero_gamma is not None)

    # ===== Confidence =====
    confidence = "MEDIUM"

    # Sample size / OI coverage caps
    if kept_contracts < 50:
        confidence = "LOW"
    if oi_real_ratio is not None and oi_real_ratio < 0.50:
        confidence = "LOW"
    if (oi_real_ratio is not None and oi_real_ratio >= 0.70 and kept_contracts >= 150):
        confidence = "HIGH"

    # If we don't have levels, don't pretend we're precise about pinning.
    if not have_levels or not have_zg:
        confidence = "LOW"

    # ===== Normalize the gamma imbalance =====
    total_gex = float(call_gex_total) + float(put_gex_total)
    net_share: float = 0.0
    if total_gex > 0:
        net_share = float(net_gex) / total_gex  # e.g. 0.0027 = +0.27%

    abs_share = abs(net_share)

    # Regime strength bands (tuned for practical use)
    # <1% = basically neutral, 1–5% = weak, >5% = strong
    if abs_share < 0.01:
        base = "NEUTRAL / MIXED"
    elif net_share > 0:
        base = "LONG GAMMA"
    else:
        base = "SHORT GAMMA"

    # ===== Pinning to walls =====
    pinned = False
    pin_dist = None
    if have_levels and call_wall is not None and put_wall is not None:
        pin_dist = min(abs(spot - call_wall), abs(spot - put_wall)) / spot

        # Index vs single-name thresholds
        is_index = (ticker in {"SPY", "QQQ", "IWM", "DIA"} ) if ticker else (spot >= 250)

        # Very tight pin definition for indices
        pin_thresh = 0.0010 if is_index else 0.0060  # 0.10% vs 0.60%
        pinned = (pin_dist is not None and pin_dist <= pin_thresh)

    # ===== Final label logic =====
    if base == "NEUTRAL / MIXED":
        # If neutral but pinned, call it pinned-neutral (market often “stuck”)
        if pinned:
            return ("PINNED / MAGNET (NEUTRAL GAMMA)", confidence)
        return ("NEUTRAL / MIXED", confidence)

    # Long/short gamma nuance vs ZG if available
    above_zg = (have_zg and zero_gamma is not None and spot > zero_gamma)

    if base == "LONG GAMMA":
        if pinned:
            return ("PINNED (LONG GAMMA NEAR WALL)", confidence)
        if above_zg:
            # Above ZG often means less “sticky” than classic below-ZG pinning,
            # but still mean-reverting vs short gamma.
            return ("LONG GAMMA (LOW VOL, MEAN-REVERTING)", confidence)
        return ("LONG GAMMA (LOW VOL, MEAN-REVERTING)", confidence)

    # SHORT GAMMA
    if pinned:
        # short gamma can still "magnet" near a wall until it breaks
        return ("PINNED / MAGNET (SHORT GAMMA NEAR WALL)", confidence)
    if above_zg:
        return ("SHORT GAMMA (HIGH VOL / CHASE REGIME)", confidence)
    return ("SHORT GAMMA (HIGH VOL BELOW ZG)", confidence)



def english_summary_for_ticker(
    ticker: str,
    spot: Optional[float],
    call_gex_total: float,
    put_gex_total: float,
    net_gex: float,
    call_wall: Optional[float],
    put_wall: Optional[float],
    zero_gamma: Optional[float],
    kept_contracts: int,
    moneyness_removed: int,
    oi_real_used: int,
    oi_proxy_used: int,
) -> str:
    spot_f = _safe_float(spot)
    call_wall_f = _safe_float(call_wall)
    put_wall_f = _safe_float(put_wall)
    zg_f = _safe_float(zero_gamma)

    oi_total_used = oi_real_used + oi_proxy_used
    oi_real_ratio = (oi_real_used / oi_total_used) if oi_total_used > 0 else None

    regime, confidence = classify_regime(
        spot=spot_f,
        net_gex=net_gex,
        call_gex_total=call_gex_total,
        put_gex_total=put_gex_total,
        call_wall=call_wall_f,
        put_wall=put_wall_f,
        zero_gamma=zg_f,
        kept_contracts=kept_contracts,
        oi_real_ratio=oi_real_ratio,
        ticker=ticker,
    )

    # Vol regime line (avoid directional claims)
    if net_gex > 0:
        flow_line = "Dealers net LONG gamma → hedging tends to dampen moves (buy dips / sell rips)."
    elif net_gex < 0:
        flow_line = "Dealers net SHORT gamma → hedging can amplify moves (sell weakness / buy strength)."
    else:
        flow_line = "Gamma looks mixed/near-neutral → weaker structural effect."

    # Levels line
    levels_bits: List[str] = []
    if spot_f is not None and put_wall_f is not None and call_wall_f is not None:
        levels_bits.append(f"Spot ~ {spot_f:.2f}. Put wall ~ {put_wall_f:.2f}; Call wall ~ {call_wall_f:.2f}.")
    if spot_f is not None and zg_f is not None:
        if spot_f > zg_f:
            levels_bits.append(f"Spot is ABOVE zero-gamma (~{zg_f:.2f}) → less pinned; more sensitive to flow/catalysts.")
        else:
            levels_bits.append(f"Spot is BELOW zero-gamma (~{zg_f:.2f}) → stickier tape; more stabilizing hedging.")

    # Data quality
    dq_bits = [
        f"Contracts kept: {kept_contracts}. Moneyness-filter removed: {moneyness_removed}."
    ]
    if oi_real_ratio is not None:
        dq_bits.append(f"Real OI coverage: {oi_real_used}/{oi_total_used} ({oi_real_ratio:.0%}).")

    # Today-style expectation
    if "PINNED" in regime:
        expectation = "Expected: range-bound / pinning around the nearest wall unless fresh options flow breaks it."
    elif "LONG GAMMA" in regime:
        expectation = "Expected: lower realized vol + more mean reversion; breakouts usually need a catalyst/flow."
    elif "SHORT GAMMA" in regime:
        expectation = "Expected: higher realized vol; moves can accelerate once price starts trending."
    else:
        expectation = "Expected: mixed structure; lean more on price action and catalysts."

    return (
        f"{ticker} — {regime} (confidence: {confidence}). "
        f"Net GEX = {net_gex:,.0f} (Call {call_gex_total:,.0f} vs Put {put_gex_total:,.0f}). "
        f"{flow_line} "
        + " ".join(levels_bits) + " "
        + " ".join(dq_bits) + " "
        + expectation
    )


# ========= RUN SCAN =========

def run_scan(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    option_client = get_option_data_client()
    stock_client = get_stock_client()
    trading_client: Optional[TradingClient] = None

    if USE_REAL_OPEN_INTEREST:
        print("Initializing TradingClient for real OI…")
        trading_client = get_trading_client()

    all_points: List[OptionPoint] = []
    summaries: List[TickerSummary] = []
    ticker_stats: Dict[str, Dict[str, Any]] = {}

    for t in tickers:
        spot = get_spot_price(t, stock_client) if USE_MONEYNESS_FILTER else None
        if spot is not None:
            print(f"\n>>> {t} spot ~ {spot:.2f}")
        else:
            print(f"\n>>> {t} spot unavailable - skipping moneyness filter for this ticker")

        oi_map = None
        if trading_client is not None:
            try:
                oi_map = build_oi_map_for_ticker(t, trading_client)
            except Exception as e:
                print(f"  [WARN] Failed to fetch OI for {t}: {e}")
                oi_map = None

        pts, stats = build_option_points_for_ticker(
            t,
            option_client,
            oi_map=oi_map,
            spot_price=spot,
        )

        all_points.extend(pts)
        summaries.append(summarize_ticker(pts, t))

        ticker_stats[t] = {**stats, "spot": spot}

    details_df = pd.DataFrame([p.__dict__ for p in all_points])
    summary_df = pd.DataFrame([s.__dict__ for s in summaries])

    # Build narrative/regime dataframe
    narrative_rows: List[Dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        t = row["ticker"]
        st = ticker_stats.get(t, {})
        spot = st.get("spot", None)

        kept_contracts = int(st.get("kept_contracts", 0) or 0)
        moneyness_removed = int(st.get("moneyness_removed", 0) or 0)
        oi_real_used = int(st.get("oi_real_used", 0) or 0)
        oi_proxy_used = int(st.get("oi_proxy_used", 0) or 0)

        call_wall = row.get("call_wall_strike", None)
        put_wall = row.get("put_wall_strike", None)
        zg = row.get("zero_gamma_strike", None)

        total_used = oi_real_used + oi_proxy_used
        oi_real_ratio = (oi_real_used / total_used) if total_used > 0 else None

        regime, confidence = classify_regime(
            spot=_safe_float(spot),
            net_gex=float(row["net_gex"]),
            call_gex_total=float(row["call_gex_total"]),
            put_gex_total=float(row["put_gex_total"]),
            call_wall=_safe_float(call_wall),
            put_wall=_safe_float(put_wall),
            zero_gamma=_safe_float(zg),
            kept_contracts=kept_contracts,
            oi_real_ratio=oi_real_ratio,
            ticker=t,
        )

        narrative = english_summary_for_ticker(
            ticker=t,
            spot=_safe_float(spot),
            call_gex_total=float(row["call_gex_total"]),
            put_gex_total=float(row["put_gex_total"]),
            net_gex=float(row["net_gex"]),
            call_wall=_safe_float(call_wall),
            put_wall=_safe_float(put_wall),
            zero_gamma=_safe_float(zg),
            kept_contracts=kept_contracts,
            moneyness_removed=moneyness_removed,
            oi_real_used=oi_real_used,
            oi_proxy_used=oi_proxy_used,
        )

        narrative_rows.append(
            {
                "ticker": t,
                "spot": _safe_float(spot),
                "regime": regime,
                "confidence": confidence,
                "oi_real_ratio": oi_real_ratio,
                "kept_contracts": kept_contracts,
                "moneyness_removed": moneyness_removed,
                "oi_real_used": oi_real_used,
                "oi_proxy_used": oi_proxy_used,
                "call_wall_strike": _safe_float(call_wall),
                "put_wall_strike": _safe_float(put_wall),
                "zero_gamma_strike": _safe_float(zg),
                "net_gex": float(row["net_gex"]),
                "call_gex_total": float(row["call_gex_total"]),
                "put_gex_total": float(row["put_gex_total"]),
                "summary": narrative,
            }
        )

    narrative_df = pd.DataFrame(narrative_rows)
    return details_df, summary_df, narrative_df


# ========= MAIN =========

if __name__ == "__main__":
    details, summary, narrative = run_scan(TICKERS)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", None)

    print("\n=== SUMMARY BY TICKER ===")
    print(summary.to_string(index=False))

    print("\n=== REGIME + ENGLISH SUMMARY ===")
    for s in narrative["summary"].tolist():
        print("-", s)

    details.to_csv("alpaca_dealer_details.csv", index=False)
    summary.to_csv("alpaca_dealer_summary.csv", index=False)
    narrative.to_csv("alpaca_dealer_narrative.csv", index=False)

    print("\nSaved: alpaca_dealer_details.csv, alpaca_dealer_summary.csv, alpaca_dealer_narrative.csv")
