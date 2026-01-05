import os
import requests                      # HTTP client for API calls
import pandas as pd                  # Tabular data handling
import numpy as np
from datetime import datetime        # Datetime handling
from typing import Iterable, Optional, Dict, Union
import matplotlib.pyplot as plt
import yfinance as yf
from src.debug_print import debug_print
from src.unused_functions.plot_shares import plot_candles_with_volatility
from src.unused_functions.plot_shares_ROI import plot_candles_with_volatility_and_target as plt_vol_trg
from src.unused_functions.plot_shares_ROI2 import plot_candles_volatility_volume_roi as ROI
from src.fetch_lse_tickers import get_ftse100
from src.debug_print import debug_print
from src.utils.retry_decorator import log_exceptions_with_retry
from pprint import pprint as pp




def get_exchange_rates(
    base: str = "EUR",
    symbols: Optional[Iterable[str]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Retrieve exchange rates using Frankfurter (ECB) with explicit unit handling for GBp (pence).

    Definitions
    -----------
    - FX(base/symbol) means: 1 unit of `base` equals X units of `symbol`
    - GBp is a unit (1 GBP = 100 GBp), not a fiat currency

    Guarantees
    ----------
    - base == symbol            → FX = 1
    - GBP / GBp                 → 100
    - GBp / GBP                 → 0.01
    - GBP/EUR = 100 × GBp/EUR

    Notes
    -----
    - start and end define the date range (inclusive). If None, defaults to today.
    - Unit conversions (GBP <-> GBp) are computed without API calls.
    """
    import pandas as pd
    import requests

    # ---------------- currency normalization ----------------
    CURRENCY_MAP = {
        "GBP": {"iso": "GBP", "scale": 1.0},
        "GBp": {"iso": "GBP", "scale": 100.0},  # 1 GBP = 100 GBp
    }

    def normalize(ccy: str):
        entry = CURRENCY_MAP.get(ccy)
        if entry:
            return entry["iso"], entry["scale"]
        return ccy, 1.0

    base_iso, base_scale = normalize(base)
    symbols = list(symbols) if symbols else []
    symbol_meta = {s: normalize(s) for s in symbols}

    # ---------------- identity detection ----------------
    identity_rates = {}  # handled without API
    api_symbols = set()  # symbols to fetch from API

    for s, (iso, symbol_scale) in symbol_meta.items():
        if s == base:
            identity_rates[s] = 1.0
        elif iso == base_iso:
            # Unit-only conversion (GBP <-> GBp)
            identity_rates[s] = symbol_scale / base_scale
        else:
            api_symbols.add(iso)

    # ---------------- prepare API params ----------------
    params = {"from": base_iso}
    if api_symbols:
        params["to"] = ",".join(sorted(api_symbols))

    # ---------------- fetch helper ----------------
    def fetch(url: str) -> pd.DataFrame:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()["rates"]
        df = pd.DataFrame.from_dict(data, orient="index").sort_index().rename_axis("date")
        df.index = pd.to_datetime(df.index).normalize()
        return df

    # ---------------- fetch FX from API ----------------
    df_api = pd.DataFrame()
    if api_symbols:
        if start is None and end is None:
            df_api = fetch("https://api.frankfurter.app/latest")
        elif start and end is None:
            df_api = fetch(f"https://api.frankfurter.app/{start:%Y-%m-%d}")
        elif start and end:
            df_api = fetch(f"https://api.frankfurter.app/{start:%Y-%m-%d}..{end:%Y-%m-%d}")
        else:
            raise ValueError("Invalid start/end combination for FX retrieval")

    # ---------------- assemble output ----------------
    # Determine index for final DataFrame
    if not df_api.empty:
        idx = df_api.index
    else:
        # Only unit conversions → generate date range
        if start and end:
            idx = pd.date_range(start=start, end=end, freq="D")
        elif start:
            idx = pd.DatetimeIndex([start])
        else:
            idx = pd.DatetimeIndex([pd.Timestamp.today().normalize()])

    out = pd.DataFrame(index=idx)

    # Identity FX columns (unit-only conversions or base==symbol)
    for s, rate in identity_rates.items():
        out[f"{base}/{s}"] = rate

    # Non-identity FX columns (from API)
    for s, (iso, symbol_scale) in symbol_meta.items():
        if s in identity_rates:
            continue
        if iso not in df_api.columns:
            raise KeyError(f"Missing FX column from API: {iso}")
        series = df_api[iso] * (symbol_scale / base_scale)
        out[f"{base}/{s}"] = series

    # Metadata
    out.attrs["currency_type"] = {
        col: ("unit" if col.endswith("/GBp") else "fiat")
        for col in out.columns
    }

    return out

def get_share_prices_2_with_fundamentals(
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    base_currency: str = "GBP",
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Retrieve daily share prices (OHLCV) with FX-normalized price metrics,
    volatility, and fundamental metrics.

    FX conversion is applied ONLY to price-dimensioned metrics.
    """

    PRICE_METRICS = {"LOW", "HIGH", "CLOSE", "RANGE"}
    frames = []
    currency_meta = {}

    # 1. PRICE + FUNDAMENTALS INGESTION
    for ticker in tickers:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(start=start, end=end, interval="1d", auto_adjust=False)
        if hist.empty:
            continue

        currency = yf_ticker.fast_info.currency
        currency_meta[ticker] = currency

        df = hist[["Low", "High", "Close", "Volume"]].copy()
        # Convert to naive datetime index
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["RANGE"] = df["High"] - df["Low"]
        df["VOLATILITY"] = df["Close"].pct_change().rolling(vol_window).std()
        df.rename(columns={"Low": "LOW", "High": "HIGH", "Close": "CLOSE", "Volume": "VOLUME"}, inplace=True)
        df = df[["LOW", "HIGH", "CLOSE", "RANGE", "VOLATILITY", "VOLUME"]]

        # Add fundamental metrics
        info = yf_ticker.info
        df["EPS"] = info.get("trailingEps")
        df["BookValue"] = info.get("bookValue")
        df["Dividend"] = info.get("dividendRate")

        df.columns = pd.MultiIndex.from_product(
            [[ticker], [currency], df.columns],
            names=["ACTION", "CURRENCY", "METRIC"]
        )
        frames.append(df)

    if not frames:
        raise RuntimeError("No share price data retrieved")

    out = pd.concat(frames, axis=1).sort_index()
    out.attrs["currency"] = currency_meta
    out.attrs["base_currency"] = base_currency
    out.attrs["vol_window"] = vol_window

    # ----------------------------------------
    print(f"{debug_print()} df:\n{df}")
    print(f"{debug_print()} out:\n{out}")
    # ----------------------------------------
    
    # 2. FX RETRIEVAL
    currencies = sorted(set(currency_meta.values()))
    fx = get_exchange_rates(base=base_currency, symbols=currencies, start=start, end=end)
    print(f"{debug_print()} → FX returned:\n{fx.head()}")

    # 3 + 4. FX ALIGNMENT + CONVERSION
    out_converted = out.copy()
    fx = fx.copy()
    # Always rename index to "DATE" before reset, regardless of current name
    fx.index.rename("DATE", inplace=True)
    fx_reset = fx.reset_index()
    fx_reset["DATE"] = pd.to_datetime(fx_reset["DATE"]).dt.tz_localize(None)



    currency_idx = out.columns.names.index("CURRENCY")
    metric_idx = out.columns.names.index("METRIC")
    action_idx = out.columns.names.index("ACTION")

    for col in out.columns:
        col_currency = col[currency_idx]
        metric = col[metric_idx]

        if metric not in PRICE_METRICS or col_currency == base_currency:
            continue

        fx_col = f"{base_currency}/{col_currency}"
        if fx_col not in fx_reset.columns:
            raise KeyError(f"Missing FX rate: {fx_col}")

        temp = pd.DataFrame({
            "DATE": out.index.tz_localize(None),  # ensure naive datetime
            "PRICE": out[col].values
        })
        temp = pd.merge_asof(
            temp.sort_values("DATE"),
            fx_reset[["DATE", fx_col]].sort_values("DATE"),
            on="DATE",
            direction="backward"
        )
        out_converted[col] = temp["PRICE"].values / temp[fx_col].values

    # 5. RELABEL COLUMNS (POST-FX)
    new_columns = []
    for col in out_converted.columns:
        action = col[action_idx]
        orig_ccy = col[currency_idx]
        metric = col[metric_idx]
        new_action = f"{action}_{orig_ccy}→{base_currency}"
        new_columns.append((new_action, base_currency, metric))

    out_converted.columns = pd.MultiIndex.from_tuples(
        new_columns, names=["ACTION", "CURRENCY", "METRIC"]
    )

    return out_converted

# =====================================================
# SCRIPT MANAUL TEST - ENTRY POINT
# =====================================================

if __name__ == "__main__":



    base_currency = "GBP"
    target_currencies = ["USD", "GBP", "EUR", "JPY"]
    cryptos = ["BTC", "ETH"]
    shares = ['RR.L', 'AAPL'] #['AAPL', 'RR.L', 'MSFT', 'NVDA', 'LDO.MI','4816.T']

    start_date = datetime(2025, 12, 1)
    end_date = datetime(2026, 1, 5)#pd.Timestamp.today().normalize() #- pd.Timedelta(days=1)
    # end_date = datetime(2026, 1, 1)
    try:
        fx = get_exchange_rates(base='GBP', symbols=['GBp'], start=start_date, end=end_date)
        print("fx", fx)
    except Exception as e:
        print(f"{debug_print()} could not run exchange_rates {type(e).__name__};{e}")



    try:
        df_shares2 = get_share_prices_2_with_fundamentals(
        tickers=shares, #shares_lse,
        start=start_date,
        end=end_date,
        base_currency = base_currency,
        vol_window = 20,
    )

        print(f"{debug_print()} get_share_prices_2:\n{df_shares2}")
    except Exception as e:
        print(f"{debug_print()} [FAILED] running get_share_prices_2 {type(e).__name__}: {e} ")
   