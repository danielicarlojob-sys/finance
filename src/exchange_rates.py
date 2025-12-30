import os
import requests                      # HTTP client for API calls
import pandas as pd                  # Tabular data handling
from datetime import datetime        # Datetime handling
from typing import Iterable, Optional, Dict, Union
import matplotlib.pyplot as plt
import yfinance as yf
from src.debug_print import debug_print



def get_exchange_rates(
    base: str = "EUR",
    symbols: Optional[Iterable[str]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Union[Dict[str, float], pd.DataFrame]:
    """
    Retrieve exchange rates using Frankfurter (ECB).

    Cases
    -----
    - No dates provided:
        Returns latest exchange rates
    - One date provided (start only):
        Returns exchange rates for that date
    - Two dates provided:
        Returns time series between dates (inclusive)

    Parameters
    ----------
    base : str
        Base currency
    symbols : Iterable[str] | None
        Target currencies (None = all)
    start : datetime | None
        Single date or range start
    end : datetime | None
        Range end

    Returns
    -------
    dict[str, float] or pd.DataFrame
    """
    params = {"from": base}

    if symbols:
        params["to"] = ",".join(symbols)

    # ---------------- latest ----------------
    if start is None and end is None:
        url = "https://api.frankfurter.app/latest"

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()["rates"]
        output = {}
        for key, value in results.items():
            temp_key = f"{base}/{key}"
            output[temp_key] = float(value)
        
        output = pd.DataFrame([output])
        output.attrs["currency_type"] = {col: "fiat" for col in output.keys()}
        return output

    # ---------------- single date ----------------
    if start and end is None:
        date = start.strftime("%Y-%m-%d")
        url = f"https://api.frankfurter.app/{date}"

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()["rates"]
        output = {}
        for key, value in results.items():
            temp_key = f"{base}/{key}"
            output[temp_key] = float(value)
        output = pd.DataFrame([output])
        output.attrs["currency_type"] = {col: "fiat" for col in output.keys()}

        return output

    # ---------------- date range ----------------
    if start and end:
        start_s = start.strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")
        url = f"https://api.frankfurter.app/{start_s}..{end_s}"

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()["rates"]
        

        df = (
            pd.DataFrame.from_dict(data, orient="index")
            .sort_index()
            .rename_axis("date")
        )
        # Rename columns to match EUR/USD style
        df.columns = [f"{base}/{c}" for c in df.columns]

        df.index = pd.to_datetime(df.index)
        df.attrs["currency_type"] = {col: "fiat" for col in df.columns}

        return df

    raise ValueError("Invalid date combination")


def get_crypto_prices(
    symbols: Iterable[str],
    vs_currency: str = "EUR",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Retrieve cryptocurrency prices using Yahoo Finance.

    Behaviour
    ----------
    - If no dates are provided:
        Returns the latest available closing price for each symbol.
    - If only start_date is provided:
        Returns the closing price on (or nearest after) that date.
    - If start_date and end_date are provided:
        Returns daily closing prices between the two dates (inclusive).

    Parameters
    ----------
    symbols : Iterable[str]
        Crypto symbols (e.g. ['BTC', 'ETH']).
    vs_currency : str, optional
        Fiat quote currency (default: 'EUR').
    start_date : datetime | None, optional
        Start date for historical retrieval.
    end_date : datetime | None, optional
        End date for historical retrieval.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with columns like EUR/BTC.
    """

    # Normalize inputs
    symbols = [s.upper() for s in symbols]
    vs_currency = vs_currency.upper()

    # Yahoo Finance tickers use the format BTC-EUR, ETH-USD, etc.
    tickers = {symbol: f"{symbol}-{vs_currency}" for symbol in symbols}

    # ---------------- latest prices ----------------
    if start_date is None and end_date is None:
        prices = {}
        today = pd.Timestamp.today().normalize()

        for symbol, ticker in tickers.items():
            data = yf.download(
                ticker,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                continue

            latest_price = data["Close"].iloc[-1]
            prices[f"{vs_currency}/{symbol}"] = float(latest_price)
            df = pd.DataFrame([prices], index=[today])
            # If yfinance returned a MultiIndex, flatten it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                df.columns.name = None
        df.attrs["currency_type"] = {col: "crypto" for col in df.columns}

        return df

    # Normalize dates
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() if end_date else start

    series = []

    # ---------------- historical prices ----------------
    for symbol, ticker in tickers.items():
        data = yf.download(
            ticker,
            start=start,
            end=end + pd.Timedelta(days=1),  # Yahoo end is exclusive
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            continue

        df_coin = data[["Close"]].rename(
            columns={"Close": f"{vs_currency}/{symbol}"}
        )

        df_coin.index = pd.to_datetime(df_coin.index).normalize()
        series.append(df_coin)

    if not series:
        raise RuntimeError("No crypto price data retrieved")

    # Align all crypto series on date index
    df = pd.concat(series, axis=1).sort_index()


    # If yfinance returned a MultiIndex, flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df.columns.name = None
    df.attrs["currency_type"] = {col: "crypto" for col in df.columns}

    return df


def get_fx_and_crypto(
    base: str,
    fiat_symbols: Iterable[str],
    crypto_symbols: Iterable[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Retrieve combined fiat FX and crypto prices.

    Fiat FX:
        - ECB daily reference rates
    Crypto:
        - CoinGecko latest snapshot

    Parameters
    ----------
    base : str
        Base fiat currency
    fiat_symbols : Iterable[str]
        Fiat currencies (e.g. USD, GBP)
    crypto_symbols : Iterable[str]
        Crypto coin IDs (e.g. bitcoin)
    start : datetime | None
        FX start date
    end : datetime | None
        FX end date

    Returns
    -------
    pd.DataFrame
        Combined DataFrame aligned on date index
    """

    # Retrieve fiat FX data
    fx_df = get_exchange_rates(
        base=base,
        symbols=fiat_symbols,
        start=start,
        end=end
    )

    # Retrieve crypto prices (latest only)
    try:
        crypto_df = get_crypto_prices(
            symbols=crypto_symbols,
            vs_currency=base,
            start_date=start,
            end_date=end
        )
    except Exception as e:
        print(f"{debug_print()} error extracting cryptos xchange rates {type(e).__name__}: {e}")


    

    # Reindex crypto prices to match FX dates (forward-fill)
    crypto_df = crypto_df.reindex(fx_df.index, method="ffill")

    # Concatenate FX and crypto along columns
    combined = pd.concat([fx_df, crypto_df], axis=1)

    # Assign attributes from data source
    combined.attrs["currency_type"] = {
    **{col: "fiat" for col in fx_df.columns},
    **{col: "crypto" for col in crypto_df.columns},
}


    return combined

def get_share_prices(
    tickers: Iterable[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Retrieve share prices using Yahoo Finance.

    Cases
    -----
    - No dates provided:
        Returns latest available closing prices
    - Only start date:
        Returns closing prices on or after that date
    - Start and end dates:
        Returns daily closing prices in the date range (inclusive)

    Parameters
    ----------
    tickers : Iterable[str]
        Equity tickers (e.g. ['AAPL', 'RR.L', 'MSFT'])
    start : datetime | None
        Single date or range start
    end : datetime | None
        Range end

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame of closing prices.
        Quote currency per ticker stored in df.attrs["currency"].
    """
    tickers = [t.upper() for t in tickers]
    today = pd.Timestamp.today().normalize()

    currency_meta = {}  # ticker -> currency
    series_dict = {}    # ticker -> pd.Series of closing prices

    # ---------- latest prices ----------
    if start is None and end is None:
        for ticker in tickers:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="5d", auto_adjust=True)

            if hist.empty:
                continue

            # Use explicit name to avoid rename issues
            s = hist["Close"].copy()
            s.name = ticker
            s.index = pd.to_datetime(s.index).normalize()
            series_dict[ticker] = s

            currency_meta[ticker] = yf_ticker.fast_info.get("currency").upper()



        if not series_dict:
            raise RuntimeError("No share price data retrieved")

        df = pd.concat(series_dict.values(), axis=1).sort_index()
        df.attrs["currency"] = currency_meta
        second_level = [s.upper() for s in list(currency_meta.values())]


        df.columns = pd.MultiIndex.from_arrays(
                                                [df.columns, second_level],
                                                names=["ACTION", "CURRENCY"],
                                                )
        return df

    # ---------- historical prices ----------
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize() if end else start_ts

    for ticker in tickers:
        yf_ticker = yf.Ticker(ticker)
        hist = yf.download(
            ticker,
            start=start_ts,
            end=end_ts + pd.Timedelta(days=1),  # Yahoo end exclusive
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if hist.empty:
            continue

        # Copy & name explicitly
        s = hist["Close"].copy()
        s.name = ticker
        s.index = pd.to_datetime(s.index).normalize()
        series_dict[ticker] = s

        currency_meta[ticker] = yf_ticker.fast_info.get("currency")


    if not series_dict:
        raise RuntimeError("No share price data retrieved")

    df = pd.concat(series_dict.values(), axis=1).sort_index()
    df.attrs["currency"] = currency_meta

    second_level = [s.upper() for s in list(currency_meta.values())]

    df.columns = pd.MultiIndex.from_arrays(
                                            [df.columns, second_level],
                                            names=["ACTION", "CURRENCY"],
                                            )
    return df

def get_share_prices_2(
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    base_currency: str = "GBP",
    fx_rates: pd.DataFrame | None = None,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Retrieve daily share prices with FX-normalised min/max and volatility.

    Parameters
    ----------
    tickers : Iterable[str]
        Equity tickers
    start, end : datetime
        Date range (inclusive)
    base_currency : str
        Target currency for normalisation
    fx_rates : pd.DataFrame | None
        FX rates indexed by Date, columns like 'USD/GBP'
    vol_window : int
        Rolling window for volatility (in trading days)

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with MultiIndex columns:
            (ACTION, METRIC)
    """
    tickers = [t.upper() for t in tickers]

    frames = []
    currency_meta = {}

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)

    for ticker in tickers:
        yf_ticker = yf.Ticker(ticker)

        hist = yf.download(
            ticker,
            start=start_ts,
            end=end_ts,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        # try:
        #     print(f"{debug_print()} [hist]:\n{hist}")
        # except Exception as e:
        #     print(f"{debug_print()} [FAILED] printing hist {type(e).__name__}: {e}")
        if hist.empty:
            continue

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        currency = yf_ticker.fast_info.get("currency").upper()
        # try:
        #     print(f"{debug_print()} [currency]:\n{currency}")
        # except Exception as e:
        #     print(f"{debug_print()} [FAILED] printing currency {type(e).__name__}: {e}")

        if currency is None:
            continue

        currency_meta[ticker] = currency.upper()

        df = hist[["Close", "Low", "High", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).normalize()
    # =========================================================================
    # SHARE VALUES EXTRACTED
    # =========================================================================
        """
        print(f"{debug_print()} currency_meta: {currency_meta}")
        try:
            # Get unique currencies from prices
            currecies_list = [s.upper() for s in list(set(df.attrs["currency"].values()))]
            print(f"{debug_print()} [currecies_list]:\n{currecies_list}")
        except Exception as e:
            print(f"{debug_print()} [FAILED] extracting currecies_list from df {type(e).__name__}: {e}")
        # Get exchange rates for currency_list
        try:
            fx_inner = get_exchange_rates(
                                base=base_currency,
                                symbols=currecies_list,
                                start=df.index.min(),
                                end=df.index.max(),
                                )
            print(f"{debug_print()} fx_inner: {fx_inner}")
        except Exception as e:
            print(f"{debug_print()} [FAILED] could not get fx_inner = get_exchange_rates {type(e).__name__}: {e}")
        """


        # ---------- FX normalisation ----------
        """
        try:
            if currency != base_currency:
                if fx_rates is None:
                    raise ValueError("FX rates required for currency normalisation")

                # pair = f"{currency}/{base_currency}"
                pair = f"{base_currency}/{currency}"
                if pair not in fx_rates.columns:
                    raise KeyError(f"Missing FX rate: {pair}")

                fx = fx_rates[pair].reindex(df.index).ffill()
                df = df.mul(fx, axis=0)
        except Exception as e:
            print(f"{debug_print()} [FAILED] running FX normalization {type(e).__name__}: {e}")
        """
        try:
            # ---------- derived metrics ----------
            df["RANGE"] = df["High"] - df["Low"]

            returns = df["Close"].pct_change()
            df["VOLATILITY"] = returns.rolling(vol_window).std()

            df = df.rename(
                columns={
                    "Close": "CLOSE",
                    "Low": "LOW",
                    "High": "HIGH",
                    "Volume": "VOLUME",
                }
            )

            try:
                df = df[["LOW", "HIGH", "CLOSE","RANGE", "VOLATILITY", "VOLUME"]]
            except:
                print(f"{debug_print()} [FAILED] reordering df's columns {type(e).__name__}: {e}")

            df.columns = pd.MultiIndex.from_product(
                [[ticker], [currency_meta[ticker]], df.columns],
                names=["ACTION", "CURRENCY", "METRIC"],
            )

            frames.append(df)
        except Exception as e:
            print(f"{debug_print()} [FAILED] running Derived metrics {type(e).__name__}: {e}")

    if not frames:
        raise RuntimeError("No share price data retrieved")

    out_temp = pd.concat(frames, axis=1).sort_index()
    out_temp.attrs["currency"] = currency_meta
    out_temp.attrs["base_currency"] = base_currency
    out_temp.attrs["vol_window"] = vol_window
    # ===========================================================
    # WORKS UNTIL THIS POINT
    # ===========================================================


    try:
        fx = get_exchange_rates(
                            base=base_currency,
                            symbols=currency_meta,
                            start=start,
                            end=end,
                            )
    except Exception as e:
        print(f"{debug_print()} [FAILED] could not get fx = get_exchange_rates {type(e).__name__}: {e}")

    # Align FX dates
    fx = fx.reindex(out_temp.index).ffill()
    out = out_temp.copy()

    currency_idx = out_temp.columns.names.index("CURRENCY")
    action_idx = out_temp.columns.names.index("ACTION")
    metric_idx = out_temp.columns.names.index("METRIC")

    # ---------- numeric conversion ----------
    for col in out.columns:
        currency = col[currency_idx]

        if currency == base_currency:
            continue

        fx_col = f"{base_currency}/{currency}"
        if fx_col not in fx.columns:
            raise KeyError(f"Missing FX rate: {fx_col}")

        out[col] = prices[col] / fx[fx_col]

    # ---------- relabel columns ----------
    new_columns = []

    for col in out.columns:
        action = col[action_idx]
        orig_ccy = col[currency_idx]

        new_action = f"{action}_{orig_ccy}→{base_currency}"

        new_columns.append(
            (new_action, base_currency)
        )

    out.columns = pd.MultiIndex.from_tuples(
        new_columns,
        names=["ACTION", "CURRENCY", "METRIC"]
    )
    return out


def convert_prices_to_base_currency(
    prices: pd.DataFrame,
    base_currency: str = "GBP",
    currency_level: str = "CURRENCY",
    action_level: str = "ACTION",
) -> pd.DataFrame:
    """
    Convert a MultiIndex-column price DataFrame to a single base currency
    and relabel columns accordingly.
    """

    if not isinstance(prices.columns, pd.MultiIndex):
        raise TypeError("prices must have MultiIndex columns")
    
    # Get unique currencies from prices
    currecies_list = [s.upper() for s in list(set(prices.attrs["currency"].values()))]

    # Get exchange rates for currency_list
    try:
        fx = get_exchange_rates(
                            base=base_currency,
                            symbols=currecies_list,
                            start=prices.index.min(),
                            end=prices.index.max(),
                            )
        print(f"{debug_print()} fx: {fx}")
    except Exception as e:
        print(f"{debug_print()} [FAILED] could not get fx = get_exchange_rates {type(e).__name__}: {e}")



    # Align FX dates
    fx = fx.reindex(prices.index).ffill()

    out = prices.copy()

    currency_idx = prices.columns.names.index(currency_level)
    action_idx = prices.columns.names.index(action_level)

    # ---------- numeric conversion ----------
    for col in prices.columns:
        currency = col[currency_idx]

        if currency == base_currency:
            continue

        fx_col = f"{base_currency}/{currency}"
        if fx_col not in fx.columns:
            raise KeyError(f"Missing FX rate: {fx_col}")

        out[col] = prices[col] / fx[fx_col]

    # ---------- relabel columns ----------
    new_columns = []

    for col in out.columns:
        action = col[action_idx]
        orig_ccy = col[currency_idx]

        new_action = f"{action}_{orig_ccy}→{base_currency}"

        new_columns.append(
            (new_action, base_currency)
        )

    out.columns = pd.MultiIndex.from_tuples(
        new_columns,
        names=[action_level, currency_level]
    )

    return out


def plot_fx_timeseries(
    df: pd.DataFrame,
    title: str | None = None,
    linewidth: float = 1.0,
    marker: str = "o",
    markersize: float = 3,
    padding_frac: float = 0.05,
) -> None:
    """
    Plot FX and crypto time series with:
    - Fiat on primary y-axis
    - Crypto on secondary y-axis
    - Unique color per data series
    - Auto-scaled y-limits per axis
    - Legend annotated with FIAT / CRYPTO / N/A
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    currency_meta = df.attrs.get("currency_type", {})

    fig, ax_fiat = plt.subplots(figsize=(10, 5))
    ax_crypto = None

    # ---------- global color mapping ----------
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        col: color_cycle[i % len(color_cycle)]
        for i, col in enumerate(df.columns)
    }

    fiat_handles = []
    crypto_handles = []

    fiat_values = []
    crypto_values = []

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        ctype = currency_meta.get(col)
        label_suffix = (
            f" - {ctype.upper()}" if isinstance(ctype, str) else " - N/A"
        )

        color = colors[col]

        if ctype == "crypto":
            if ax_crypto is None:
                ax_crypto = ax_fiat.twinx()

            line, = ax_crypto.plot(
                series.index,
                series.values,
                color=color,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                label=f"{col}{label_suffix}",
            )

            crypto_handles.append(line)
            crypto_values.append(series.values)

        else:
            line, = ax_fiat.plot(
                series.index,
                series.values,
                color=color,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                label=f"{col}{label_suffix}",
            )

            fiat_handles.append(line)
            fiat_values.append(series.values)

    # ---------- auto-scale y-axes ----------

    def _autoscale(ax, values):
        if not values:
            return

        data = pd.concat(
            [pd.Series(v) for v in values], ignore_index=True
        )

        ymin = data.min()
        ymax = data.max()

        if ymin == ymax:
            pad = abs(ymin) * padding_frac if ymin != 0 else 1.0
        else:
            pad = (ymax - ymin) * padding_frac

        ax.set_ylim(ymin - pad, ymax + pad)

    _autoscale(ax_fiat, fiat_values)

    if ax_crypto is not None:
        _autoscale(ax_crypto, crypto_values)

    # ---------- labels, grid, legend ----------

    ax_fiat.set_xlabel("Date")
    ax_fiat.set_ylabel("Fiat Exchange Rate")

    if ax_crypto is not None:
        ax_crypto.set_ylabel("Crypto Exchange Rate")

    ax_fiat.grid(True, linestyle="--", alpha=0.4)

    handles = fiat_handles + crypto_handles
    labels = [h.get_label() for h in handles]

    if handles:
        ax_fiat.legend(handles, labels)

    if title:
        ax_fiat.set_title(title)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_currency = "GBP"
    target_currencies = ["USD", "GBP", "EUR", "JPY"]
    cryptos = ["BTC", "ETH"]
    shares = ['AAPL', 'RR.L', 'MSFT', 'NVDA', 'LDO.MI','4816.T']

    start_date = datetime(2024, 6, 1)
    end_date = datetime(2025, 12, 29)

    OLD_PROCESS = False
    if OLD_PROCESS == True:
        try:
            df_crypto = get_crypto_prices(
                                symbols=cryptos,
                                vs_currency=base_currency,
                                start_date=start_date,
                                end_date=end_date
                                )
            
        except Exception as e:
            print(f"{debug_print()} error extracting cryptos xchange rates {type(e).__name__}: {e}") 

        
        rates = get_exchange_rates("EUR", target_currencies)
        print(f"latest rates type: {type(rates)}")
        print("latest rates:", rates)

        rates = get_exchange_rates(
                                    base=base_currency,
                                    symbols=target_currencies,
                                    start=start_date
                                    )
        print(f"rates from {start_date} type: {type(rates)}")
        print(f"rates from {start_date}: {rates}")
        df = get_exchange_rates(
                                    base=base_currency,
                                    symbols=target_currencies,
                                    start=start_date,
                                    end=end_date
                                )
        print(f"rates between {start_date} and {end_date} type: {type(df)}")
        print(f"rates between {start_date} and {end_date}:\n{df}")
        df = get_fx_and_crypto(
        base=base_currency,
        fiat_symbols=target_currencies,
        crypto_symbols=cryptos,
        start=start_date,
        end=end_date,
                                )
        print(f"rates with cryptos between {start_date} and {end_date} type: {type(df)}")
        print(f"rates with cryptos between {start_date} and {end_date}:\n{df}")
        print(f"df attributes:\n{df.attrs["currency_type"]}")

        
        plot_fx_timeseries(
        df,
        title="EUR FX Rates",
        linewidth=0.8,
        marker=".",
        markersize=2
    )
    # --------------------------------
    # END   OLD_PROCESS
    # --------------------------------
    rates = get_exchange_rates(
                        base=base_currency,
                        symbols=target_currencies,
                        start=start_date,
                        end=end_date,
                        )
    # print(f"rates:\n{rates}")
    try:
        df_shares2 = get_share_prices_2(
        tickers=shares,
        start=start_date,
        end=end_date,
        base_currency = base_currency,
        fx_rates = rates,
        vol_window = 20,
    )
        """
        out.attrs["currency"] = currency_meta
        out.attrs["base_currency"] = base_currency
        out.attrs["vol_window"] = vol_window
        """
        print(f"{debug_print()} get_share_prices_2 type:\n{type(df_shares2)}")
        print(f"{debug_print()} get_share_prices_2:\n{df_shares2}")
    except Exception as e:
        print(f"{debug_print()} [FAILED] running get_share_prices_2 {type(e).__name__}: {e} ")
    try:
        currency_meta_2 = df_shares2.attrs["currency"]
        base_currency_2 = df_shares2.attrs["base_currency"]
        vol_window_2 = df_shares2.attrs["vol_window"]
        print(f"{debug_print()} currency_meta_2 type:\n{type(currency_meta_2)}")
        print(f"{debug_print()} currency_meta_2 :\n{currency_meta_2}")        
        
        print(f"{debug_print()} base_currency_2 type:\n{type(base_currency_2)}")
        print(f"{debug_print()} base_currency_2 :\n{base_currency_2}")

        print(f"{debug_print()} vol_window_2 type:\n{type(vol_window_2)}")
        print(f"{debug_print()} vol_window_2 :\n{vol_window_2}")

        # Get unique currencies from prices
        currecies_list = [s.upper() for s in list(set(currency_meta_2.values()))]
        print(f"{debug_print()} currecies_list type:\n{type(currecies_list)}")
        print(f"{debug_print()} currecies_list :\n{currecies_list}")



    except Exception as e:
        print(f"{debug_print()} [FAILED] running get_share_prices_2 {type(e).__name__}: {e} ")
        # Get exchange rates for currency_list
    try:
        # currency_meta_2 = df_shares2.attrs["currency"]
        # currecies_list = [s.upper() for s in list(set(currency_meta_2.values()))]
        fx_2 = get_exchange_rates(
                            base=base_currency,
                            symbols=currecies_list,
                            start=start_date,
                            end=end_date,
                            )
        print(f"{debug_print()} fx_2: {fx_2}")
    except Exception as e:
        print(f"{debug_print()} [FAILED] could not get fx_2 = get_exchange_rates {type(e).__name__}: {e}")


    
    if OLD_PROCESS == True:
        df_shares = get_share_prices(
        tickers=shares,
        start=start_date,
        end=end_date,
    )
        print(f"df_shares.attr:\n{df_shares.attrs}")
        print(f"df_shares:\n{df_shares}")


        
        try:
            df_converted = convert_prices_to_base_currency(
            prices= df_shares,
            base_currency = "GBP",
            currency_level = "CURRENCY",
        )
            print(f"{debug_print()} df_convereted:\n{df_converted}")
            
        except Exception as e:
            print(f"{debug_print()} ERROR on convert_prices_to_base_currency {type(e).__name__}: {e}") 
        
        plot_fx_timeseries(
            df_converted,
            title="EUR FX Rates",
            linewidth=0.8,
            marker=".",
            markersize=2
        )
        
        


