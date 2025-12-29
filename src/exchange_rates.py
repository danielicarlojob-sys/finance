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

def convert_prices_to_base_currency(
    prices: pd.DataFrame,
    fx: pd.DataFrame,
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
    target_currencies = ["USD", "EUR", "JPY"]
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

    df_shares = get_share_prices(
    tickers=shares,
    start=start_date,
    end=end_date,
)
    print(f"df_shares.attr:\n{df_shares.attrs}")
    print(f"df_shares:\n{df_shares}")

    rates = get_exchange_rates(
                            base=base_currency,
                            symbols=target_currencies,
                            start=start_date,
                            end=end_date,
                            )
    print(f"rates:\n{rates}")
    try:
        df_converted = convert_prices_to_base_currency(
        prices= df_shares,
        fx = rates,
        base_currency = "GBP",
        currency_level = "CURRENCY",
    )
        print(f"{debug_print()} df_convereted:\n{df_converted}")
        
    except Exception as e:
        print(f"{debug_print()} ERROR on convert_prices_to_base_currency {type(e).__name__}: {e}") 
    
    


