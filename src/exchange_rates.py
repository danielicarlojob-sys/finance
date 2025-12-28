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

    return combined


def plot_fx_timeseries(
    df: pd.DataFrame,
    title: str | None = None,
    linewidth: float = 1.0,
    marker: str = "o",
    markersize: float = 3,
) -> None:
    """
    Plot multiple FX time series on a single axis with consistent styling.

    - Same line width for all series
    - Same marker and marker size
    - Different colors per column (automatic)
    - Legend enabled

    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame with numeric columns
    title : str | None
        Optional plot title
    linewidth : float
        Line width for all series
    marker : str
        Marker style for all series
    markersize : float
        Marker size for all series
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    fig, ax = plt.subplots(figsize=(10, 5))

    for col in df.columns:
        ax.plot(
            df.index,
            df[col],
            label=col,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
        )

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if title:
        ax.set_title(title)

    ax.set_xlabel("Date")
    ax.set_ylabel("Exchange Rate")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_currency = "GBP"
    target_currencies = ["USD", "EUR", "JPY"]
    cryptos = ["BTC", "ETH"]
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2025, 12, 29)



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
    
    plot_fx_timeseries(
    df,
    title="EUR FX Rates",
    linewidth=0.8,
    marker=".",
    markersize=2
)

