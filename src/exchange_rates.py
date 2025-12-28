import requests
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Iterable, Optional, Union, Dict


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

    rates = get_exchange_rates("EUR", ["USD", "GBP"])
    print(f"latest rates type: {type(rates)}")
    print("latest rates:", rates)

    start_date = datetime(2024, 6, 1)
    end_date = datetime(2025, 12, 29)
    rates = get_exchange_rates(
                                base="EUR",
                                symbols=["USD", "GBP"],
                                start=start_date
                                )
    print(f"rates from {start_date} type: {type(rates)}")
    print(f"rates from {start_date}: {rates}")
    df = get_exchange_rates(
                                base="EUR",
                                symbols=["USD", "GBP"],
                                start=start_date,
                                end=end_date
                            )
    print(f"rates between {start_date} and {end_date} type: {type(df)}")
    print(f"rates between {start_date} and {end_date}:\n{df}")
    plot_fx_timeseries(
    df,
    title="EUR FX Rates",
    linewidth=0.8,
    marker=".",
    markersize=2
)

