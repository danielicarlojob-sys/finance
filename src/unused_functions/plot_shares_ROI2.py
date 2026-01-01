"""
| Price Move | Volume | Interpretation                 |
| ---------- | ------ | ------------------------------ |
| Up         | High   | Strong, credible rally         |
| Up         | Low    | Weak or speculative move       |
| Down       | High   | Strong distribution / sell-off |
| Down       | Low    | Lack of conviction             |

Combined interpretation (price + volume)
-------------------------------------------

When you read both bar charts together:

    Green candle + tall green volume bar
    → strong bullish day with institutional participation

    Red candle + tall red volume bar
    → aggressive selling, possible trend continuation

    Large wick + small body + high volume
    → indecision, potential reversal

    Small body + low volume
    → noise, low informational content
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_candles_volatility_volume_roi(
    df: pd.DataFrame,
    actions: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    roi_target: float = 0.05,
    purchase_date: pd.Timestamp | None = None,
    volume_col: str = "VOLUME"
):
    """
    Plot daily candlestick prices with:
      - rolling volatility overlay
      - volume bars
      - purchase date marker (industry-standard)
      - ROI target marker (industry-standard)

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame: (ACTION, CURRENCY, METRIC)
    actions : list[str] | None
        Subset of instruments to plot (default all)
    start, end : pd.Timestamp | None
        Filter dates
    roi_target : float
        Target ROI (5% default)
    purchase_date : pd.Timestamp | None
        Date at which the asset is assumed to be purchased.
        If provided, ROI is evaluated relative to the CLOSE price on (or after)
        this date. If None, the first available datapoint is used.
    volume_col : str
        Column name for volume
    """
    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------
    if df.columns.nlevels != 3:
        raise ValueError("Expected 3-level MultiIndex columns")

    data = df.copy()

    # Optional date filtering
    if start:
        data = data.loc[data.index >= pd.Timestamp(start)]
    if end:
        data = data.loc[data.index <= pd.Timestamp(end)]

    # Resolve ACTIONs
    all_actions = data.columns.get_level_values("ACTION").unique()
    if actions is None:
        actions = all_actions
    else:
        missing = set(actions) - set(all_actions)
        if missing:
            raise KeyError(f"Unknown ACTION(s): {missing}")

    # ------------------------------------------------------------------
    # One plot per ACTION
    # ------------------------------------------------------------------
    for action in actions:
        sub = data[action]

        # Drop currency level if present
        if isinstance(sub.columns, pd.MultiIndex):
            sub = sub.droplevel("CURRENCY", axis=1)

        # Ensure minimum OHLC data
        required = {"LOW", "HIGH", "CLOSE"}
        if not required.issubset(sub.columns):
            raise KeyError(f"{action} missing required metrics {required}")

        sub = sub.sort_index().dropna(how="all")
        if sub.empty:
            continue

        # Synthesize OPEN if missing
        if "OPEN" not in sub.columns:
            sub["OPEN"] = sub["CLOSE"].shift(1)

        # X-axis index for bar-based plotting
        x = np.arange(len(sub))
        width = 0.6

        fig, ax_price = plt.subplots(figsize=(15, 7))

        # ------------------------------------------------------------------
        # Candlesticks
        # ------------------------------------------------------------------
        for i, (_, row) in enumerate(sub.iterrows()):
            if pd.isna(row["OPEN"]) or pd.isna(row["CLOSE"]):
                continue

            color = "green" if row["CLOSE"] >= row["OPEN"] else "red"

            # Wick
            ax_price.vlines(
                x[i],
                row["LOW"],
                row["HIGH"],
                color=color,
                linewidth=1,
                zorder=1
            )

            # Body
            ax_price.bar(
                x[i],
                row["CLOSE"] - row["OPEN"],
                width,
                bottom=row["OPEN"],
                color=color,
                alpha=0.7,
                zorder=2
            )

        # ------------------------------------------------------------------
        # Volume (secondary axis)
        # ------------------------------------------------------------------
        if volume_col in sub.columns:
            ax_vol = ax_price.twinx()

            vol_colors = [
                "green" if c >= o else "red"
                for o, c in zip(sub["OPEN"], sub["CLOSE"])
            ]

            ax_vol.bar(
                x,
                sub[volume_col],
                color=vol_colors,
                alpha=0.3,
                width=0.6,
                zorder=0
            )

            ax_vol.set_ylabel("Volume")
            ax_vol.set_ylim(0, sub[volume_col].max() * 3)

        # ------------------------------------------------------------------
        # Volatility overlay (third axis)
        # ------------------------------------------------------------------
        if "VOLATILITY" in sub.columns:
            ax_vol2 = ax_price.twinx()
            ax_vol2.plot(
                x,
                sub["VOLATILITY"],
                color="blue",
                linestyle="--",
                linewidth=2,
                zorder=3
            )
            ax_vol2.set_ylabel("Volatility")
            ax_vol2.grid(False)

        # ------------------------------------------------------------------
        # Resolve purchase index
        # ------------------------------------------------------------------
        if purchase_date is not None:
            purchase_date = pd.Timestamp(purchase_date)
            valid_dates = sub.index[sub.index >= purchase_date]
            purchase_idx = None if valid_dates.empty else sub.index.get_loc(valid_dates[0])
        else:
            purchase_idx = 0

        # ------------------------------------------------------------------
        # INDUSTRY-STANDARD ENTRY MARKER (BUY)
        # ------------------------------------------------------------------
        if purchase_idx is not None:
            buy_price = sub["CLOSE"].iloc[purchase_idx]

            ax_price.scatter(
                            x[purchase_idx],
                            buy_price,
                            marker="^",          # ▲ Buy marker
                            s=120,
                            color="red",
                            edgecolor="black",
                            linewidth=0.8,
                            zorder=11
                        )


            # Vertical entry line
            ax_price.axvline(
                x=x[purchase_idx],
                color="red",
                linestyle=":",
                linewidth=2,
                alpha=0.9,
                zorder=9
            )

            # Entry label near top of chart
            ax_price.text(
                x[purchase_idx],
                sub["HIGH"].max(),
                "BUY",
                color="black",
                fontsize=10,
                ha="center",
                va="bottom",
                zorder=10
            )

        # ------------------------------------------------------------------
        # INDUSTRY-STANDARD ROI MARKER (EXIT CONDITION)
        # ------------------------------------------------------------------
        if purchase_idx is not None:
            target_price = buy_price * (1 + roi_target)

            for i in range(purchase_idx, len(sub)):
                if sub["CLOSE"].iloc[i] >= target_price:
                    ax_price.scatter(
                        x[i],
                        sub["CLOSE"].iloc[i],
                        marker="v",          # ▼ Exit marker
                        s=140,
                        color="lime",
                        edgecolor="green",
                        linewidth=0.8,
                        zorder=11
                    )

                    ax_price.axvline(
                        x=x[i],
                        color="lime",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.9,
                        zorder=9
                    )

                    ax_price.text(
                        x[i],
                        sub["HIGH"].max(),
                        f"{roi_target * 100:.0f}% ROI",
                        color="green",
                        fontsize=10,
                        ha="center",
                        va="bottom",
                        zorder=10
                    )
                    break

        # ------------------------------------------------------------------
        # Formatting
        # ------------------------------------------------------------------
        step = max(len(x) // 10, 1)
        ax_price.set_xticks(x[::step])
        ax_price.set_xticklabels(
            sub.index.strftime("%Y-%m-%d")[::step],
            rotation=45,
            ha="right"
        )

        ax_price.set_ylabel("Price")
        ax_price.set_title(
            f"{action} – Candlestick + Volatility + Volume + Entry & ROI"
        )
        ax_price.grid(True, axis="y", alpha=0.3)
        price_min = sub["LOW"].min()
        price_max = sub["HIGH"].max()

        ax_price.set_ylim(price_min * 0.98, price_max * 1.02)


        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    df_shares2 = pd.read_csv('df_shares2.csv')
    print(df_shares2.head())
    try:
        # from src.plot_shares_ROI2 import plot_candles_volatility_volume_roi as ROI
        
        actions_list   = df_shares2.columns.get_level_values("ACTION").unique().to_list()
        print(actions_list)
        currencies_list = df_shares2.columns.get_level_values("CURRENCY").unique()
        metrics   = df_shares2.columns.get_level_values("METRIC").unique()

        plot_candles_volatility_volume_roi(
            df=df_shares2,
            actions=['RR.L_GBP→GBP'],
            start=df_shares2.index.min(),
            end=df_shares2.index.max(),
            purchase_date='2025-01-01',
            roi_target=0.55
        )
    except Exception as e:
        print(f"[FAILED] plot_candles_volatility_volume_roi {type(e).__name__}: {e} ")
