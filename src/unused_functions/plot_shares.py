import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_candles_with_volatility(
    df: pd.DataFrame,
    actions: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
):
    """
    Plot candlestick-style daily prices with rolling volatility overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame with MultiIndex columns
        (ACTION, CURRENCY, METRIC)
    actions : list[str] | None
        Subset of ACTIONs to plot (default: all)
    start, end : pd.Timestamp | None
        Optional date filtering
    """

    if df.columns.nlevels != 3:
        raise ValueError("Expected 3-level MultiIndex columns")

    data = df.copy()

    # Date filtering
    if start is not None:
        data = data.loc[data.index >= pd.Timestamp(start)]
    if end is not None:
        data = data.loc[data.index <= pd.Timestamp(end)]

    all_actions = data.columns.get_level_values("ACTION").unique()

    if actions is None:
        actions = all_actions
    else:
        missing = set(actions) - set(all_actions)
        if missing:
            raise KeyError(f"Unknown ACTION(s): {missing}")

    for action in actions:
        sub = data[action]

        # Drop currency level (FX already normalised)
        if isinstance(sub.columns, pd.MultiIndex):
            sub = sub.droplevel("CURRENCY", axis=1)

        required = {"LOW", "HIGH", "CLOSE"}
        if not required.issubset(sub.columns):
            raise KeyError(f"{action} missing required metrics {required}")

        sub = sub.sort_index().dropna(how="all")

        if sub.empty:
            continue

        # OPEN fallback
        if "OPEN" not in sub.columns:
            sub["OPEN"] = sub["CLOSE"].shift(1)

        fig, ax_price = plt.subplots(figsize=(13, 6))

        x = np.arange(len(sub))
        width = 0.6

        # ---- Candlesticks ----
        for i, (_, row) in enumerate(sub.iterrows()):
            if pd.isna(row["OPEN"]) or pd.isna(row["CLOSE"]):
                continue

            color = "green" if row["CLOSE"] >= row["OPEN"] else "red"

            # Wick (LOW–HIGH)
            ax_price.vlines(
                x[i],
                row["LOW"],
                row["HIGH"],
                color=color,
                linewidth=1,
                zorder=1,
            )

            # Body (OPEN–CLOSE)
            ax_price.bar(
                x[i],
                row["CLOSE"] - row["OPEN"],
                width,
                bottom=row["OPEN"],
                color=color,
                alpha=0.7,
                zorder=2,
            )

        ax_price.set_ylabel("Price")
        ax_price.set_title(f"{action} – Candlesticks with rolling volatility")

        ax_price.set_xticks(x[:: max(len(x) // 10, 1)])
        ax_price.set_xticklabels(
            sub.index.strftime("%Y-%m-%d")[:: max(len(x) // 10, 1)],
            rotation=45,
            ha="right",
        )

        ax_price.grid(True, axis="y", alpha=0.3)

        # ---- Volatility overlay ----
        if "VOLATILITY" in sub.columns:
            ax_vol = ax_price.twinx()
            ax_vol.plot(
                x,
                sub["VOLATILITY"],
                color="blue",
                linestyle="--",
                linewidth=2,
                label="Rolling volatility",
            )
            ax_vol.set_ylabel("Volatility")
            ax_vol.grid(False)

            # Combined legend
            lines_1, labels_1 = ax_price.get_legend_handles_labels()
            lines_2, labels_2 = ax_vol.get_legend_handles_labels()
            ax_price.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="upper left",
            )

        plt.tight_layout()
        plt.show()