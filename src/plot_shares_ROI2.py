import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_candles_volatility_volume_roi(
    df: pd.DataFrame,
    actions: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    roi_target: float = 0.05,
    volume_col: str = "VOLUME"
):
    """
    Plot daily candlestick prices with:
      - rolling volatility overlay
      - volume bars
      - ROI target markers

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
    volume_col : str
        Column name for volume
    """
    if df.columns.nlevels != 3:
        raise ValueError("Expected 3-level MultiIndex columns")

    data = df.copy()
    if start: data = data.loc[data.index >= pd.Timestamp(start)]
    if end: data = data.loc[data.index <= pd.Timestamp(end)]

    all_actions = data.columns.get_level_values("ACTION").unique()
    if actions is None:
        actions = all_actions
    else:
        missing = set(actions) - set(all_actions)
        if missing:
            raise KeyError(f"Unknown ACTION(s): {missing}")

    for action in actions:
        sub = data[action]
        if isinstance(sub.columns, pd.MultiIndex):
            sub = sub.droplevel("CURRENCY", axis=1)

        required = {"LOW", "HIGH", "CLOSE"}
        if not required.issubset(sub.columns):
            raise KeyError(f"{action} missing required metrics {required}")

        sub = sub.sort_index().dropna(how="all")
        if sub.empty:
            continue

        if "OPEN" not in sub.columns:
            sub["OPEN"] = sub["CLOSE"].shift(1)

        x = np.arange(len(sub))
        width = 0.6

        fig, ax_price = plt.subplots(figsize=(15, 7))

        # --- Candlestick ---
        for i, (_, row) in enumerate(sub.iterrows()):
            if pd.isna(row["OPEN"]) or pd.isna(row["CLOSE"]):
                continue

            color = "green" if row["CLOSE"] >= row["OPEN"] else "red"

            # Wick
            ax_price.vlines(x[i], row["LOW"], row["HIGH"], color=color, linewidth=1, zorder=1)

            # Body
            ax_price.bar(x[i], row["CLOSE"] - row["OPEN"], width, bottom=row["OPEN"],
                         color=color, alpha=0.7, zorder=2)

        # --- Volume bars ---
        if volume_col in sub.columns:
            ax_vol = ax_price.twinx()
            vol_colors = ['green' if c >= o else 'red' for o, c in zip(sub["OPEN"], sub["CLOSE"])]
            ax_vol.bar(x, sub[volume_col], color=vol_colors, alpha=0.3, width=0.6, zorder=0)
            ax_vol.set_ylabel("Volume")
            ax_vol.set_ylim(0, sub[volume_col].max()*3)  # scaled to show candles clearly

        # --- Volatility overlay ---
        if "VOLATILITY" in sub.columns:
            ax_vol2 = ax_price.twinx()
            ax_vol2.plot(x, sub["VOLATILITY"], color="blue", linestyle="--", linewidth=2, label="Volatility")
            ax_vol2.set_ylabel("Volatility")
            ax_vol2.grid(False)
            ax_vol2.set_zorder(3)

        # --- ROI target ---
        buy_price = sub["CLOSE"].iloc[0]
        target_price = buy_price * (1 + roi_target)
        for i, (_, row) in enumerate(sub.iterrows()):
            if row["CLOSE"] >= target_price:
                ax_price.annotate(
                    f"{roi_target*100:.0f}% ROI",
                    xy=(x[i], target_price),
                    xytext=(x[i], target_price*1.01),
                    arrowprops=dict(facecolor='lime', shrink=0.05, width=1, headwidth=8),
                    color='green',
                    fontsize=9,
                    ha='center'
                )

        # --- Formatting ---
        ax_price.set_xticks(x[:: max(len(x)//10,1)])
        ax_price.set_xticklabels(sub.index.strftime("%Y-%m-%d")[:: max(len(x)//10,1)], rotation=45, ha="right")
        ax_price.set_ylabel("Price")
        ax_price.set_title(f"{action} – Candlestick + Volatility + Volume + ROI target")
        ax_price.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()
