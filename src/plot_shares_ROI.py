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
from pathlib import Path
import pandas as pd
import numpy as np
from src.debug_print import debug_print
from src.utils.retry_decorator import log_exceptions_with_retry

# ------------------------------------------------
# HELPER function
# ------------------------------------------------
def bind_axis_color(ax, color: str, label: str):
    """
    Bind y-axis label, ticks, and spine color to a given artist color.
    """
    ax.set_ylabel(label, color=color)
    ax.tick_params(axis="y", colors=color)
    ax.spines["right"].set_color(color)

def annotate_with_offset(ax, x, y, text, color, offset_idx=0):
    """
    Place annotation with a vertical offset to avoid overlap.
    offset_idx is an integer (0, 1, 2, ...)
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset = y_range * 0.03 * offset_idx  # 3% per level

    ax.text(
        x,
        y + offset,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
        color=color,
        zorder=12,
    )

# ------------------------------------------------
# MAIN function
# ------------------------------------------------
@log_exceptions_with_retry(
    max_retries=5,
    prefix_fn=debug_print,
    retry_delay=1.0,   # optional
)
def plot_candles_volatility_volume_roi(
    df: pd.DataFrame,
    actions: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    roi_target: float = 0.05,
    purchase_date: pd.Timestamp | None = None,
    volume_col: str = "VOLUME",
):
    """
    Plot candlestick price charts with volume, volatility overlay, and
    buy/ROI markers for one or more financial instruments.

    The function is designed for technical analysis and exploratory
    decision support. It combines price action, trading activity
    (volume), volatility regime, and a deterministic ROI trigger
    into a single visualisation.

    Args:
        df (pd.DataFrame):
            Input price data with a 3-level MultiIndex on columns:
            (ACTION, CURRENCY, METRIC). The index must be datetime-like.

        actions (list[str] | None):
            List of ACTION identifiers to plot. If None, all available
            ACTIONs in the DataFrame are plotted.

        start (pd.Timestamp | None):
            Optional start date for filtering the time series.

        end (pd.Timestamp | None):
            Optional end date for filtering the time series.

        roi_target (float):
            Target return on investment expressed as a fraction.
            Example: 0.05 corresponds to a 5% ROI.

        purchase_date (pd.Timestamp | None):
            Date at which the asset is assumed to be purchased.
            If provided, the first trading date on or after this
            timestamp is used as the entry point.
            If None, the first available data point is used.

        volume_col (str):
            Column name corresponding to traded volume.

    Raises:
        ValueError:
            If the input DataFrame does not have exactly three column levels.

        KeyError:
            If required OHLC metrics are missing for a given ACTION.

    Returns:
        None:
            The function produces Matplotlib figures as a side effect
            and does not return any objects.

    Notes:
        - Candlestick color encodes daily price direction.
        - Volume bars confirm or weaken price moves.
        - Volatility is plotted on a separate y-axis and is assumed
          to be dimensionless (e.g. rolling std of returns).
        - ROI is triggered using CLOSE prices only (no intraday logic).
    """

    # ------------------------------------------------------------------
    # Validate expected MultiIndex structure
    # ------------------------------------------------------------------
    if df.columns.nlevels != 3:
        raise ValueError("Expected 3-level MultiIndex columns")

    # Create a defensive copy to avoid mutating user data
    data = df.copy()

    # ------------------------------------------------------------------
    # Optional date filtering on the index
    # ------------------------------------------------------------------
    if start:
        data = data.loc[data.index >= pd.Timestamp(start)]
    if end:
        data = data.loc[data.index <= pd.Timestamp(end)]

    # ------------------------------------------------------------------
    # Resolve which ACTIONs will be plotted
    # ------------------------------------------------------------------
    all_actions = data.columns.get_level_values("ACTION").unique()
    actions = all_actions if actions is None else actions

    # ------------------------------------------------------------------
    # Iterate over each selected ACTION independently
    # ------------------------------------------------------------------
    for action in actions:
        # Extract data for the current ACTION
        sub = data[action]

        # Drop currency level if still present
        if isinstance(sub.columns, pd.MultiIndex):
            sub = sub.droplevel("CURRENCY", axis=1)

        # Ensure minimum required OHLC metrics exist
        required = {"LOW", "HIGH", "CLOSE"}
        if not required.issubset(sub.columns):
            raise KeyError(f"{action} missing required metrics {required}")

        # Sort chronologically and drop empty rows
        sub = sub.sort_index().dropna(how="all")
        if sub.empty:
            continue

        # Synthesize OPEN price if missing using previous CLOSE
        if "OPEN" not in sub.columns:
            sub["OPEN"] = sub["CLOSE"].shift(1)

        # Generate a numeric x-axis for bar-based plotting
        x = np.arange(len(sub))
        width = 0.6

        # Create the main price axis
        fig, ax_price = plt.subplots(figsize=(15, 7))

        # ------------------------------------------------------------------
        # Candlestick rendering (price action)
        # ------------------------------------------------------------------
        for i, row in enumerate(sub.itertuples()):
            # Skip bars with incomplete OHLC data
            if pd.isna(row.OPEN) or pd.isna(row.CLOSE):
                continue

            # Green for up days, red for down days
            color = "green" if row.CLOSE >= row.OPEN else "red"

            # Draw high–low wick
            ax_price.vlines(
                x[i],
                row.LOW,
                row.HIGH,
                color=color,
                linewidth=1,
                zorder=1,
            )

            # Draw open–close body
            ax_price.bar(
                x[i],
                row.CLOSE - row.OPEN,
                width,
                bottom=row.OPEN,
                color=color,
                alpha=0.7,
                zorder=2,
            )

        # ------------------------------------------------------------------
        # Volume axis (secondary y-axis)
        # ------------------------------------------------------------------
        if volume_col in sub.columns:
            # Create a secondary y-axis sharing the same x-axis
            ax_vol = ax_price.twinx()

            # Color volume bars by price direction
            vol_colors = [
                "green" if c >= o else "red"
                for o, c in zip(sub["OPEN"], sub["CLOSE"])
            ]

            # Draw volume bars
            ax_vol.bar(
                x,
                sub[volume_col],
                color=vol_colors,
                alpha=0.3,
                width=width,
                zorder=0,
            )

            # Bind axis styling to volume context
            volume_axis_color = "darkgreen"
            bind_axis_color(ax_vol, volume_axis_color, "Volume")

            # Robust scaling using rolling 95th percentile
            vol_upper = (
                sub[volume_col]
                .rolling(window=20, min_periods=5)
                .quantile(0.95)
                .max()
            )

            ax_vol.set_ylim(0, vol_upper * 1.5)

        # ------------------------------------------------------------------
        # Volatility axis (third y-axis, offset outward)
        # ------------------------------------------------------------------
        if "VOLATILITY" in sub.columns:
            ax_vol2 = ax_price.twinx()

            # Offset the spine to avoid overlap with volume axis
            ax_vol2.spines["right"].set_position(("outward", 60))

            # Plot volatility time series
            line, = ax_vol2.plot(
                x,
                sub["VOLATILITY"],
                linestyle="--",
                linewidth=2,
                zorder=3,
            )

            # Bind axis styling to the plotted line color
            volatility_color = line.get_color()
            bind_axis_color(ax_vol2, volatility_color, "Volatility")

            # Disable grid on auxiliary axis
            ax_vol2.grid(False)

        # ------------------------------------------------------------------
        # Resolve purchase index (entry point)
        # ------------------------------------------------------------------
        if purchase_date is not None:
            purchase_date = pd.Timestamp(purchase_date)

            idx = df.index

            # Align timezone
            if idx.tz is not None and purchase_date.tz is None:
                purchase_date = purchase_date.tz_localize(idx.tz)
            elif idx.tz is None and purchase_date.tz is not None:
                purchase_date = purchase_date.tz_convert(None)

            valid_dates = sub.index[sub.index >= purchase_date]
            purchase_idx = (
                None if valid_dates.empty else sub.index.get_loc(valid_dates[0])
            )
        else:
            purchase_idx = 0

        # ------------------------------------------------------------------
        # BUY marker (entry signal)
        # ------------------------------------------------------------------
        if purchase_idx is not None:
            buy_price = sub["CLOSE"].iloc[purchase_idx]

            # Plot buy marker
            ax_price.scatter(
                x[purchase_idx],
                buy_price,
                marker="^",
                s=120,
                color="red",
                edgecolor="black",
                linewidth=0.8,
                zorder=11,
            )

            # Vertical reference line at entry
            ax_price.axvline(
                x=x[purchase_idx],
                color="black",
                linestyle=":",
                linewidth=2,
                alpha=0.9,
            )

            # Annotate buy price
            y_position_to_avoid_text_overlap_purchase = sub["HIGH"].max() * 0.85

            ax_price.text(
                x[purchase_idx],
                y_position_to_avoid_text_overlap_purchase,
                f"BUY @ {buy_price:.2f}",
                color="red",
                ha="center",
                va="bottom",
                fontsize=10,
                zorder=12,
            )



        # ------------------------------------------------------------------
        # ROI exit marker
        # ------------------------------------------------------------------
        if purchase_idx is not None:
            target_price = buy_price * (1 + roi_target)

            # Scan forward in time for first ROI hit
            for i in range(purchase_idx, len(sub)):
                if sub["CLOSE"].iloc[i] >= target_price:
                    exit_price = sub["CLOSE"].iloc[i]
                    days_to_achieve_ROI = x[i] - x[purchase_idx]
                    # Plot exit marker
                    ax_price.scatter(
                        x[i],
                        exit_price,
                        marker="v",
                        s=140,
                        color="lime",
                        edgecolor="green",
                        linewidth=0.8,
                        zorder=11,
                    )

                    # Vertical reference line at exit
                    ax_price.axvline(
                        x=x[i],
                        color="lime",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.9,
                    )
                    y_position_to_avoid_text_overlap_ROI = sub["HIGH"].max() * 0.95

                    # Annotate ROI and exit price
                    ax_price.text(
                        x[i],
                        y_position_to_avoid_text_overlap_ROI,
                        f"{roi_target*100:.0f}% ROI @ {exit_price:.2f}\nDAYS FROM PURCHASE: {days_to_achieve_ROI:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="green",
                        zorder=12,
                    )
 
                    break

        # ------------------------------------------------------------------
        # Final formatting and layout
        # ------------------------------------------------------------------
        step = max(len(x) // 10, 1)

        ax_price.set_xticks(x[::step])
        ax_price.set_xticklabels(
            sub.index.strftime("%Y-%m-%d")[::step],
            rotation=45,
            ha="right",
        )

        ax_price.set_ylabel("Price")
        ax_price.set_title(f"{action} – Price, Volume, Volatility & ROI")
        ax_price.grid(True, axis="y", alpha=0.3)

        # Add small vertical padding to price axis
        ax_price.set_ylim(
            sub["LOW"].min() * 0.98,
            sub["HIGH"].max() * 1.02,
        )

        plt.tight_layout()
        # -------------- SAVE PLOT --------------
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        filename = output_dir / f"{action.replace('/', '_')}_ROI.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)



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
