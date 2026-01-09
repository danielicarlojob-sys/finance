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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@log_exceptions_with_retry(
    max_retries=5,
    prefix_fn=debug_print,
    retry_delay=1.0,
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
    Interactive candlestick plot with volume, volatility, and ROI logic.
    Output is a standalone HTML file per ACTION with full zoom/hover support.
    """

    # ------------------------------------------------------------------
    # Validate structure
    # ------------------------------------------------------------------
    if df.columns.nlevels != 3:
        raise ValueError("Expected 3-level MultiIndex columns")

    data = df.copy()

    if start:
        data = data.loc[data.index >= pd.Timestamp(start)]
    if end:
        data = data.loc[data.index <= pd.Timestamp(end)]

    all_actions = data.columns.get_level_values("ACTION").unique()
    actions = all_actions if actions is None else actions

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Iterate over actions
    # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Purchase index resolution
        # ------------------------------------------------------------------
        if purchase_date is not None:
            purchase_date = pd.Timestamp(purchase_date)
            valid_dates = sub.index[sub.index >= purchase_date]
            purchase_idx = None if valid_dates.empty else sub.index.get_loc(valid_dates[0])
        else:
            purchase_idx = 0

        # ------------------------------------------------------------------
        # Create figure with 3 y-axes
        # ------------------------------------------------------------------
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"secondary_y": True}]],
        )

        # ------------------------------------------------------------------
        # Candlestick trace
        # ------------------------------------------------------------------
        fig.add_trace(
            go.Candlestick(
                x=sub.index,
                open=sub["OPEN"],
                high=sub["HIGH"],
                low=sub["LOW"],
                close=sub["CLOSE"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            secondary_y=False,
        )

        # ------------------------------------------------------------------
        # Volume trace
        # ------------------------------------------------------------------
        if volume_col in sub.columns:
            fig.add_trace(
                go.Bar(
                    x=sub.index,
                    y=sub[volume_col],
                    name="Volume",
                    opacity=0.3,
                    marker_color="darkgreen",
                ),
                secondary_y=True,
            )

        # ------------------------------------------------------------------
        # Volatility trace (3rd axis)
        # ------------------------------------------------------------------
        if "VOLATILITY" in sub.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=sub["VOLATILITY"],
                    mode="lines",
                    name="Volatility",
                    line=dict(dash="dash", width=2),
                    yaxis="y3",
                )
            )

        # ------------------------------------------------------------------
        # BUY marker
        # ------------------------------------------------------------------
        if purchase_idx is not None:
            buy_price = sub["CLOSE"].iloc[purchase_idx]
            buy_date = sub.index[purchase_idx]

            fig.add_trace(
                go.Scatter(
                    x=[buy_date],
                    y=[buy_price],
                    mode="markers+text",
                    marker=dict(
                        symbol="triangle-up",
                        size=14,
                        color="red",
                        line=dict(width=1, color="black"),
                    ),
                    text=[f"BUY @ {buy_price:.2f}"],
                    textposition="top center",
                    name="Buy",
                )
            )

        # ------------------------------------------------------------------
        # ROI exit marker
        # ------------------------------------------------------------------
        if purchase_idx is not None:
            target_price = buy_price * (1 + roi_target)

            for i in range(purchase_idx, len(sub)):
                if sub["CLOSE"].iloc[i] >= target_price:
                    exit_price = sub["CLOSE"].iloc[i]
                    exit_date = sub.index[i]
                    days = (exit_date - buy_date).days

                    fig.add_trace(
                        go.Scatter(
                            x=[exit_date],
                            y=[exit_price],
                            mode="markers+text",
                            marker=dict(
                                symbol="triangle-down",
                                size=16,
                                color="lime",
                                line=dict(width=1, color="green"),
                            ),
                            text=[f"{roi_target*100:.0f}% ROI @ {exit_price:.2f}<br>{days} days"],
                            textposition="top center",
                            name="ROI Exit",
                        )
                    )
                    break

        # ------------------------------------------------------------------
        # Layout & axes
        # ------------------------------------------------------------------
        fig.update_layout(
            title=f"{action} – Interactive Price, Volume, Volatility & ROI",
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
            ),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right"),
            yaxis3=dict(
                title="Volatility",
                overlaying="y",
                side="right",
                position=1.08,
            ),
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        # ------------------------------------------------------------------
        # Save interactive HTML
        # ------------------------------------------------------------------
        filename = output_dir / f"{action.replace('/', '_')}_ROI.html"
        fig.write_html(filename, include_plotlyjs="cdn")


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
