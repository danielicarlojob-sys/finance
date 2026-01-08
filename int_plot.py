import plotly.graph_objects as go
import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=20),
    "Open":  [100 + i for i in range(20)],
    "High":  [102 + i for i in range(20)],
    "Low":   [98 + i for i in range(20)],
    "Close": [101 + i for i in range(20)],
})

fig = go.Figure(
    data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
        )
    ]
)

fig.update_layout(
    title="Interactive Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True,
)

# Save interactive plot
fig.write_html("candlestick_interactive.html")
