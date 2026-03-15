"""
Interactive Plotly charts for backtest visualization.

Layout:
  Row 1 (large): Candlestick + trade markers + optional indicator overlays
  Row 2 (medium): Equity curve vs Buy & Hold
  Row 3 (small):  Volume bars
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.engine.backtester import BacktestResult


def plot_backtest(
    result: BacktestResult,
    df: pd.DataFrame,
    indicators: Optional[dict[str, pd.Series]] = None,
    title: str = "",
    show: bool = True,
    save_html: Optional[str] = None,
) -> go.Figure:
    """
    Plot full backtest report as interactive Plotly chart.

    Parameters
    ----------
    result      : BacktestResult from run_backtest()
    df          : OHLCV DataFrame used in the backtest
    indicators  : Optional dict of {label: pd.Series} to overlay on the candle chart
    title       : Chart title
    show        : Open in browser (default True)
    save_html   : Path to save HTML file (optional)

    Returns
    -------
    plotly Figure
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.30, 0.15],
        vertical_spacing=0.03,
        subplot_titles=("Price & Trades", "Equity Curve", "Volume"),
    )

    _add_candlesticks(fig, df, row=1)
    _add_indicators(fig, indicators or {}, row=1)
    _add_trade_markers(fig, result.trades, df, row=1)
    _add_equity_curve(fig, result.equity, df, row=2)
    _add_volume(fig, df, row=3)

    display_title = title or f"{result.symbol} {result.timeframe} — {result.params}"
    _style(fig, display_title)

    if save_html:
        fig.write_html(save_html)
    if show:
        fig.show()

    return fig


# ------------------------------------------------------------------ helpers

def _add_candlesticks(fig: go.Figure, df: pd.DataFrame, row: int):
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=row, col=1,
    )


def _add_indicators(fig: go.Figure, indicators: dict[str, pd.Series], row: int):
    colors = ["#2196F3", "#FF9800", "#9C27B0", "#00BCD4"]
    for i, (label, series) in enumerate(indicators.items()):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.85,
            ),
            row=row, col=1,
        )


def _add_trade_markers(fig: go.Figure, trades: pd.DataFrame, df: pd.DataFrame, row: int):
    if trades.empty:
        return

    long_entries = trades[trades["side"] == "long"]
    short_entries = trades[trades["side"] == "short"]

    def _get_price(times, offset_pct, high_or_low):
        prices = []
        for t in times:
            idx = df.index.get_indexer([t], method="nearest")[0]
            base = df[high_or_low].iloc[idx]
            prices.append(base * (1 + offset_pct))
        return prices

    # Long entry: green triangle up below bar
    if not long_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=long_entries["entry_time"],
                y=_get_price(long_entries["entry_time"], -0.015, "low"),
                mode="markers",
                name="Long entry",
                marker=dict(symbol="triangle-up", size=10, color="#26a69a", line=dict(width=1, color="white")),
                customdata=long_entries[["entry_price", "exit_price", "pnl_pct"]].values,
                hovertemplate=(
                    "<b>Long Entry</b><br>"
                    "Entry: %{customdata[0]:.4f}<br>"
                    "Exit: %{customdata[1]:.4f}<br>"
                    "PnL: %{customdata[2]:.2f}%<extra></extra>"
                ),
            ),
            row=row, col=1,
        )
        # Long exit: red triangle down above bar
        fig.add_trace(
            go.Scatter(
                x=long_entries["exit_time"],
                y=_get_price(long_entries["exit_time"], 0.015, "high"),
                mode="markers",
                name="Long exit",
                marker=dict(symbol="triangle-down", size=10, color="#ef5350", line=dict(width=1, color="white")),
                showlegend=False,
                hovertemplate="<b>Long Exit</b><br>%{x}<extra></extra>",
            ),
            row=row, col=1,
        )

    # Short entry: red triangle down above bar
    if not short_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=short_entries["entry_time"],
                y=_get_price(short_entries["entry_time"], 0.015, "high"),
                mode="markers",
                name="Short entry",
                marker=dict(symbol="triangle-down", size=10, color="#ef5350", line=dict(width=1, color="white")),
                customdata=short_entries[["entry_price", "exit_price", "pnl_pct"]].values,
                hovertemplate=(
                    "<b>Short Entry</b><br>"
                    "Entry: %{customdata[0]:.4f}<br>"
                    "Exit: %{customdata[1]:.4f}<br>"
                    "PnL: %{customdata[2]:.2f}%<extra></extra>"
                ),
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=short_entries["exit_time"],
                y=_get_price(short_entries["exit_time"], -0.015, "low"),
                mode="markers",
                name="Short exit",
                marker=dict(symbol="triangle-up", size=10, color="#26a69a", line=dict(width=1, color="white")),
                showlegend=False,
                hovertemplate="<b>Short Exit</b><br>%{x}<extra></extra>",
            ),
            row=row, col=1,
        )


def _add_equity_curve(fig: go.Figure, equity: pd.Series, df: pd.DataFrame, row: int):
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Strategy",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.08)",
        ),
        row=row, col=1,
    )

    # Buy & Hold reference
    initial = equity.iloc[0]
    bnh = (df["close"] / df["close"].iloc[0]) * initial
    bnh = bnh.reindex(equity.index, method="ffill")
    fig.add_trace(
        go.Scatter(
            x=bnh.index,
            y=bnh.values,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#9E9E9E", width=1.5, dash="dot"),
        ),
        row=row, col=1,
    )


def _add_volume(fig: go.Figure, df: pd.DataFrame, row: int):
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            showlegend=False,
        ),
        row=row, col=1,
    )


def _style(fig: go.Figure, title: str):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_dark",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(row=1, col=1, title_text="Price")
    fig.update_yaxes(row=2, col=1, title_text="Capital")
    fig.update_yaxes(row=3, col=1, title_text="Volume")
