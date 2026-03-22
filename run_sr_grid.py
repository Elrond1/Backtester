"""
S/R Grid Strategy — Backtest Runner
=====================================
Backtests BTC/USDT from 2020-01-01 to today using 1-second bar precision.

Strategy:
  - Support/Resistance = midpoint of (max_high + min_low) over last 30 D1 bars
  - Enter LONG when price is within 0.5% of S/R level
  - First order:  3% of $10,000 = $300
  - Averaging:    +$300 every 4%, 4%, then every 2% thereafter
  - Take Profit:  3% above weighted average entry price

Data requirements (downloaded once, cached in DuckDB):
  - D1 bars since 2019-01-01   (~1 MB)
  - 1s bars since 2020-01-01   (~2 GB, takes 30-60 min on first run)

Usage:
    python run_sr_grid.py
"""

import os
import tempfile
import webbrowser
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.data.manager import get_ohlcv, _cache, _parse_dt, _fetch_1s_incremental
from backtester.engine.sr_grid_engine import run_sr_grid_backtest_chunked

# ── Strategy parameters ────────────────────────────────────────────────────────

SYMBOL          = "BTC/USDT"
START_DATE      = "2020-01-01"
INITIAL_CAP     = 10_000.0

ORDER_SIZE_PCT  = 0.03    # 3% of initial capital per order = $300
FIRST_STEP      = 0.04    # first averaging after 4% drop
SECOND_STEP     = 0.04    # second averaging after another 4%
SUBSEQ_STEP     = 0.02    # each subsequent averaging every 2%
TAKE_PROFIT     = 0.03    # TP: 3% above average entry
MAX_ORDERS      = 10      # maximum grid depth
LOOKBACK_D1       = 30    # D1 bars for long S/R (support)
LOOKBACK_SHORT_D1 = 7     # D1 bars for short S/R (resistance, shorter = closer to price)
TOLERANCE         = 0.015 # 1.5% price tolerance around S/R for entry
MA_PERIOD         = 200   # MA trend filter: above → long only, below → short only
SHORT_SL          = 0.15  # close SHORT if price rises 15% above avg entry
COMMISSION      = 0.001   # 0.1% per trade side (Binance)
SLIPPAGE        = 0.0005  # 0.05% price impact

# ── Download data ──────────────────────────────────────────────────────────────

print("=" * 58)
print("  S/R Grid Backtest — BTC/USDT (2020 to today)")
print("=" * 58)

print("\n[1/3] Loading D1 bars (extra year for S/R lookback)...")
df_d1 = get_ohlcv(SYMBOL, "1d", since="2019-01-01")
print(f"      {len(df_d1):,} daily bars  ({df_d1.index[0].date()} → {df_d1.index[-1].date()})")

print("\n[2/3] Downloading 1-second bars (first run: ~15 min, then instant)...")
start_dt = _parse_dt(START_DATE)
end_dt   = datetime.now(timezone.utc)
_fetch_1s_incremental(_cache, SYMBOL, start_dt, end_dt)
print("      1-second bars ready in cache.")

# ── Run backtest ───────────────────────────────────────────────────────────────

print("\n[3/3] Running backtest (month by month, no RAM spike)...")
result = run_sr_grid_backtest_chunked(
    cache           = _cache,
    df_d1           = df_d1,
    symbol          = SYMBOL,
    start           = start_dt,
    end             = end_dt,
    initial_capital = INITIAL_CAP,
    order_size_pct  = ORDER_SIZE_PCT,
    first_avg_step  = FIRST_STEP,
    second_avg_step = SECOND_STEP,
    subsequent_step = SUBSEQ_STEP,
    take_profit_pct = TAKE_PROFIT,
    max_orders      = MAX_ORDERS,
    lookback_d1       = LOOKBACK_D1,
    lookback_short_d1 = LOOKBACK_SHORT_D1,
    entry_tolerance = TOLERANCE,
    commission      = COMMISSION,
    slippage        = SLIPPAGE,
    ma_period         = MA_PERIOD,
    short_stop_loss   = SHORT_SL,
)

print("\n")
print(result)

# ── Build Plotly report ────────────────────────────────────────────────────────

year_now = datetime.now().year
final_eq = result.equity.iloc[-1]
total_pnl = final_eq - INITIAL_CAP
total_pnl_pct = (final_eq / INITIAL_CAP - 1) * 100
n_trades = len(result.trades)

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=[
        "Equity Curve (USD)",
        "BTC/USDT Daily Price + S/R Level + Trades",
        "Trade PnL per Close (USD)",
    ],
    row_heights=[0.30, 0.48, 0.22],
    vertical_spacing=0.06,
    shared_xaxes=True,
)

# ── Row 1: Equity curve ──
eq = result.equity
running_max = eq.cummax()
drawdown = (eq - running_max) / running_max * 100
max_dd_val = drawdown.min()
max_dd_idx = drawdown.idxmin()
peak_idx   = running_max.loc[:max_dd_idx].idxmax()

fig.add_trace(go.Scatter(
    x=eq.index, y=eq.values,
    mode="lines", name="Equity",
    line=dict(color="#00b894", width=2),
    fill="tozeroy", fillcolor="rgba(0,184,148,0.08)",
), row=1, col=1)

# Max drawdown shaded region
fig.add_vrect(
    x0=peak_idx, x1=max_dd_idx,
    fillcolor="rgba(231,76,60,0.12)",
    layer="below", line_width=0,
    annotation_text=f"Max DD: {max_dd_val:.1f}%",
    annotation_position="top left",
    annotation_font_color="#e74c3c",
    row=1, col=1,
)

fig.add_hline(
    y=INITIAL_CAP,
    line_dash="dot",
    line_color="rgba(255,255,255,0.3)",
    annotation_text=f"Initial ${INITIAL_CAP:,.0f}",
    annotation_font_color="gray",
    row=1, col=1,
)

# ── Row 2: Daily candlestick + S/R ──
df_d1_plot = df_d1.loc[START_DATE:]
fig.add_trace(go.Candlestick(
    x=df_d1_plot.index,
    open=df_d1_plot["open"],
    high=df_d1_plot["high"],
    low=df_d1_plot["low"],
    close=df_d1_plot["close"],
    name="BTCUSDT",
    increasing_line_color="#2ecc71",
    decreasing_line_color="#e74c3c",
    showlegend=True,
), row=2, col=1)

sr_plot = result.sr_levels.loc[START_DATE:]
fig.add_trace(go.Scatter(
    x=sr_plot.index,
    y=sr_plot.values,
    mode="lines",
    name="S/R Level",
    line=dict(color="#f39c12", width=1.5, dash="dash"),
), row=2, col=1)

if not result.ma_levels.empty:
    ma_plot = result.ma_levels.loc[START_DATE:]
    fig.add_trace(go.Scatter(
        x=ma_plot.index,
        y=ma_plot.values,
        mode="lines",
        name=f"MA{MA_PERIOD}",
        line=dict(color="#9b59b6", width=1.5),
    ), row=2, col=1)

# Trade markers — separate LONG and SHORT
if not result.trades.empty:
    longs  = result.trades[result.trades["side"] == "long"]
    shorts = result.trades[result.trades["side"] == "short"]

    # Long entries
    if not longs.empty:
        fig.add_trace(go.Scatter(
            x=longs["entry_time"], y=longs["avg_entry"],
            mode="markers", name="Long Entry",
            marker=dict(color="#2ecc71", size=10, symbol="triangle-up"),
            text=[f"LONG  Orders:{r.n_orders}  Invested:${r.invested_usd:,.0f}"
                  for r in longs.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=longs["exit_time"], y=longs["exit_price"],
            mode="markers", name="Long Exit (TP)",
            marker=dict(color="#27ae60", size=10, symbol="triangle-down"),
            text=[f"LONG TP  PnL:${r.pnl_usd:+,.2f} ({r.pnl_pct:+.2f}%)"
                  for r in longs.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ), row=2, col=1)

    # Short entries
    if not shorts.empty:
        fig.add_trace(go.Scatter(
            x=shorts["entry_time"], y=shorts["avg_entry"],
            mode="markers", name="Short Entry",
            marker=dict(color="#e74c3c", size=10, symbol="triangle-down"),
            text=[f"SHORT  Orders:{r.n_orders}  Invested:${r.invested_usd:,.0f}"
                  for r in shorts.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=shorts["exit_time"], y=shorts["exit_price"],
            mode="markers", name="Short Exit (TP)",
            marker=dict(color="#c0392b", size=10, symbol="triangle-up"),
            text=[f"SHORT TP  PnL:${r.pnl_usd:+,.2f} ({r.pnl_pct:+.2f}%)"
                  for r in shorts.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ), row=2, col=1)

    # ── Row 3: PnL bars ──
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in result.trades["pnl_usd"]]
    fig.add_trace(go.Bar(
        x=result.trades["exit_time"],
        y=result.trades["pnl_usd"],
        name="Trade PnL",
        marker_color=colors,
        hovertemplate="PnL: $%{y:,.2f}<extra></extra>",
    ), row=3, col=1)

# ── Layout ──
fig.update_layout(
    title=dict(
        text=(
            f"S/R Grid Strategy — BTC/USDT (2020–{year_now})<br>"
            f"<sup>Final equity: ${final_eq:,.0f} | "
            f"Total PnL: ${total_pnl:+,.0f} ({total_pnl_pct:+.1f}%) | "
            f"Trades: {n_trades}</sup>"
        ),
        font=dict(size=15),
    ),
    height=1150,
    template="plotly_dark",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False,
    xaxis3_rangeslider_visible=False,
)

fig.update_yaxes(title_text="USD", row=1, col=1)
fig.update_yaxes(title_text="Price (USD)", row=2, col=1)
fig.update_yaxes(title_text="PnL (USD)", row=3, col=1)

# ── Save and open in browser ──
html_path = os.path.join(tempfile.gettempdir(), "sr_grid_backtest.html")
fig.write_html(html_path, auto_open=False)

print(f"\nReport saved → {html_path}")
print("Opening in browser...")
webbrowser.open(f"file://{html_path}")
