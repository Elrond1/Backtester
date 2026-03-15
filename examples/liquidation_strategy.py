"""
Example: LiquidationSpike strategy on BTC/USDT 1h.

Requires free Coinglass API key:
  1. Register at https://coinglass.com
  2. Profile → API → copy key
  3. export COINGLASS_API_KEY=your_key_here

Strategy logic:
  - When hourly short liquidations > $50M → shorts squeezed → go LONG (trend)
  - When hourly long liquidations > $50M → longs squeezed → go SHORT (trend)
  - Hold for 3 bars (3 hours), then exit

Capital: $10,000. Commission: 0.1% (Binance spot taker).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester.data import get_ohlcv, get_liquidations
from backtester.engine.backtester import run_backtest
from backtester.engine.optimizer import grid_search
from backtester.strategy.liquidation_spike import LiquidationSpike
from backtester.visualization.charts import plot_backtest

SYMBOL    = "BTC/USDT"
TIMEFRAME = "1h"
SINCE     = "2023-01-01"
UNTIL     = "2024-01-01"
CAPITAL   = 10_000.0
COMMISSION = 0.001
SLIPPAGE   = 0.0005

# ── 1. Load data ──────────────────────────────────────────────────────────────
print(f"\nLoading OHLCV {SYMBOL} {TIMEFRAME}...")
df = get_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
print(f"Candles: {len(df):,}  ({df.index[0]} → {df.index[-1]})")

print("\nLoading liquidations from Coinglass...")
liq = get_liquidations(
    symbol=SYMBOL,
    since=SINCE,
    until=UNTIL,
    timeframe=TIMEFRAME,
)
print(f"Liquidation bars: {len(liq):,}")
print(f"\nLiquidation stats (USD):")
print(liq[["liq_long", "liq_short", "liq_total"]].describe().to_string())

# ── 2. Single backtest ────────────────────────────────────────────────────────
print("\n" + "─"*50)
strategy = LiquidationSpike(
    threshold_usd=50_000_000,   # $50M per hour triggers signal
    side="both",                # trade both long and short squeezes
    direction="trend",          # follow the momentum
    hold_bars=3,                # hold 3 hours after signal
)
print(f"Strategy: {strategy}")

result = run_backtest(
    df, strategy,
    initial_capital=CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    aux={"liq": liq},
)
print(result.report())
print(f"\nSample trades:\n{result.trades.head(10).to_string(index=False)}")

# Visualize liquidation spikes on the chart
liq_spike_long  = liq["liq_long"].reindex(df.index, method="ffill")
liq_spike_short = liq["liq_short"].reindex(df.index, method="ffill")

plot_backtest(
    result, df,
    title=f"{SYMBOL} {TIMEFRAME} — Liquidation Spike (${strategy.threshold_usd/1e6:.0f}M)",
    save_html="liquidation_backtest.html",
)

# ── 3. Grid search ────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("Grid search: threshold × hold_bars...")

results = grid_search(
    LiquidationSpike,
    param_grid={
        "threshold_usd": [20_000_000, 50_000_000, 100_000_000, 200_000_000],
        "hold_bars":      [1, 2, 3, 6],
        "direction":      ["trend", "contrarian"],
    },
    df=df,
    metric="sharpe_ratio",
    initial_capital=CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    aux={"liq": liq},
)

print("\nTop 10 combinations:")
print(results.head(10)[[
    "threshold_usd", "hold_bars", "direction",
    "sharpe_ratio", "total_return_pct", "max_drawdown_pct", "total_trades"
]].to_string(index=False))
