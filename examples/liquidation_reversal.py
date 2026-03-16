"""
Example: LiquidationReversalSignals — AlgoAlpha-style indicator backtest.

Logic recap:
  1. Z-score of directional volume (up/down split) detects abnormal liquidation pressure
  2. When a volume spike occurs AGAINST the Supertrend direction:
       Bearish ST + big buying   → shorts being squeezed → bullish reversal pending
       Bullish ST + big selling  → longs being liquidated → bearish reversal pending
  3. When Supertrend flips within timeout_bars of the spike → confirmed entry signal

No external data needed — uses standard OHLCV volume from Binance.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester.data import get_ohlcv
from backtester.engine.backtester import run_backtest
from backtester.engine.optimizer import grid_search
from backtester.strategy.liquidation_reversal import LiquidationReversalSignals
from backtester.visualization.charts import plot_backtest

SYMBOL     = "BTC/USDT"
TIMEFRAME  = "1h"
SINCE      = "2024-01-01"   # Binance free data — go further back for a proper test
UNTIL      = "2025-03-01"
CAPITAL    = 10_000.0
COMMISSION = 0.001
SLIPPAGE   = 0.0005

print(f"\nLoading OHLCV {SYMBOL} {TIMEFRAME}  {SINCE} → {UNTIL}...")
df = get_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
print(f"Candles: {len(df):,}  ({df.index[0]} → {df.index[-1]})")

# Lower timeframe for volume analysis (key to AlgoAlpha logic)
LTF = "15m"
print(f"\nLoading lower TF {LTF} for directional volume analysis...")
df_ltf = get_ohlcv(SYMBOL, LTF, since=SINCE, until=UNTIL)
print(f"LTF candles: {len(df_ltf):,}\n")

# ── Single backtest ───────────────────────────────────────────────────────────
strategy = LiquidationReversalSignals(
    zscore_length=20,
    zscore_threshold=3.0,
    timeout_bars=3,
    st_period=10,
    st_multiplier=4.0,
    hold_bars=12,
    macro_filter=True,
    macro_period=200,
)
print(f"Strategy: {strategy}\n")

result = run_backtest(
    df, strategy,
    initial_capital=CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    aux={"lower_tf": df_ltf},
)
print(result.report())

if len(result.trades):
    print(f"\nSample trades (first 20):\n"
          f"{result.trades[['entry_time','side','entry_price','exit_price','pnl_pct','duration']].head(20).to_string(index=False)}\n")

plot_backtest(
    result, df,
    title=f"{SYMBOL} {TIMEFRAME} — Liquidation Reversal Signals",
    save_html="liquidation_reversal.html",
)

# ── Grid search ───────────────────────────────────────────────────────────────
print("─" * 55)
print("Grid search...")

results = grid_search(
    LiquidationReversalSignals,
    param_grid={
        "zscore_threshold": [2.0, 2.5, 3.0],
        "timeout_bars":     [3, 5, 8],
        "st_multiplier":    [2.0, 3.0, 4.0],
        "hold_bars":        [6, 12, 24],
        "macro_filter":     [False, True],
    },
    df=df,
    metric="sharpe_ratio",
    initial_capital=CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    aux={"lower_tf": df_ltf},
    strategy_kwargs={"zscore_length": 20, "st_period": 10},
)

print("\nTop 10 by Sharpe ratio:")
print(results.head(10)[[
    "zscore_threshold", "timeout_bars", "st_multiplier", "hold_bars",
    "sharpe_ratio", "total_return_pct", "max_drawdown_pct", "total_trades",
]].to_string(index=False))
