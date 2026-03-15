"""
Example: LiquidationBounce — contrarian strategy.

When longs get liquidated (price dumped) → buy the bounce.
When shorts get liquidated (price pumped) → short the reversal.

Requires: export COINALYZE_API_KEY=your_key
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester.data import get_ohlcv, get_liquidations
from backtester.engine.backtester import run_backtest
from backtester.engine.optimizer import grid_search
from backtester.strategy.liquidation_bounce import LiquidationBounce
from backtester.visualization.charts import plot_backtest

SYMBOL    = "BTC/USDT"
TIMEFRAME = "1h"
SINCE     = "2025-12-17"
UNTIL     = None
CAPITAL   = 10_000.0
COMMISSION = 0.001
SLIPPAGE   = 0.0005

print(f"\nLoading OHLCV {SYMBOL} {TIMEFRAME}...")
df = get_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
print(f"Candles: {len(df):,}  ({df.index[0]} → {df.index[-1]})")

print("\nLoading liquidations...")
liq = get_liquidations(symbol=SYMBOL, since=SINCE, until=UNTIL, timeframe=TIMEFRAME)
print(f"Liquidation bars: {len(liq):,}\n")

# ── Single backtest ───────────────────────────────────────────────────────────
strategy = LiquidationBounce(
    long_liq_threshold=50,
    short_liq_threshold=50,
    hold_bars=3,
)
print(f"Strategy: {strategy}\n")

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
print(f"\nAll trades:\n{result.trades[['entry_time','side','entry_price','exit_price','pnl_pct','duration']].to_string(index=False)}\n")

plot_backtest(
    result, df,
    title=f"{SYMBOL} — Liquidation Bounce (contrarian)",
    save_html="liquidation_bounce.html",
)

# ── Grid search ───────────────────────────────────────────────────────────────
print("─"*55)
print("Grid search...")

results = grid_search(
    LiquidationBounce,
    param_grid={
        "long_liq_threshold":  [20, 50, 100, 200],
        "short_liq_threshold": [20, 50, 100, 200],
        "hold_bars":           [1, 2, 3, 6, 12],
        "rsi_filter":          [False, True],
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

print("\nTop 10 by Sharpe ratio:")
print(results.head(10)[[
    "long_liq_threshold", "short_liq_threshold", "hold_bars", "rsi_filter",
    "sharpe_ratio", "total_return_pct", "max_drawdown_pct", "total_trades"
]].to_string(index=False))
