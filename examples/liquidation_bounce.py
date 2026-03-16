"""
Example: LiquidationBounce — contrarian strategy with TP/SL + trend filter.

Idea:
  When longs get liquidated (price dumped) → buy the bounce.
  When shorts get liquidated (price pumped) → short the reversal.

Key insight #1: only trade LARGE liquidation events (z-score > threshold),
not every dip. After truly anomalous cascades, a bounce is much more likely.

Key insight #2: trade WITH the macro trend:
  - In uptrend (price > MA): only buy dips after long liq cascades
  - In downtrend (price < MA): only short pumps after short liq cascades
  Prevents catching falling knives in a bear market.

Key insight #3: take profit quickly, cut losses hard.
  TP=1.5%, SL=0.7% — asymmetric in terms of R:R but bounces are short-lived.

Requires: export COINALYZE_API_KEY=your_key
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester.data import get_ohlcv, get_liquidations
from backtester.engine.backtester import run_backtest
from backtester.engine.optimizer import grid_search
from backtester.strategy.liquidation_bounce import LiquidationBounce
from backtester.visualization.charts import plot_backtest

SYMBOL     = "BTC/USDT"
TIMEFRAME  = "1h"
SINCE      = "2025-12-17"
UNTIL      = None
CAPITAL    = 10_000.0
COMMISSION = 0.001
SLIPPAGE   = 0.0005

print(f"\nLoading OHLCV {SYMBOL} {TIMEFRAME}...")
df = get_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
print(f"Candles: {len(df):,}  ({df.index[0]} → {df.index[-1]})")

print("\nLoading liquidations...")
liq = get_liquidations(symbol=SYMBOL, since=SINCE, until=UNTIL, timeframe=TIMEFRAME)
print(f"Liquidation bars: {len(liq):,}\n")

# ── Single backtest ───────────────────────────────────────────────────────────
# z-score mode: only anomalous liquidation spikes (> 2σ over 7-day rolling window)
# trend_filter: only long in uptrend / only short in downtrend
# TP=1.5%, SL=0.7% — quick exits before trend resumes
strategy = LiquidationBounce(
    zscore_mode=True,
    zscore_threshold=2.0,
    hold_bars=4,
    trend_filter=True,
    trend_period=48,     # 48h = 2-day MA for macro trend
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
    take_profit=0.015,   # exit at +1.5%
    stop_loss=0.007,     # cut at -0.7%
)
print(result.report())
if len(result.trades):
    print(f"\nAll trades:\n{result.trades[['entry_time','side','entry_price','exit_price','pnl_pct','duration']].to_string(index=False)}\n")

plot_backtest(
    result, df,
    title=f"{SYMBOL} — Liquidation Bounce (z-score + trend filter, TP/SL)",
    save_html="liquidation_bounce.html",
)

# ── Grid search ───────────────────────────────────────────────────────────────
print("─"*55)
print("Grid search (z-score + trend filter)...")

results = grid_search(
    LiquidationBounce,
    param_grid={
        "zscore_threshold": [1.5, 2.0, 2.5, 3.0],
        "hold_bars":        [2, 3, 4, 6],
        "trend_period":     [24, 48, 96],
    },
    df=df,
    metric="sharpe_ratio",
    initial_capital=CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    aux={"liq": liq},
    run_kwargs={"take_profit": 0.015, "stop_loss": 0.007},
    strategy_kwargs={"zscore_mode": True, "trend_filter": True},
)

print("\nTop 10 by Sharpe ratio:")
print(results.head(10)[[
    "zscore_threshold", "hold_bars", "trend_period",
    "sharpe_ratio", "total_return_pct", "max_drawdown_pct", "total_trades"
]].to_string(index=False))
