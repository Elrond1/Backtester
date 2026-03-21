"""
MT4-style "Every Tick" backtest using 1-second OHLCV bars.

Demonstrates run_tick_backtest() vs run_backtest() on the same strategy
so you can see the accuracy difference in TP/SL execution.

How it works
------------
* Signals are generated on 1h bars (unchanged from regular backtest).
* TP and SL are checked on every 1-second sub-bar using the bar-direction
  heuristic (bullish bar → O→Low→High→C, bearish → O→High→Low→C).
* This eliminates the "SL always first" bias of bar-by-bar engines and
  gives precision comparable to MT4 Every Tick mode.

Storage
-------
1s klines are cached in DuckDB (~15-30 MB/month per symbol).
Download once, reuse forever — incremental updates on repeated runs.
Binance 1s klines are available from approximately 2020 onwards.

Usage
-----
    python examples/tick_backtest_example.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester.data.manager import get_ohlcv, get_seconds
from backtester.engine import run_backtest, run_tick_backtest
from backtester.strategy.ema_rsi_macd import EmaRsiMacd
from backtester.visualization.charts import plot_backtest

SYMBOL    = "BTC/USDT"
SINCE     = "2023-01-01"   # change to "2020-01-01" for a longer test
TF        = "1h"
CAPITAL   = 10_000.0
TP        = 0.03           # 3% take profit
SL        = 0.015          # 1.5% stop loss

strategy = EmaRsiMacd(
    ema_period=200,
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    swing_period=20,
    rr_ratio=2.0,
)

if __name__ == "__main__":
    # ── Download data ─────────────────────────────────────────────────────────
    print(f"Downloading {SYMBOL} {TF} from {SINCE} …")
    df = get_ohlcv(SYMBOL, TF, since=SINCE)
    print(f"  {len(df):,} candles  ({df.index[0].date()} → {df.index[-1].date()})")

    print(f"\nDownloading {SYMBOL} 1s bars from {SINCE} …")
    print("  (first run downloads from Binance — subsequent runs use local cache)")
    df_1s = get_seconds(SYMBOL, since=SINCE)
    print(f"  {len(df_1s):,} 1-second bars  ({df_1s.index[0].date()} → {df_1s.index[-1].date()})")

    # ── Standard bar-by-bar backtest ──────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("  Standard backtest  (bar-by-bar, SL always checked first)")
    print(f"{'─' * 50}")
    result_std = run_backtest(
        df, strategy,
        initial_capital=CAPITAL,
        take_profit=TP,
        stop_loss=SL,
        symbol=SYMBOL,
        timeframe=TF,
    )
    print(result_std.report())

    # ── MT4-style tick backtest ───────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("  Tick backtest  (1s bars, MT4 Every Tick precision)")
    print(f"{'─' * 50}")
    result_tick = run_tick_backtest(
        df, df_1s, strategy,
        initial_capital=CAPITAL,
        take_profit=TP,
        stop_loss=SL,
        symbol=SYMBOL,
        timeframe=TF,
    )
    print(result_tick.report())

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print("  COMPARISON:  Standard  vs  Tick (1s)")
    print(f"{'═' * 55}")
    keys = [
        "total_return_pct", "cagr_pct", "sharpe_ratio",
        "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades",
    ]
    print(f"  {'Metric':<25} {'Standard':>12} {'Tick (1s)':>12}")
    print(f"  {'-' * 51}")
    for k in keys:
        v_std  = result_std.metrics.get(k, "—")
        v_tick = result_tick.metrics.get(k, "—")
        label  = k.replace("_", " ").title()
        print(f"  {label:<25} {str(v_std):>12} {str(v_tick):>12}")
    print(f"{'═' * 55}")

    # ── Charts ────────────────────────────────────────────────────────────────
    plot_backtest(result_tick, df,
                  title=f"{SYMBOL} {TF} — Tick backtest (1s) | TP {TP*100:.0f}% SL {SL*100:.0f}%",
                  save_html="tick_backtest.html")
    print("\nChart saved → tick_backtest.html")
