"""
LiquidationGrid — final backtest with optimized parameters.

Optimal config found via grid search (1h, 2020-2026):
  Signal: zscore=2.5, min_move=0.15, absorption=0.45
  Backtest: tp=2%, max_hold=120h, max_grid_loss=2%
  Result: WR 89.1%, Sharpe 0.63, Return +0.42%, DD -0.23%

Filters:
  BB(20,2) — close below lower band for long / above upper for short
  Absorption — bar closed in top 45% (long) / bottom 45% (short) of range
  EMA 200 — macro trend direction
  RSI(14) — avoid extremes (>70 for longs, <30 for shorts)
  max_grid_loss_pct — equity stop for black-swan protection
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
import pandas as pd

from backtester.data import get_ohlcv
from backtester.strategy.liquidation_grid import LiquidationGridTrigger
from backtester.engine.grid_backtester import run_grid_backtest
from backtester.engine.backtester import BacktestResult
from backtester.visualization.charts import plot_backtest

SYMBOL  = "BTC/USDT"
CAPITAL = 10_000.0
TODAY   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df_1m  = get_ohlcv(SYMBOL, "1m",  since="2025-12-01", until=TODAY)
df_15m = get_ohlcv(SYMBOL, "15m", since="2022-01-01", until=TODAY)
df_1h  = get_ohlcv(SYMBOL, "1h",  since="2020-01-01", until=TODAY)

print(f"1m:  {len(df_1m):,} bars")
print(f"15m: {len(df_15m):,} bars")
print(f"1h:  {len(df_1h):,} bars")

# ── Optimal params from grid search ──────────────────────────────────────────
STRATEGY_PARAMS = dict(
    zscore_window         = 20,
    zscore_threshold      = 2.5,    # ← tuned
    min_move_pct          = 0.15,   # ← tuned
    use_ema_filter        = True,
    ema_period            = 200,
    use_rsi_filter        = True,
    rsi_period            = 14,
    rsi_long_max          = 70.0,
    rsi_short_min         = 30.0,
    use_adx_filter        = False,
    use_bb_filter         = True,   # ← key filter
    bb_period             = 20,
    bb_dev                = 2.0,
    use_kc_filter         = False,
    use_absorption_filter = True,   # ← key filter
    absorption_min_pos    = 0.45,   # ← tuned
    absorption_max_pos    = 0.55,
)

GRID_PARAMS = dict(
    size_l1          = 0.01,
    size_l2          = 0.01,
    size_rest        = 0.012,
    n_levels         = 5,
    step             = 0.010,
    tp_pct           = 0.020,   # ← tuned: 2% TP
    sl_pct           = 0.0,
    max_grid_loss_pct= 0.02,    # ← equity stop: close if down >2% of capital
    commission       = 0.0004,
    slippage         = 0.0003,
    initial_capital  = CAPITAL,
    symbol           = SYMBOL,
)

# ── Per-timeframe settings ────────────────────────────────────────────────────
TESTS = [
    {"tf": "1m",  "df": df_1m,  "since": "2025-12-01", "cooldown": 5,
     "max_hold": 120 * 60},   # 120h = 7200 min
    {"tf": "15m", "df": df_15m, "since": "2022-01-01", "cooldown": 8,
     "max_hold": 120 * 4},    # 120h = 480 × 15min
    {"tf": "1h",  "df": df_1h,  "since": "2020-01-01", "cooldown": 4,
     "max_hold": 120},        # 120h = 120 × 1h
]

for t in TESTS:
    print(f"\n{'='*60}")
    print(f"  BTC/USDT {t['tf']}  {t['since']} → {TODAY}")
    print(f"{'='*60}")

    df = t["df"]
    print(f"Candles: {len(df):,}")

    strat   = LiquidationGridTrigger(**{**STRATEGY_PARAMS, "cooldown_bars": t["cooldown"]})
    signals = strat.generate_signals(df)
    n_long  = (signals == 1).sum()
    n_short = (signals == -1).sum()
    print(f"Triggers: {n_long} long  {n_short} short")

    if n_long + n_short == 0:
        print("No triggers — skipping.")
        continue

    result = run_grid_backtest(
        df, signals,
        max_hold_bars = t["max_hold"],
        timeframe     = t["tf"],
        **GRID_PARAMS,
    )
    print(result.report())

    if len(result.trades):
        cols = ["entry_time","side","avg_entry","exit_price",
                "levels_filled","pnl_pct","pnl_usd","reason","duration"]
        print(f"\nFirst 20 trades:")
        print(result.trades[cols].head(20).to_string(index=False))
        tp_n  = (result.trades["reason"] == "TP").sum()
        to_n  = (result.trades["reason"] == "timeout").sum()
        gs_n  = (result.trades["reason"] == "grid_stop").sum()
        print(f"\nTP: {tp_n}   Timeout: {to_n}   Grid-stop: {gs_n}")

        # Per-year breakdown
        if len(result.trades) >= 5:
            result.trades["year"] = result.trades["entry_time"].dt.year
            by_year = result.trades.groupby("year").agg(
                trades=("pnl_usd","count"),
                wins=("pnl_usd", lambda x: (x>0).sum()),
                pnl=("pnl_usd","sum"),
            )
            by_year["wr_%"] = (by_year["wins"] / by_year["trades"] * 100).round(1)
            by_year["pnl"] = by_year["pnl"].round(2)
            print(f"\nPer-year breakdown:")
            print(by_year.to_string())

    fname = f"liquidation_grid_{t['tf']}.html"
    compat = BacktestResult(
        equity    = result.equity,
        returns   = result.returns,
        positions = result.positions,
        trades    = result.trades.rename(columns={"avg_entry": "entry_price"}) if len(result.trades) else result.trades,
        metrics   = result.metrics,
        params    = result.params,
        symbol    = SYMBOL,
        timeframe = t["tf"],
    )
    plot_backtest(
        compat, df,
        title=f"{SYMBOL} {t['tf']} — Grid (BB+Absorption+EMA+RSI)",
        save_html=fname,
    )
    print(f"\nChart saved: {fname}")
