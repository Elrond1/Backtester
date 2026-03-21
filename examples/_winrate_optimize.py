"""Optimize liquidation_reversal for max win rate (min 10 trades)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester.data import get_ohlcv
from backtester.engine.optimizer import grid_search
from backtester.engine.backtester import run_backtest
from backtester.strategy.liquidation_reversal import LiquidationReversalSignals
from backtester.visualization.charts import plot_backtest

SYMBOL     = "BTC/USDT"
TIMEFRAME  = "1h"
SINCE      = "2024-01-01"
UNTIL      = "2025-01-01"
CAPITAL    = 10_000.0
COMMISSION = 0.001
SLIPPAGE   = 0.0005

print("Loading data...")
df     = get_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
df_ltf = get_ohlcv(SYMBOL, "15m",    since=SINCE, until=UNTIL)
print(f"HTF candles: {len(df):,}  LTF candles: {len(df_ltf):,}\n")

results = grid_search(
    LiquidationReversalSignals,
    param_grid={
        "zscore_threshold": [3.0, 3.5, 4.0],
        "timeout_bars":     [2, 3, 5],
        "st_multiplier":    [3.0, 4.0],
        "hold_bars":        [12, 24],
        "macro_filter":     [True],
    },
    df=df,
    metric="win_rate_pct",
    initial_capital=CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    aux={"lower_tf": df_ltf},
    strategy_kwargs={"zscore_length": 20, "st_period": 10},
)

# Filter: at least 15 trades
filtered = results[results["total_trades"] >= 15].reset_index(drop=True)

print("\nTop 15 by Win Rate (min 10 trades):")
cols = ["zscore_threshold","timeout_bars","st_multiplier","hold_bars","macro_filter",
        "win_rate_pct","profit_factor","sharpe_ratio","total_return_pct","max_drawdown_pct","total_trades"]
print(filtered[cols].head(15).to_string(index=False))

# Run best config
best = filtered.iloc[0]
print(f"\n=== Best config ===")
print(f"zscore_threshold={best.zscore_threshold}, timeout_bars={int(best.timeout_bars)}, "
      f"st_multiplier={best.st_multiplier}, hold_bars={int(best.hold_bars)}, macro_filter={best.macro_filter}")

strategy = LiquidationReversalSignals(
    zscore_length=20,
    zscore_threshold=best.zscore_threshold,
    timeout_bars=int(best.timeout_bars),
    st_period=10,
    st_multiplier=best.st_multiplier,
    hold_bars=int(best.hold_bars),
    macro_filter=bool(best.macro_filter),
)

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
    print(f"\nTrades:\n{result.trades[['entry_time','side','entry_price','exit_price','pnl_pct','duration']].to_string(index=False)}\n")

plot_backtest(
    result, df,
    title=f"{SYMBOL} {TIMEFRAME} — Liquidation Reversal (WinRate Optimized)",
    save_html="liquidation_reversal_winrate_2020.html",
)
print("Chart saved: liquidation_reversal_winrate.html")
