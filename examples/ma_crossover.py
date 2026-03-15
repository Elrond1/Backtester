"""
Example: Moving Average Crossover strategy on BTC/USDT 1h.

1. Downloads historical data from data.binance.vision (cached in DuckDB)
2. Runs vectorized backtest
3. Prints performance report
4. Plots interactive Plotly chart (opens in browser)
5. (Optional) Grid search for best MA parameters
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from backtester.data import get_ohlcv
from backtester.engine.backtester import run_backtest
from backtester.engine.optimizer import grid_search
from backtester.strategy.base import Strategy
from backtester.strategy.indicators import sma, ema, rsi
from backtester.visualization.charts import plot_backtest


class MACrossover(Strategy):
    """
    Classic dual moving average crossover.
    Goes long when fast MA crosses above slow MA, exits when it crosses below.
    """

    def __init__(self, fast: int = 20, slow: int = 50, ma_type: str = "sma"):
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        fn = sma if self.ma_type == "sma" else ema
        fast_ma = fn(df["close"], self.fast)
        slow_ma = fn(df["close"], self.slow)
        return (fast_ma > slow_ma).astype(int)


class RSIStrategy(Strategy):
    """
    Buy when RSI crosses below oversold level, sell when crosses above overbought.
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        r = rsi(df["close"], self.period)
        signal = pd.Series(0, index=df.index)
        signal[r < self.oversold] = 1
        signal[r > self.overbought] = -1
        # Hold position until opposite signal
        signal = signal.replace(0, pd.NA).ffill().fillna(0)
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "1h"
    SINCE = "2023-01-01"
    UNTIL = "2024-01-01"
    CAPITAL = 10_000.0
    COMMISSION = 0.001   # 0.1% (Binance spot taker fee)
    SLIPPAGE = 0.0005

    print(f"\nDownloading {SYMBOL} {TIMEFRAME} from {SINCE} to {UNTIL}...")
    df = get_ohlcv(SYMBOL, TIMEFRAME, since=SINCE, until=UNTIL)
    print(f"Loaded {len(df):,} candles  ({df.index[0]} → {df.index[-1]})\n")

    # ── 1. Single backtest ────────────────────────────────────────────────────
    strategy = MACrossover(fast=20, slow=50)
    result = run_backtest(
        df, strategy,
        initial_capital=CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
    )
    print(result.report())
    print(f"\nTrades:\n{result.trades.head(10).to_string(index=False)}\n")

    # ── 2. Plot ───────────────────────────────────────────────────────────────
    from backtester.strategy.indicators import sma as _sma
    indicators = {
        f"SMA {strategy.fast}": _sma(df["close"], strategy.fast),
        f"SMA {strategy.slow}": _sma(df["close"], strategy.slow),
    }
    plot_backtest(
        result, df,
        indicators=indicators,
        title=f"{SYMBOL} {TIMEFRAME} — MA Crossover ({strategy.fast}/{strategy.slow})",
        save_html="backtest_result.html",
    )

    # ── 3. Grid search (optional, comment out to skip) ────────────────────────
    print("\nRunning grid search (fast x slow combinations)...")
    opt_results = grid_search(
        MACrossover,
        param_grid={"fast": [10, 20, 30, 50], "slow": [50, 100, 150, 200]},
        df=df,
        metric="sharpe_ratio",
        initial_capital=CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
    )
    print("\nTop 10 parameter combinations by Sharpe ratio:")
    print(opt_results.head(10)[
        ["fast", "slow", "sharpe_ratio", "total_return_pct", "max_drawdown_pct", "win_rate_pct"]
    ].to_string(index=False))
