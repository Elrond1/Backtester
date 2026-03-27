"""
BreakBar + HalfTrend filter.

Фильтр направления:
  - Long  только когда HalfTrend = +1 (uptrend)
  - Short только когда HalfTrend = -1 (downtrend)

MM: risk_per_trade = 6% (DD ~25%)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from backtester.data import get_ohlcv
from backtester.strategy.breakbar import BreakBar
from backtester.strategy.indicators import halftrend
from backtester.visualization.charts import plot_backtest
from examples.breakbar_optimize import run_mm_backtest


class BreakBarHalfTrend(BreakBar):
    """
    BreakBar с фильтром HalfTrend.

    Параметры HalfTrend
    -------------------
    ht_amplitude  : половина окна для rolling min/max (default 2)
    ht_channel    : множитель ATR для канала (default 2.0)
    ht_atr_period : период ATR внутри HalfTrend (default 100)
    """

    def __init__(
        self,
        min_size_pct: float = 7.0,
        min_bars_in: int = 10,
        rr_ratio: float = 2.5,
        ht_amplitude: int = 2,
        ht_channel: float = 2.0,
        ht_atr_period: int = 100,
        **kwargs,
    ):
        super().__init__(
            min_size_pct=min_size_pct,
            min_bars_in=min_bars_in,
            rr_ratio=rr_ratio,
            **kwargs,
        )
        self.ht_amplitude  = ht_amplitude
        self.ht_channel    = ht_channel
        self.ht_atr_period = ht_atr_period

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        sig = super().generate_signals(df, aux).copy()

        ht_dir, _ = halftrend(
            df,
            amplitude=self.ht_amplitude,
            channel_dev=self.ht_channel,
            atr_period=self.ht_atr_period,
        )

        # Фильтр: оставляем только сигналы совпадающие с направлением HalfTrend
        sig[(sig == 1)  & (ht_dir != 1)]  = 0
        sig[(sig == -1) & (ht_dir != -1)] = 0

        return sig


if __name__ == "__main__":
    SYMBOL  = "BTC/USDT"
    SINCE   = "2020-01-01"
    TF      = "4h"
    CAPITAL = 10_000.0
    RISK    = 0.06   # 6% risk per trade → DD ~25%

    print(f"  Загружаю {SYMBOL} {TF} …")
    df = get_ohlcv(SYMBOL, TF, since=SINCE)
    print(f"  {len(df):,} свечей  ({df.index[0].date()} → {df.index[-1].date()})")

    # Baseline: без фильтра
    base = BreakBar(min_size_pct=7.0, min_bars_in=10, rr_ratio=2.5)
    res_base = run_mm_backtest(df, base, initial_capital=CAPITAL,
                               risk_per_trade=RISK, symbol=SYMBOL, timeframe=TF)

    # С HalfTrend
    strat = BreakBarHalfTrend(
        min_size_pct=7.0, min_bars_in=10, rr_ratio=2.5,
        ht_amplitude=2, ht_channel=2.0, ht_atr_period=100,
    )
    res_ht = run_mm_backtest(df, strat, initial_capital=CAPITAL,
                             risk_per_trade=RISK, symbol=SYMBOL, timeframe=TF)

    # Сравнение
    print(f"\n{'═'*58}")
    print(f"  {'Метрика':<28} {'Без фильтра':>12} {'HalfTrend':>12}")
    print(f"  {'-'*56}")
    keys = [
        ("total_return_pct",  "Return %"),
        ("cagr_pct",          "CAGR %"),
        ("max_drawdown_pct",  "Max DD %"),
        ("win_rate_pct",      "Win Rate %"),
        ("profit_factor",     "Profit Factor"),
        ("sharpe_ratio",      "Sharpe"),
        ("total_trades",      "Trades"),
    ]
    for k, label in keys:
        v1 = res_base.metrics.get(k, "—")
        v2 = res_ht.metrics.get(k, "—")
        print(f"  {label:<28} {str(v1):>12} {str(v2):>12}")
    print(f"{'═'*58}")

    # HalfTrend line для графика
    ht_dir, ht_line = halftrend(df, amplitude=2, channel_dev=2.0, atr_period=100)

    html = "breakbar_halftrend.html"
    plot_backtest(
        res_ht, df,
        indicators={"HalfTrend": ht_line},
        title=(f"{SYMBOL} {TF} — BreakBar + HalfTrend filter  "
               f"min_size=7%  bars_in=10  RR=2.5  risk=6% | {SINCE}→"),
        save_html=html,
        show=False,
    )
    print(f"\n  Chart → {html}")

    import subprocess
    subprocess.Popen(["open", "-a", "Google Chrome", html])
