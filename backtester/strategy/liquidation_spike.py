"""
LiquidationSpike strategy.

Logic (trend-following):
  short liq spike (shorts squeezed = price pumped)  → go LONG  (ride the squeeze)
  long  liq spike (longs  squeezed = price dumped)  → go SHORT (ride the dump)

Optional MA filter avoids trading against the macro trend:
  ma_filter=True → only long when price > MA, only short when price < MA

Uses aux["liq"] DataFrame with columns liq_long, liq_short (in millions USD).
"""

import pandas as pd
import numpy as np

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import sma


class LiquidationSpike(Strategy):
    """
    Parameters
    ----------
    long_liq_threshold  : long liq spike in $M → SHORT signal (longs got wiped)
    short_liq_threshold : short liq spike in $M → LONG  signal (shorts got wiped)
    hold_bars           : bars to hold after entry
    zscore_mode         : use z-score instead of absolute threshold
    zscore_window       : rolling window for z-score
    zscore_threshold    : z-score cutoff (e.g. 2.0)
    ma_filter           : only trade in direction of MA trend
    ma_period           : MA period for trend filter
    min_spike_pct       : min abs bar move (%) to confirm spike, 0 = disabled
    """

    def __init__(
        self,
        long_liq_threshold:  float = 50,
        short_liq_threshold: float = 50,
        hold_bars:     int   = 3,
        zscore_mode:   bool  = False,
        zscore_window: int   = 168,
        zscore_threshold: float = 2.0,
        ma_filter:     bool  = False,
        ma_period:     int   = 48,
        min_spike_pct: float = 0.0,
    ):
        self.long_liq_threshold  = long_liq_threshold
        self.short_liq_threshold = short_liq_threshold
        self.hold_bars           = hold_bars
        self.zscore_mode         = zscore_mode
        self.zscore_window       = zscore_window
        self.zscore_threshold    = zscore_threshold
        self.ma_filter           = ma_filter
        self.ma_period           = ma_period
        self.min_spike_pct       = min_spike_pct

    def generate_signals(self, df: pd.DataFrame, aux: dict | None = None) -> pd.Series:
        if aux is None or "liq" not in aux:
            raise ValueError("LiquidationSpike requires aux={'liq': liquidations_df}")

        liq = aux["liq"].reindex(df.index, method="ffill")
        long_vol  = liq["liq_long"].fillna(0)
        short_vol = liq["liq_short"].fillna(0)

        if self.zscore_mode:
            long_score  = self._zscore(long_vol,  self.zscore_window)
            short_score = self._zscore(short_vol, self.zscore_window)
            long_spike  = long_score  > self.zscore_threshold
            short_spike = short_score > self.zscore_threshold
        else:
            long_spike  = long_vol  > self.long_liq_threshold
            short_spike = short_vol > self.short_liq_threshold

        # Optional: require meaningful price move on the spike bar
        if self.min_spike_pct > 0:
            bar_move = (df["close"] / df["open"] - 1).abs()
            confirmed = bar_move > (self.min_spike_pct / 100)
            long_spike  = long_spike  & confirmed
            short_spike = short_spike & confirmed

        # short liq spike → ride pump → LONG
        # long  liq spike → ride dump → SHORT
        buy_signal  = short_spike
        sell_signal = long_spike

        # MA trend filter: only trade in macro direction
        if self.ma_filter:
            ma = sma(df["close"], self.ma_period)
            above_ma = df["close"] > ma
            buy_signal  = buy_signal  & above_ma
            sell_signal = sell_signal & ~above_ma

        raw = pd.Series(0.0, index=df.index)
        raw[buy_signal]  =  1.0
        raw[sell_signal] = -1.0

        return self._extend_signal(raw, self.hold_bars)

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window, min_periods=24).mean()
        std  = series.rolling(window, min_periods=24).std()
        return (series - mean) / std.replace(0, np.nan)

    @staticmethod
    def _extend_signal(raw: pd.Series, hold_bars: int) -> pd.Series:
        out = raw.values.copy().astype(float)
        last_val  = 0.0
        bars_held = 0
        for i in range(len(out)):
            if raw.values[i] != 0:
                last_val  = raw.values[i]
                bars_held = hold_bars
            if bars_held > 0:
                out[i]     = last_val
                bars_held -= 1
            else:
                out[i] = 0.0
        return pd.Series(out, index=raw.index)
