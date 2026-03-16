"""
LiquidationBounce — contrarian strategy.

Logic (mean reversion after liquidation cascade):
  long  liq spike (longs wiped = price dumped hard) → go LONG  (buy the dip)
  short liq spike (shorts wiped = price pumped hard) → go SHORT (sell the rip)

Rationale:
  After a forced liquidation cascade, most of the move is already in.
  The market is oversold/overbought short-term → bounce is likely.
  This is the opposite of LiquidationSpike (trend-following).

Uses aux["liq"] with columns liq_long, liq_short in millions USD (Coinalyze).
"""

import pandas as pd
import numpy as np

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import sma, rsi


class LiquidationBounce(Strategy):
    """
    Parameters
    ----------
    long_liq_threshold  : long liq spike ($M) → LONG entry (catch the bounce)
    short_liq_threshold : short liq spike ($M) → SHORT entry (fade the squeeze)
    hold_bars           : bars to hold after entry
    zscore_mode         : use z-score instead of absolute threshold
    zscore_window       : rolling window for z-score baseline
    zscore_threshold    : z-score cutoff to trigger signal
    rsi_filter          : only enter if RSI confirms oversold/overbought
    rsi_period          : RSI period
    rsi_oversold        : RSI threshold for LONG entry (e.g. 40 = enter only if RSI < 40)
    rsi_overbought      : RSI threshold for SHORT entry (e.g. 60)
    min_spike_pct       : min abs bar move (%) to confirm spike, 0 = disabled
    trend_filter        : only trade in macro direction — LONG only above MA, SHORT only below MA
                          prevents catching falling knives in downtrends
    trend_period        : MA period for trend filter (default 48 = 2 days on 1h)
    """

    def __init__(
        self,
        long_liq_threshold:  float = 50,
        short_liq_threshold: float = 50,
        hold_bars:           int   = 3,
        zscore_mode:         bool  = False,
        zscore_window:       int   = 168,
        zscore_threshold:    float = 2.0,
        rsi_filter:          bool  = False,
        rsi_period:          int   = 14,
        rsi_oversold:        float = 40,
        rsi_overbought:      float = 60,
        min_spike_pct:       float = 0.0,
        trend_filter:        bool  = False,
        trend_period:        int   = 48,
    ):
        self.long_liq_threshold  = long_liq_threshold
        self.short_liq_threshold = short_liq_threshold
        self.hold_bars           = hold_bars
        self.zscore_mode         = zscore_mode
        self.zscore_window       = zscore_window
        self.zscore_threshold    = zscore_threshold
        self.rsi_filter          = rsi_filter
        self.rsi_period          = rsi_period
        self.rsi_oversold        = rsi_oversold
        self.rsi_overbought      = rsi_overbought
        self.min_spike_pct       = min_spike_pct
        self.trend_filter        = trend_filter
        self.trend_period        = trend_period

    def generate_signals(self, df: pd.DataFrame, aux: dict | None = None) -> pd.Series:
        if aux is None or "liq" not in aux:
            raise ValueError("LiquidationBounce requires aux={'liq': liquidations_df}")

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

        # Require meaningful price move on spike bar to confirm real cascade
        if self.min_spike_pct > 0:
            bar_move = (df["close"] / df["open"] - 1).abs()
            confirmed = bar_move > (self.min_spike_pct / 100)
            long_spike  = long_spike  & confirmed
            short_spike = short_spike & confirmed

        # CONTRARIAN:
        # big long liq  (price dumped) → go LONG  (bounce expected)
        # big short liq (price pumped) → go SHORT (reversal expected)
        buy_signal  = long_spike
        sell_signal = short_spike

        # Optional RSI confirmation: don't buy if already recovering,
        # don't short if already reversing
        if self.rsi_filter:
            r = rsi(df["close"], self.rsi_period)
            buy_signal  = buy_signal  & (r < self.rsi_oversold)
            sell_signal = sell_signal & (r > self.rsi_overbought)

        # Trend filter: only trade in macro direction
        # Avoids catching falling knives (long) in downtrend
        # or fading legitimate pumps in an uptrend
        if self.trend_filter:
            ma = sma(df["close"], self.trend_period)
            above_ma = df["close"] > ma
            buy_signal  = buy_signal  & above_ma    # only long if above MA
            sell_signal = sell_signal & ~above_ma   # only short if below MA

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
