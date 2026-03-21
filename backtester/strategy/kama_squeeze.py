"""
KAMA + Squeeze Momentum strategy (v2).

Entry filters (all must align):
  - Squeeze just released (squeezed >= min_squeeze_bars, then expands)
  - Histogram lime  (positive & rising) for long  → price above KAMA
  - Histogram red   (negative & falling) for short → price below KAMA
  - KAMA slope: KAMA rising for long, falling for short (compare N bars back)

Exit:
  - Histogram crosses zero line (neutral zone reached)
  - Reverse signal flips position directly
"""

import pandas as pd
import numpy as np

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import kama, squeeze_momentum


class KamaSqueezeStrategy(Strategy):
    """
    Parameters
    ----------
    kama_period       : KAMA look-back period          (default 21)
    kama_slope_bars   : bars back to measure KAMA slope (default 3)
    sq_length         : Squeeze / BB / KC period        (default 20)
    sq_mult_bb        : BB std-dev multiplier           (default 2.0)
    sq_mult_kc        : KC ATR multiplier               (default 1.5)
    sq_mom            : Momentum linreg period          (default 12)
    min_squeeze_bars  : Minimum bars squeeze must hold  (default 3)
    """

    def __init__(
        self,
        kama_period:      int   = 21,
        kama_slope_bars:  int   = 3,
        sq_length:        int   = 20,
        sq_mult_bb:       float = 2.0,
        sq_mult_kc:       float = 1.5,
        sq_mom:           int   = 12,
        min_squeeze_bars: int   = 3,
    ):
        self.kama_period      = kama_period
        self.kama_slope_bars  = kama_slope_bars
        self.sq_length        = sq_length
        self.sq_mult_bb       = sq_mult_bb
        self.sq_mult_kc       = sq_mult_kc
        self.sq_mom           = sq_mom
        self.min_squeeze_bars = min_squeeze_bars

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        close = df["close"]

        # ── Indicators ────────────────────────────────────────────────────────
        kama_line = kama(close, period=self.kama_period)
        squeeze_on, histogram, color = squeeze_momentum(
            df,
            length=self.sq_length,
            mult_bb=self.sq_mult_bb,
            mult_kc=self.sq_mult_kc,
            length_mom=self.sq_mom,
        )

        # KAMA slope: rising / falling
        kama_prev = kama_line.shift(self.kama_slope_bars)
        kama_rising  = kama_line > kama_prev
        kama_falling = kama_line < kama_prev

        # Count consecutive squeeze bars (rolling sum of squeeze_on)
        squeeze_bars = squeeze_on.astype(int).rolling(
            self.min_squeeze_bars, min_periods=1
        ).sum()
        # Squeeze that lasted long enough, just released this bar
        valid_release = (
            (~squeeze_on)
            & (squeeze_on.shift(1).fillna(False))
            & (squeeze_bars.shift(1).fillna(0) >= self.min_squeeze_bars)
        )

        # ── Entry conditions ──────────────────────────────────────────────────
        long_entry  = (
            valid_release
            & (color == "lime")
            & (close > kama_line)
            & kama_rising
        )
        short_entry = (
            valid_release
            & (color == "red")
            & (close < kama_line)
            & kama_falling
        )

        # ── Exit: histogram crosses zero ──────────────────────────────────────
        hist_prev    = histogram.shift(1)
        long_exit    = (hist_prev > 0) & (histogram <= 0)   # positive → zero/negative
        short_exit   = (hist_prev < 0) & (histogram >= 0)   # negative → zero/positive

        # ── Build position series bar-by-bar ──────────────────────────────────
        signal   = pd.Series(0, index=df.index, dtype=int)
        position = 0

        for i in range(len(df)):
            if position == 0:
                if long_entry.iloc[i]:
                    position = 1
                elif short_entry.iloc[i]:
                    position = -1
            elif position == 1:
                if long_exit.iloc[i]:
                    position = 0
                    if short_entry.iloc[i]:
                        position = -1
            elif position == -1:
                if short_exit.iloc[i]:
                    position = 0
                    if long_entry.iloc[i]:
                        position = 1
            signal.iloc[i] = position

        return signal

    def get_raw_data(self, df: pd.DataFrame) -> tuple:
        """Return raw indicator arrays for use in advanced simulations."""
        close = df["close"]
        kama_line = kama(close, period=self.kama_period)
        squeeze_on, histogram, color = squeeze_momentum(
            df,
            length=self.sq_length,
            mult_bb=self.sq_mult_bb,
            mult_kc=self.sq_mult_kc,
            length_mom=self.sq_mom,
        )
        # Precompute squeeze consecutive-bar count
        squeeze_bars = squeeze_on.astype(int).rolling(
            self.min_squeeze_bars, min_periods=1
        ).sum()
        # valid_release: squeeze just ended after enough bars
        valid_release = (
            (~squeeze_on)
            & (squeeze_on.shift(1).fillna(False))
            & (squeeze_bars.shift(1).fillna(0) >= self.min_squeeze_bars)
        )
        kama_prev = kama_line.shift(self.kama_slope_bars)
        kama_rising  = kama_line > kama_prev
        kama_falling = kama_line < kama_prev
        return kama_line, squeeze_on, histogram, color, valid_release, kama_rising, kama_falling

    def get_indicators(self, df: pd.DataFrame) -> dict:
        close     = df["close"]
        kama_line = kama(close, period=self.kama_period)
        _, histogram, _ = squeeze_momentum(
            df,
            length=self.sq_length,
            mult_bb=self.sq_mult_bb,
            mult_kc=self.sq_mult_kc,
            length_mom=self.sq_mom,
        )
        return {
            f"KAMA {self.kama_period}": kama_line,
            "Squeeze Hist":             histogram,
        }
