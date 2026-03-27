"""
BreakBar strategy — port of the BreakBar.mq4 indicator.

Logic
-----
1. Find an "impulse" bar: High-Low >= min_bar_size  AND  |Open-Close| >= min_bar_size/10
2. Define a zone: Level_H = High + dist_hl,  Level_L = Low - dist_hl
3. Count subsequent bars whose BODY (max/min of Open,Close) stays fully inside the zone.
4. Once >= min_bars_in consolidation bars are counted, the first bar whose body
   exits the zone is the signal:
     - Body breaks UP  (max(O,C) > Level_H) → long  (+1)
     - Body breaks DOWN (min(O,C) < Level_L) → short (-1)
5. If zone_in_zone=True  : outer scan continues after a signal (nested zones allowed).
   If zone_in_zone=False : outer scan jumps to the breakout bar (no nesting).

Settings from screenshot
------------------------
min_bar_size = 30   (price units, e.g. $30 for BTC/USDT)
dist_hl      = 0
min_bars_in  = 5
zone_in_zone = True
"""

import numpy as np
import pandas as pd

from backtester.strategy.base import Strategy


class BreakBar(Strategy):
    """
    Parameters
    ----------
    min_bar_size : float
        Minimum High-Low range of the impulse bar in price units (MinSizeBarHL).
        For BTC/USDT this is USD (e.g. 30 = $30 minimum range).
    dist_hl : float
        Extra distance added to High / subtracted from Low when defining the zone (DistHL).
    min_bars_in : int
        Minimum number of consolidation bars required inside the zone before a
        breakout signal is valid (MinBarIn).
    zone_in_zone : bool
        If True, the scanner continues after a signal (nested zones allowed).
        If False, the scanner jumps to the breakout bar after each signal.
    swing_period : int
        Lookback period for swing-high / swing-low (used as fallback SL).
    rr_ratio : float
        Risk/reward ratio for take-profit calculation.
    """

    def __init__(
        self,
        min_size_pct: float = 2.0,   # min (High-Low)/Close in % — replaces fixed min_bar_size
        dist_hl: float = 0.0,
        min_bars_in: int = 5,
        zone_in_zone: bool = True,
        swing_period: int = 20,
        rr_ratio: float = 2.0,
    ):
        self.min_size_pct = min_size_pct
        self.dist_hl = dist_hl
        self.min_bars_in = min_bars_in
        self.zone_in_zone = zone_in_zone
        self.swing_period = swing_period
        self.rr_ratio = rr_ratio

    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        n = len(df)
        high  = df["high"].values
        low   = df["low"].values
        op    = df["open"].values
        close = df["close"].values

        signals  = np.zeros(n, dtype=int)
        sl_array = np.full(n, np.nan)   # zone-boundary SL for each signal bar

        i = 0
        while i < n - 1:
            # ── Step 1: is bar i an impulse bar? ──────────────────────────
            min_bar_size = self.min_size_pct / 100.0 * close[i]
            min_body     = min_bar_size / 10.0
            if (high[i] - low[i]) >= min_bar_size and \
               abs(op[i] - close[i]) >= min_body:

                level_h = high[i] + self.dist_hl
                level_l = low[i]  - self.dist_hl

                cnt = 0
                j   = i + 1

                # ── Step 2: scan forward for consolidation + breakout ──
                while j < n:
                    body_high = max(op[j], close[j])
                    body_low  = min(op[j], close[j])
                    in_zone   = (body_high <= level_h) and (body_low >= level_l)

                    if in_zone:
                        cnt += 1
                    else:
                        # Bar exits zone
                        if cnt < self.min_bars_in:
                            break   # not enough consolidation — discard this impulse bar

                        # Sufficient consolidation: this exit bar is the breakout
                        res = 0
                        if body_high > level_h:
                            res = 1
                        if body_low < level_l:
                            res = -1

                        if res != 0:
                            if signals[j] == 0:   # don't overwrite an earlier signal
                                signals[j]  = res
                                # SL: opposite side of the zone
                                sl_array[j] = level_l if res == 1 else level_h

                            if not self.zone_in_zone:
                                i = j - 1  # outer loop will do i += 1 → starts at j
                            break   # done with this impulse bar

                    j += 1

            i += 1

        # Store zone SL for use by the backtest runner
        self._sl_zone  = sl_array

        return pd.Series(signals, index=df.index)

    # ------------------------------------------------------------------
    def zone_sl_levels(self, df: pd.DataFrame) -> pd.Series:
        """
        Zone-boundary stop-loss levels, one per bar.
        Call generate_signals() first (or let the backtest runner call it).
        For long signals  : SL = Level_L (bottom of the breakout zone).
        For short signals : SL = Level_H (top of the breakout zone).
        NaN where there is no signal.
        """
        if not hasattr(self, "_sl_zone"):
            self.generate_signals(df)
        return pd.Series(self._sl_zone, index=df.index)

    # ------------------------------------------------------------------
    def swing_levels(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Fallback swing-low / swing-high for SL when zone SL is unavailable."""
        sw_low  = df["low"].rolling(self.swing_period,  min_periods=1).min().shift(1)
        sw_high = df["high"].rolling(self.swing_period, min_periods=1).max().shift(1)
        return sw_low, sw_high
