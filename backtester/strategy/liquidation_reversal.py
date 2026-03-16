"""
LiquidationReversalSignals — Python port of the AlgoAlpha TradingView indicator.

Original idea (AlgoAlpha):
  Combines z-score analysis of directional volume with Supertrend to detect
  reversals driven by forced liquidation events.

Logic:
  1. Split volume into up-volume (bullish bars) and down-volume (bearish bars).
  2. Compute rolling z-score for each direction.
  3. Volume spike during the opposite Supertrend direction = likely forced liquidation:
       - Bearish Supertrend + up_vol spike  → shorts being squeezed → expect bullish reversal
       - Bullish Supertrend + down_vol spike → longs being liquidated → expect bearish reversal
  4. Set a "flag" lasting timeout_bars.
  5. When Supertrend flips within the flag window → confirmed reversal → entry signal.

Note: AlgoAlpha uses a lower timeframe for volume analysis. Here we use the chart
timeframe. Pass lower-TF OHLCV as aux["lower_tf"] to enable proper LTF volume split.
"""

import pandas as pd
import numpy as np

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import atr, sma, supertrend


class LiquidationReversalSignals(Strategy):
    """
    Parameters
    ----------
    zscore_length    : Rolling window for z-score normalisation
    zscore_threshold : Spike threshold (standard deviations above mean)
    timeout_bars     : How many bars after a spike the Supertrend flip is valid
    st_period        : Supertrend ATR period
    st_multiplier    : Supertrend ATR multiplier
    hold_bars        : Bars to hold after confirmed signal (0 = hold until next signal)
    macro_filter     : Only trade in macro trend direction (avoids fighting strong trends)
    macro_period     : SMA period for macro trend (default 200 bars)
    """

    def __init__(
        self,
        zscore_length:    int   = 20,
        zscore_threshold: float = 2.0,
        timeout_bars:     int   = 5,
        st_period:        int   = 10,
        st_multiplier:    float = 3.0,
        hold_bars:        int   = 0,
        macro_filter:     bool  = False,
        macro_period:     int   = 200,
    ):
        self.zscore_length    = zscore_length
        self.zscore_threshold = zscore_threshold
        self.timeout_bars     = timeout_bars
        self.st_period        = st_period
        self.st_multiplier    = st_multiplier
        self.hold_bars        = hold_bars
        self.macro_filter     = macro_filter
        self.macro_period     = macro_period

    def generate_signals(self, df: pd.DataFrame, aux: dict | None = None) -> pd.Series:
        # ── Directional volume ────────────────────────────────────────────────────
        # If aux["lower_tf"] is provided, aggregate lower-TF volume into main bars.
        # This is the key insight of AlgoAlpha: splitting 5m/15m volume is much
        # more precise for detecting forced liquidation events than 1h volume.
        if aux is not None and "lower_tf" in aux:
            ltf = aux["lower_tf"]
            is_bull_ltf   = ltf["close"] >= ltf["open"]
            ltf_up_vol    = ltf["volume"].where(is_bull_ltf,  0.0)
            ltf_down_vol  = ltf["volume"].where(~is_bull_ltf, 0.0)
            # Detect main TF resolution and resample
            tf_seconds = int(df.index.to_series().diff().median().total_seconds())
            tf_str     = f"{tf_seconds}s"
            up_vol   = ltf_up_vol.resample(tf_str).sum().reindex(df.index).fillna(0)
            down_vol = ltf_down_vol.resample(tf_str).sum().reindex(df.index).fillna(0)
        else:
            is_bull  = df["close"] >= df["open"]
            up_vol   = df["volume"].where(is_bull,  0.0)
            down_vol = df["volume"].where(~is_bull, 0.0)

        up_z   = self._zscore(up_vol,   self.zscore_length)
        down_z = self._zscore(down_vol, self.zscore_length)

        up_spike   = up_z   > self.zscore_threshold   # abnormal buying
        down_spike = down_z > self.zscore_threshold   # abnormal selling

        # ── Supertrend ────────────────────────────────────────────────────────────
        st_dir, _ = supertrend(df, self.st_period, self.st_multiplier)
        st_flip_bull = (st_dir == 1) & (st_dir.shift(1) == -1)   # flipped to bullish
        st_flip_bear = (st_dir == -1) & (st_dir.shift(1) == 1)   # flipped to bearish

        # ── Signal generation (bar-by-bar: flags expire) ──────────────────────────
        n               = len(df)
        st_dir_vals     = st_dir.values
        up_spike_vals   = up_spike.values
        down_spike_vals = down_spike.values
        flip_bull_vals  = st_flip_bull.values
        flip_bear_vals  = st_flip_bear.values
        raw             = np.zeros(n)

        long_ttl  = 0   # bars remaining for long reversal flag
        short_ttl = 0   # bars remaining for short reversal flag

        for i in range(1, n):
            # Set new flags: volume spike in the opposite Supertrend context
            if down_spike_vals[i] and st_dir_vals[i] == 1:
                # Big selling in an uptrend → longs liquidated → short reversal pending
                short_ttl = self.timeout_bars + 1

            if up_spike_vals[i] and st_dir_vals[i] == -1:
                # Big buying in a downtrend → shorts squeezed → long reversal pending
                long_ttl = self.timeout_bars + 1

            # Confirmed signal: Supertrend flips while flag is active
            if flip_bull_vals[i] and long_ttl > 0:
                raw[i] = 1.0
                long_ttl = 0   # consume flag

            if flip_bear_vals[i] and short_ttl > 0:
                raw[i] = -1.0
                short_ttl = 0  # consume flag

            # Countdown flags
            if long_ttl  > 0: long_ttl  -= 1
            if short_ttl > 0: short_ttl -= 1

        raw_series = pd.Series(raw, index=df.index)

        # Macro trend filter: only long above MA, only short below MA
        if self.macro_filter:
            macro_ma  = sma(df["close"], self.macro_period)
            above_ma  = df["close"] > macro_ma
            raw_series = raw_series.copy()
            raw_series[(raw_series == 1)  & ~above_ma] = 0
            raw_series[(raw_series == -1) &  above_ma] = 0

        if self.hold_bars > 0:
            return self._extend_signal(raw_series, self.hold_bars)
        return raw_series

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window, min_periods=max(5, window // 2)).mean()
        std  = series.rolling(window, min_periods=max(5, window // 2)).std()
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
