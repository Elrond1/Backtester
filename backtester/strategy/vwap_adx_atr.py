"""
Anchored VWAP + ADX + ATR strategy.

Entry rules
-----------
Long  : candle low  ≤ anchored VWAP and close > VWAP (bounce from above)
        AND ADX > adx_threshold (strong trend)
Short : candle high ≥ anchored VWAP and close < VWAP (rejection from below)
        AND ADX > adx_threshold

Exit rules (handled in the custom backtest runner)
-----------
Stop-loss  : 2 × ATR(14) from entry price
Take-profit: price closes on the opposite side of the anchored VWAP
ADX exit   : ADX drops below adx_exit_level (trend weakening)
Reversal   : opposite signal flips the position

Anchored VWAP anchors reset whenever price breaks to a new N-bar extreme
(no look-ahead bias — only past N bars are examined at each bar).
"""

import numpy as np
import pandas as pd

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import adx as _adx, atr as _atr


# ── Anchored VWAP ─────────────────────────────────────────────────────────────

def anchored_vwap(df: pd.DataFrame, swing_period: int = 20) -> pd.Series:
    """
    Rolling Anchored VWAP.

    The cumulative VWAP resets whenever price breaks out to a new N-bar
    swing low or swing high (only looking back, no future data).

    Parameters
    ----------
    df           : OHLCV DataFrame
    swing_period : Look-back window for detecting new extremes

    Returns
    -------
    pd.Series aligned with df.index
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume  = df["volume"]

    # A "new swing low" means current low < minimum of past N bars (shift avoids current bar)
    past_low  = df["low"].shift(1).rolling(swing_period, min_periods=1).min()
    past_high = df["high"].shift(1).rolling(swing_period, min_periods=1).max()

    is_anchor = (df["low"] < past_low) | (df["high"] > past_high)

    typical_arr = typical.values
    volume_arr  = volume.values
    anchor_arr  = is_anchor.values

    n           = len(df)
    vwap_vals   = np.empty(n)
    vwap_vals[:] = np.nan
    cum_tpv     = 0.0
    cum_vol     = 0.0

    for i in range(n):
        if anchor_arr[i] or cum_vol == 0.0:
            cum_tpv = 0.0
            cum_vol = 0.0
        cum_tpv += typical_arr[i] * volume_arr[i]
        cum_vol += volume_arr[i]
        if cum_vol > 0.0:
            vwap_vals[i] = cum_tpv / cum_vol

    return pd.Series(vwap_vals, index=df.index, name="anchored_vwap")


# ── Strategy ──────────────────────────────────────────────────────────────────

class VwapAdxAtr(Strategy):
    """
    Anchored VWAP + ADX(14) + ATR(14) trend-following strategy.

    Parameters
    ----------
    swing_period        : Bars to look back when detecting new swing extremes
                          (VWAP anchor reset interval). Larger = fewer resets = more
                          stable VWAP. Default 100 ≈ 4 days on 1h.
    adx_period          : ADX calculation period (default 14)
    adx_threshold       : Minimum ADX to confirm trend strength for entry (default 25)
    adx_exit_level      : ADX level below which we exit (trend weakening, default 20)
    atr_period          : ATR calculation period (default 14)
    atr_sl_mult         : Stop-loss distance in ATR multiples (default 2.0)
    cooldown_bars       : Minimum bars between any two entries (default 10)
    min_vwap_dist_pct   : Minimum distance from VWAP at bounce (as fraction of price).
                          Filters out signals too close to VWAP (noise). Default 0.002.
    """

    def __init__(
        self,
        swing_period:      int   = 100,
        adx_period:        int   = 14,
        adx_threshold:     float = 25.0,
        adx_exit_level:    float = 20.0,
        atr_period:        int   = 14,
        atr_sl_mult:       float = 2.0,
        cooldown_bars:     int   = 10,
        min_vwap_dist_pct: float = 0.002,
    ):
        self.swing_period      = swing_period
        self.adx_period        = adx_period
        self.adx_threshold     = adx_threshold
        self.adx_exit_level    = adx_exit_level
        self.atr_period        = atr_period
        self.atr_sl_mult       = atr_sl_mult
        self.cooldown_bars     = cooldown_bars
        self.min_vwap_dist_pct = min_vwap_dist_pct

        # Populated after generate_signals(); used by the custom backtest runner
        self.vwap_line: pd.Series | None = None
        self.adx_line:  pd.Series | None = None
        self.atr_line:  pd.Series | None = None

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        vwap_s = anchored_vwap(df, self.swing_period)
        adx_s  = _adx(df, self.adx_period)
        atr_s  = _atr(df, self.atr_period)

        self.vwap_line = vwap_s
        self.adx_line  = adx_s
        self.atr_line  = atr_s

        close   = df["close"]
        vwap_np = vwap_s.values

        # Distance of close from VWAP as fraction of VWAP
        dist_pct = (close - vwap_s).abs() / vwap_s.replace(0, np.nan)

        # Long: wick pierced VWAP (low <= vwap), closed above with minimum distance
        long_cond = (
            (df["low"]  <= vwap_s) &
            (df["close"] > vwap_s) &
            (dist_pct    > self.min_vwap_dist_pct) &
            (adx_s > self.adx_threshold)
        )

        # Short: wick pierced VWAP (high >= vwap), closed below with minimum distance
        short_cond = (
            (df["high"] >= vwap_s) &
            (df["close"] < vwap_s) &
            (dist_pct    > self.min_vwap_dist_pct) &
            (adx_s > self.adx_threshold)
        )

        raw = pd.Series(0, index=df.index, dtype=int)
        raw[long_cond]  =  1
        raw[short_cond] = -1

        # Apply cooldown: after any signal, suppress signals for cooldown_bars bars
        signals = raw.values.copy()
        cooldown_left = 0
        for i in range(len(signals)):
            if cooldown_left > 0:
                signals[i] = 0
                cooldown_left -= 1
            elif signals[i] != 0:
                cooldown_left = self.cooldown_bars

        return pd.Series(signals, index=df.index, dtype=int)
