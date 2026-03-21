"""
EMA 200 + RSI 14 + MACD (12, 26, 9) strategy.

Entry conditions
----------------
Long  : Close > EMA200  AND  RSI crosses 30 from below  AND  MACD histogram rising
Short : Close < EMA200  AND  RSI crosses 70 from above  AND  MACD histogram falling

Exit
----
Stop-loss  : nearest local swing low (long) or swing high (short) — rolling min/max
             over `swing_period` bars, computed without look-ahead.
Take-profit: 1 : rr_ratio risk/reward (default 1:2).
"""

import pandas as pd

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import ema, rsi, macd


class EmaRsiMacd(Strategy):
    """
    Parameters
    ----------
    ema_period     : Trend filter EMA period (default 200)
    rsi_period     : RSI period (default 14)
    rsi_oversold   : RSI oversold threshold (default 30)
    rsi_overbought : RSI overbought threshold (default 70)
    macd_fast      : MACD fast EMA (default 12)
    macd_slow      : MACD slow EMA (default 26)
    macd_signal    : MACD signal EMA (default 9)
    swing_period   : Lookback bars for local swing low/high SL (default 20)
    rr_ratio       : Risk/reward ratio for TP (default 2.0)
    """

    def __init__(
        self,
        ema_period: int = 200,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        swing_period: int = 20,
        rr_ratio: float = 2.0,
    ):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.swing_period = swing_period
        self.rr_ratio = rr_ratio

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        close = df["close"]

        ema200 = ema(close, self.ema_period)
        r = rsi(close, self.rsi_period)
        _, _, hist = macd(close, self.macd_fast, self.macd_slow, self.macd_signal)

        # RSI zone crossovers
        rsi_cross_up   = (r.shift(1) < self.rsi_oversold)  & (r >= self.rsi_oversold)
        rsi_cross_down = (r.shift(1) > self.rsi_overbought) & (r <= self.rsi_overbought)

        # MACD histogram direction
        macd_rising  = hist > hist.shift(1)
        macd_falling = hist < hist.shift(1)

        signals = pd.Series(0, index=df.index)
        signals[(close > ema200) & rsi_cross_up   & macd_rising]  = 1
        signals[(close < ema200) & rsi_cross_down & macd_falling] = -1
        return signals

    def swing_levels(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Per-bar swing low (SL for longs) and swing high (SL for shorts).
        Shifted by 1 to avoid look-ahead — safe to use at entry bar.
        """
        sw_low  = df["low"].rolling(self.swing_period, min_periods=1).min().shift(1)
        sw_high = df["high"].rolling(self.swing_period, min_periods=1).max().shift(1)
        return sw_low, sw_high
