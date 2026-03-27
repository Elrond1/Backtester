"""
SAR + EMA 200 + MACD strategy with rebound entry.

Entry conditions (pattern bar)
--------------------------------
Long  : ohlc4 > EMA200  AND  SAR < low  AND  MACD histogram crosses 0 upward
Short : ohlc4 < EMA200  AND  SAR > high AND  MACD histogram crosses 0 downward

Entry confirmation
------------------
After the pattern bar, wait for `rebound_candles` bars:
  Long  : all rebound bars must be bearish (close < open) — pullback before entry
  Short : all rebound bars must be bullish (close > open) — bounce before entry
Entry signal is emitted at the last rebound bar.

Exit
----
Stop-loss  : nearest swing low (long) or swing high (short) — rolling min/max
             over `swing_period` bars, no look-ahead.
Take-profit: rr_ratio × risk from SL.
"""

import pandas as pd

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import ema, macd, parabolic_sar


class SarEmaMacd(Strategy):
    """
    Parameters
    ----------
    ema_period      : Trend filter EMA period (default 200)
    sar_start       : SAR acceleration factor start (default 0.02)
    sar_increment   : SAR AF increment (default 0.02)
    sar_maximum     : SAR AF maximum (default 0.2)
    macd_fast       : MACD fast EMA (default 12)
    macd_slow       : MACD slow EMA (default 26)
    macd_signal     : MACD signal EMA (default 9)
    rebound_candles : Bars of pullback/bounce required before entry (default 3)
    swing_period    : Lookback bars for SL swing level (default 20)
    rr_ratio        : Risk/reward ratio for TP (default 2.0)
    """

    def __init__(
        self,
        ema_period: int = 200,
        sar_start: float = 0.02,
        sar_increment: float = 0.02,
        sar_maximum: float = 0.2,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rebound_candles: int = 3,
        swing_period: int = 20,
        rr_ratio: float = 2.0,
    ):
        self.ema_period = ema_period
        self.sar_start = sar_start
        self.sar_increment = sar_increment
        self.sar_maximum = sar_maximum
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rebound_candles = rebound_candles
        self.swing_period = swing_period
        self.rr_ratio = rr_ratio

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        close = df["close"]
        ohlc4 = (df["open"] + df["high"] + df["low"] + close) / 4

        ema200 = ema(close, self.ema_period)
        sar    = parabolic_sar(df, self.sar_start, self.sar_increment, self.sar_maximum)
        _, _, hist = macd(close, self.macd_fast, self.macd_slow, self.macd_signal)

        hist_prev = hist.shift(1)

        # Base pattern conditions
        base_buy  = (ohlc4 > ema200) & (sar < df["low"])  & (hist_prev < 0) & (hist > 0)
        base_sell = (ohlc4 < ema200) & (sar > df["high"]) & (hist_prev > 0) & (hist < 0)

        # Rebound bars: N consecutive candles after pattern in opposite direction
        N       = self.rebound_candles
        bearish = (close < df["open"])
        bullish = (close > df["open"])

        # rolling(N).min() == 1 means all N bars satisfy the condition
        all_bearish = bearish.astype(int).rolling(N, min_periods=N).min().astype(bool)
        all_bullish = bullish.astype(int).rolling(N, min_periods=N).min().astype(bool)

        # Pattern was N bars ago; rebound covers the N bars ending at current bar
        base_buy_ago  = base_buy.shift(N).fillna(False)
        base_sell_ago = base_sell.shift(N).fillna(False)

        buy_entry  = base_buy_ago  & all_bearish
        sell_entry = base_sell_ago & all_bullish

        signals = pd.Series(0, index=df.index)
        signals[buy_entry]  =  1
        signals[sell_entry] = -1
        return signals

    def swing_levels(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        sw_low  = df["low"].rolling(self.swing_period,  min_periods=1).min().shift(1)
        sw_high = df["high"].rolling(self.swing_period, min_periods=1).max().shift(1)
        return sw_low, sw_high
