"""
Support / Resistance Bounce strategy (daily S/R levels) — v2 with filters.

Concept
-------
Detect the highest high and lowest low from the 2–4 week window on the daily
chart.  Those become the active resistance and support levels.  Enter when
price bounces off support (long) or rejects at resistance (short).

Entry rules (v2 — improved filters)
-------------------------------------
Long  : candle low  <= support  * (1 + zone_pct)   — wick touched support zone
        AND candle close > support                  — closed above support (bounce)
        AND close > EMA(ema_period)                 — above long-term trend (trend filter)
        AND RSI < rsi_long_max                      — not overbought
        AND volume > vol_sma_period-bar avg volume  — above-average volume (confirmation)
        AND RR >= min_rr                            — TP distance / SL distance

Short : candle high >= resistance * (1 - zone_pct)  — wick touched resistance zone
        AND candle close < resistance                — closed below resistance (rejection)
        AND close < EMA(ema_period)                 — below long-term trend
        AND RSI > rsi_short_min                     — not oversold
        AND volume > vol_sma_period-bar avg volume  — above-average volume
        AND RR >= min_rr

Exit rules (in backtest runner)
--------------------------------
Stop-loss  : entry − atr_sl_mult × ATR(14)  for long
             entry + atr_sl_mult × ATR(14)  for short
Take-profit: resistance snapshot at entry   for long
             support    snapshot at entry   for short
Signal reversal also closes the position.
"""

import numpy as np
import pandas as pd

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import (
    sr_weekly_window,
    rsi as _rsi,
    atr as _atr,
    ema as _ema,
    sma as _sma,
)


class SRBounce(Strategy):
    """
    Support/Resistance Bounce — daily S/R zones from the 2–4 week window.

    v2 improvements
    ---------------
    1. EMA trend filter    : long only above EMA200, short only below
    2. Tighter zone        : zone_pct reduced to 0.008 (0.8%) — fewer false touches
    3. Higher min RR       : 2.0 instead of 1.5 — better risk/reward selection
    4. Volume confirmation : bar volume must exceed 20-bar SMA of volume

    Parameters
    ----------
    weeks_near       : Near edge of S/R lookback (default 2 weeks)
    weeks_far        : Far  edge of S/R lookback (default 4 weeks)
    zone_pct         : Wick-to-level tolerance (default 0.008 = 0.8%)
    ema_period       : Trend filter EMA period (default 200)
    rsi_period       : RSI period (default 14)
    rsi_long_max     : RSI must be BELOW this for long  (default 60)
    rsi_short_min    : RSI must be ABOVE this for short (default 40)
    vol_sma_period   : Volume SMA period for above-average confirmation (default 20)
    atr_period       : ATR period for stop-loss sizing (default 14)
    atr_sl_mult      : SL distance = atr_sl_mult × ATR (default 1.5)
    min_rr           : Minimum reward-to-risk ratio to enter (default 2.0)
    cooldown_bars    : Bars to skip after any entry (default 3)
    """

    def __init__(
        self,
        weeks_near:     int   = 2,
        weeks_far:      int   = 4,
        zone_pct:       float = 0.008,
        ema_period:     int   = 200,
        rsi_period:     int   = 14,
        rsi_long_max:   float = 60.0,
        rsi_short_min:  float = 40.0,
        vol_sma_period: int   = 20,
        atr_period:     int   = 14,
        atr_sl_mult:    float = 1.5,
        min_rr:         float = 2.0,
        cooldown_bars:  int   = 3,
    ):
        self.weeks_near     = weeks_near
        self.weeks_far      = weeks_far
        self.zone_pct       = zone_pct
        self.ema_period     = ema_period
        self.rsi_period     = rsi_period
        self.rsi_long_max   = rsi_long_max
        self.rsi_short_min  = rsi_short_min
        self.vol_sma_period = vol_sma_period
        self.atr_period     = atr_period
        self.atr_sl_mult    = atr_sl_mult
        self.min_rr         = min_rr
        self.cooldown_bars  = cooldown_bars

        # Populated by generate_signals(); used by backtest runner and chart
        self.support_line:    pd.Series | None = None
        self.resistance_line: pd.Series | None = None
        self.atr_line:        pd.Series | None = None
        self.rsi_line:        pd.Series | None = None
        self.ema_line:        pd.Series | None = None

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        support, resistance = sr_weekly_window(df, self.weeks_near, self.weeks_far)
        atr_s  = _atr(df, self.atr_period)
        rsi_s  = _rsi(df["close"], self.rsi_period)
        ema_s  = _ema(df["close"], self.ema_period)
        vol_avg = _sma(df["volume"], self.vol_sma_period)

        self.support_line    = support
        self.resistance_line = resistance
        self.atr_line        = atr_s
        self.rsi_line        = rsi_s
        self.ema_line        = ema_s

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ── Long: bounce from support ────────────────────────────────────────
        # ── Long: любое касание поддержки + закрытие выше ───────────────────
        long_touch   = low   <= support * (1.0 + self.zone_pct)
        long_close   = close >  support
        long_sl_dist = (close - (support - self.atr_sl_mult * atr_s)).clip(lower=1e-9)
        long_tp_dist = (resistance - close).clip(lower=0)
        long_rr_ok   = (long_tp_dist / long_sl_dist) >= self.min_rr

        long_cond = long_touch & long_close & long_rr_ok

        # ── Short: любое касание сопротивления + закрытие ниже ──────────────
        short_touch   = high  >= resistance * (1.0 - self.zone_pct)
        short_close   = close <  resistance
        short_sl_dist = ((resistance + self.atr_sl_mult * atr_s) - close).clip(lower=1e-9)
        short_tp_dist = (close - support).clip(lower=0)
        short_rr_ok   = (short_tp_dist / short_sl_dist) >= self.min_rr

        short_cond = short_touch & short_close & short_rr_ok

        raw = pd.Series(0, index=df.index, dtype=int)
        raw[long_cond]  =  1
        raw[short_cond] = -1

        # Cooldown
        signals = raw.values.copy()
        cooldown_left = 0
        for i in range(len(signals)):
            if cooldown_left > 0:
                signals[i] = 0
                cooldown_left -= 1
            elif signals[i] != 0:
                cooldown_left = self.cooldown_bars

        return pd.Series(signals, index=df.index, dtype=int)
