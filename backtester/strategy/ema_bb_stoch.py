"""
«Среднесрочный Трендолов» v3 — EMA 50/200 + Bollinger Bands (20, 2) + Stochastic (14, 3, 3)

Изменения v3 (диагноз: слишком много сигналов, шорты убыточны):
  1. bb_touch_bars=1  — только точное касание BB на текущей свече
  2. stoch_cross_bars=1 — точный кроссовер на текущей или предыдущей свече
  3. trend_bars=5  — устойчивый тренд минимум 5 свечей
  4. RSI фильтр: не перекупленность для лонгов (RSI < rsi_max_long)
  5. Stochastic должен быть растущим 2 бара подряд (%K > %K[1])
  6. long_only режим: шорты только при очень сильном нисходящем тренде
  7. Cooldown: пропуск сигналов после выхода из позиции (в симуляторе)

Entry Long:
  EMA50 > EMA200 последние trend_bars баров
  close <= lower_BB (тело свечи касается полосы)
  slow_%K пересекал уровень oversold снизу вверх (текущий или предыдущий бар)
  slow_%K > slow_%D  (бычье расположение)
  slow_%K растёт 2 бара подряд
  RSI < rsi_max_long

Entry Short (зеркально, только при long_only=False):
  EMA50 < EMA200 последние trend_bars баров
  close >= upper_BB
  slow_%K пересекал overbought сверху вниз
  slow_%K < slow_%D
  slow_%K падает 2 бара подряд
  RSI > rsi_min_short
"""

import pandas as pd
import numpy as np

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import (
    ema as _ema,
    bollinger_bands,
    stochastic,
    atr as _atr,
    rsi as _rsi,
    sma,
)


class EmaBbStoch(Strategy):
    """
    Parameters
    ----------
    ema_fast         : Fast EMA period (default 50)
    ema_slow         : Slow EMA period (default 200)
    bb_period        : Bollinger Bands period (default 20)
    bb_std           : Bollinger Bands std-dev (default 2.0)
    stoch_k          : Stochastic raw %K lookback (default 14)
    stoch_smooth     : Smoothing for Slow %K (default 3)
    stoch_d          : %D period (default 3)
    stoch_oversold   : Oversold threshold (default 20)
    stoch_overbought : Overbought threshold (default 80)
    trend_bars       : Bars EMA50 must consistently cross EMA200 (default 5)
    rsi_period       : RSI period for momentum filter (default 14)
    rsi_max_long     : Max RSI for long entry (default 60)
    rsi_min_short    : Min RSI for short entry (default 40)
    long_only        : Only trade longs if True (default False)
    swing_period     : Bars for rolling swing SL (default 10)
    atr_period       : ATR period (default 14)
    atr_sl_mult      : Min SL = atr_sl_mult × ATR (default 1.0)
    rr_ratio         : Risk/reward for TP (default 2.0)
    risk_pct         : Risk per trade fraction (default 0.015)
    """

    def __init__(
        self,
        ema_fast: int = 50,
        ema_slow: int = 200,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 14,
        stoch_smooth: int = 3,
        stoch_d: int = 3,
        stoch_oversold: int = 20,
        stoch_overbought: int = 80,
        trend_bars: int = 5,
        rsi_period: int = 14,
        rsi_max_long: int = 60,
        rsi_min_short: int = 40,
        long_only: bool = False,
        swing_period: int = 10,
        atr_period: int = 14,
        atr_sl_mult: float = 1.0,
        rr_ratio: float = 2.0,
        risk_pct: float = 0.015,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stoch_k = stoch_k
        self.stoch_smooth = stoch_smooth
        self.stoch_d = stoch_d
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.trend_bars = trend_bars
        self.rsi_period = rsi_period
        self.rsi_max_long = rsi_max_long
        self.rsi_min_short = rsi_min_short
        self.long_only = long_only
        self.swing_period = swing_period
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio = rr_ratio
        self.risk_pct = risk_pct

    # ── Indicators ────────────────────────────────────────────────────────────

    def calc_indicators(self, df: pd.DataFrame):
        close = df["close"]

        ema50  = _ema(close, self.ema_fast)
        ema200 = _ema(close, self.ema_slow)
        upper, middle, lower = bollinger_bands(close, self.bb_period, self.bb_std)

        raw_k, _ = stochastic(df, self.stoch_k, self.stoch_smooth)
        slow_k   = sma(raw_k, self.stoch_smooth)
        slow_d   = sma(slow_k, self.stoch_d)

        rsi_vals = _rsi(close, self.rsi_period)
        atr_vals = _atr(df, self.atr_period)

        return ema50, ema200, upper, middle, lower, slow_k, slow_d, rsi_vals, atr_vals

    # ── Signals ───────────────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        close = df["close"]
        ema50, ema200, upper, _, lower, slow_k, slow_d, rsi_vals, _ = \
            self.calc_indicators(df)

        # 1. Устойчивый тренд минимум trend_bars свечей
        trend_up   = (ema50 > ema200).rolling(self.trend_bars).min().astype(bool)
        trend_down = (ema50 < ema200).rolling(self.trend_bars).min().astype(bool)

        # 2. Касание BB телом свечи (close, не тенью)
        bb_lower_touch = close <= lower
        bb_upper_touch = close >= upper

        # 3. Stoch кроссовер: был ниже уровня на прошлом баре, выше на текущем
        k_cross_up   = (slow_k.shift(1) < self.stoch_oversold)  & (slow_k >= self.stoch_oversold)
        k_cross_down = (slow_k.shift(1) > self.stoch_overbought) & (slow_k <= self.stoch_overbought)

        # Допускаем также кроссовер на предыдущем баре (задержка 1)
        k_cross_up_recent   = k_cross_up | k_cross_up.shift(1).fillna(False)
        k_cross_down_recent = k_cross_down | k_cross_down.shift(1).fillna(False)

        # 4. %K > %D (бычье/медвежье расположение)
        k_above_d = slow_k > slow_d
        k_below_d = slow_k < slow_d

        # 5. %K растёт / падает 2 бара подряд
        k_rising  = (slow_k > slow_k.shift(1)) & (slow_k.shift(1) > slow_k.shift(2))
        k_falling = (slow_k < slow_k.shift(1)) & (slow_k.shift(1) < slow_k.shift(2))

        # 6. RSI фильтр
        rsi_ok_long  = rsi_vals < self.rsi_max_long
        rsi_ok_short = rsi_vals > self.rsi_min_short

        signals = pd.Series(0, index=df.index)

        long_cond = (
            trend_up
            & bb_lower_touch
            & k_cross_up_recent
            & k_above_d
            & k_rising
            & rsi_ok_long
        )
        signals[long_cond] = 1

        if not self.long_only:
            short_cond = (
                trend_down
                & bb_upper_touch
                & k_cross_down_recent
                & k_below_d
                & k_falling
                & rsi_ok_short
            )
            signals[short_cond] = -1

        return signals

    # ── SL levels ─────────────────────────────────────────────────────────────

    def swing_levels(self, df: pd.DataFrame):
        """Swing low/high с минимальным отступом 1 × ATR. Без look-ahead."""
        close    = df["close"]
        atr_vals = _atr(df, self.atr_period)
        min_dist = self.atr_sl_mult * atr_vals

        sw_low  = df["low"].rolling(self.swing_period, min_periods=1).min().shift(1)
        sw_high = df["high"].rolling(self.swing_period, min_periods=1).max().shift(1)

        sw_low  = (close - min_dist).where(close - sw_low  < min_dist, sw_low)
        sw_high = (close + min_dist).where(sw_high - close < min_dist, sw_high)

        return sw_low, sw_high
