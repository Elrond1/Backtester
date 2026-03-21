"""
LiquidationGrid — grid entry triggered by liquidation-proxy signals.

No external API needed — detects liquidation events via:
  1. Abnormal volume spike (z-score > threshold)
  2. Significant price move in the same bar (confirms forced liquidation)

Signals:
  Volume spike + bar DOWN (bearish) → longs liquidated → LONG grid
  Volume spike + bar UP  (bullish) → shorts liquidated → SHORT grid

Filters applied (all optional):
  EMA / HMA 200    — macro trend direction (HMA = less lag vs EMA)
  RSI              — avoid extremes
  ADX              — skip strong-trend bars (mean-reversion only in ranging)
  Absorbing candle — long when down-bar closed in top % of range
  Bollinger Bands  — long only below lower BB
  Keltner Channel  — long only below lower KC (ATR-based)
  CMF              — Chaikin Money Flow: money flow confirms buyer absorption
"""

import pandas as pd
import numpy as np

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import (
    ema as calc_ema,
    hma as calc_hma,
    rsi as calc_rsi,
    adx as calc_adx,
    bollinger_bands as calc_bb,
    atr as calc_atr,
    cmf as calc_cmf,
)


class LiquidationGridTrigger(Strategy):
    def __init__(
        self,
        zscore_window:    int   = 20,
        zscore_threshold: float = 4.0,
        min_move_pct:     float = 0.3,
        cooldown_bars:    int   = 30,
        # EMA macro-trend filter
        ema_period:       int   = 200,
        use_ema_filter:   bool  = True,
        # HMA trend filter (faster, less lag — overrides EMA when enabled)
        use_hma_filter:   bool  = False,
        hma_period:       int   = 200,
        # RSI extremes filter
        rsi_period:       int   = 14,
        rsi_long_max:     float = 65.0,
        rsi_short_min:    float = 35.0,
        use_rsi_filter:   bool  = True,
        # ADX ranging-market filter
        adx_period:       int   = 14,
        adx_max:          float = 30.0,
        use_adx_filter:   bool  = False,
        # Absorbing candle filter
        use_absorption_filter: bool  = True,
        absorption_min_pos:    float = 0.4,
        absorption_max_pos:    float = 0.6,
        # Bollinger Bands filter
        use_bb_filter: bool  = False,
        bb_period:     int   = 20,
        bb_dev:        float = 2.0,
        # Keltner Channel filter
        use_kc_filter:  bool  = False,
        kc_period:      int   = 20,
        kc_atr_period:  int   = 10,
        kc_mult:        float = 1.5,
        # CMF filter (Chaikin Money Flow)
        use_cmf_filter: bool  = False,
        cmf_period:     int   = 20,
        cmf_long_min:   float = -0.10,   # not strongly negative = buyers absorbing
        cmf_short_max:  float =  0.10,   # not strongly positive = sellers absorbing
    ):
        self.zscore_window    = zscore_window
        self.zscore_threshold = zscore_threshold
        self.min_move_pct     = min_move_pct
        self.cooldown_bars    = cooldown_bars

        self.ema_period     = ema_period
        self.use_ema_filter = use_ema_filter

        self.use_hma_filter = use_hma_filter
        self.hma_period     = hma_period

        self.rsi_period     = rsi_period
        self.rsi_long_max   = rsi_long_max
        self.rsi_short_min  = rsi_short_min
        self.use_rsi_filter = use_rsi_filter

        self.adx_period     = adx_period
        self.adx_max        = adx_max
        self.use_adx_filter = use_adx_filter

        self.use_absorption_filter = use_absorption_filter
        self.absorption_min_pos    = absorption_min_pos
        self.absorption_max_pos    = absorption_max_pos

        self.use_bb_filter = use_bb_filter
        self.bb_period     = bb_period
        self.bb_dev        = bb_dev

        self.use_kc_filter    = use_kc_filter
        self.kc_period        = kc_period
        self.kc_atr_period    = kc_atr_period
        self.kc_mult          = kc_mult

        self.use_cmf_filter = use_cmf_filter
        self.cmf_period     = cmf_period
        self.cmf_long_min   = cmf_long_min
        self.cmf_short_max  = cmf_short_max

    def generate_signals(self, df: pd.DataFrame, aux: dict | None = None) -> pd.Series:
        close = df["close"]

        # ── Volume spike detection ────────────────────────────────────────────
        vol    = df["volume"]
        mean   = vol.rolling(self.zscore_window, min_periods=5).mean()
        std    = vol.rolling(self.zscore_window, min_periods=5).std().replace(0, np.nan)
        zscore = (vol - mean) / std

        bar_move = (close / df["open"] - 1) * 100

        vol_spike     = zscore > self.zscore_threshold
        long_trigger  = vol_spike & (bar_move < -self.min_move_pct)
        short_trigger = vol_spike & (bar_move >  self.min_move_pct)

        # ── CMF filter — money flow confirms absorption ───────────────────────
        # After a down-spike (long liquidations), if CMF > cmf_long_min,
        # buyers are absorbing the selling → good long setup
        if self.use_cmf_filter:
            cmf_vals = calc_cmf(df, self.cmf_period)
            long_trigger  = long_trigger  & (cmf_vals > self.cmf_long_min)
            short_trigger = short_trigger & (cmf_vals < self.cmf_short_max)

        # ── Bollinger Bands filter ────────────────────────────────────────────
        if self.use_bb_filter:
            bb_upper, bb_mid, bb_lower = calc_bb(close, self.bb_period, self.bb_dev)
            long_trigger  = long_trigger  & (close < bb_lower)
            short_trigger = short_trigger & (close > bb_upper)

        # ── Keltner Channel filter ────────────────────────────────────────────
        if self.use_kc_filter:
            kc_mid   = calc_ema(close, self.kc_period)
            kc_atr   = calc_atr(df, self.kc_atr_period)
            kc_lower = kc_mid - self.kc_mult * kc_atr
            kc_upper = kc_mid + self.kc_mult * kc_atr
            long_trigger  = long_trigger  & (close < kc_lower)
            short_trigger = short_trigger & (close > kc_upper)

        # ── Absorbing candle filter ───────────────────────────────────────────
        if self.use_absorption_filter:
            bar_range  = (df["high"] - df["low"]).replace(0, np.nan)
            close_pos  = (close - df["low"]) / bar_range
            long_trigger  = long_trigger  & (close_pos >= self.absorption_min_pos)
            short_trigger = short_trigger & (close_pos <= self.absorption_max_pos)

        # ── HMA or EMA trend filter ───────────────────────────────────────────
        if self.use_hma_filter:
            trend_line = calc_hma(close, self.hma_period)
            long_trigger  = long_trigger  & (close > trend_line)
            short_trigger = short_trigger & (close < trend_line)
        elif self.use_ema_filter:
            trend_line = calc_ema(close, self.ema_period)
            long_trigger  = long_trigger  & (close > trend_line)
            short_trigger = short_trigger & (close < trend_line)

        # ── RSI filter ────────────────────────────────────────────────────────
        if self.use_rsi_filter:
            rsi_vals = calc_rsi(close, self.rsi_period)
            long_trigger  = long_trigger  & (rsi_vals < self.rsi_long_max)
            short_trigger = short_trigger & (rsi_vals > self.rsi_short_min)

        # ── ADX filter ────────────────────────────────────────────────────────
        if self.use_adx_filter:
            adx_vals = calc_adx(df, self.adx_period)
            ranging  = adx_vals < self.adx_max
            long_trigger  = long_trigger  & ranging
            short_trigger = short_trigger & ranging

        raw = pd.Series(0.0, index=df.index)
        raw[long_trigger]  =  1.0
        raw[short_trigger] = -1.0

        if self.cooldown_bars > 0:
            raw = self._apply_cooldown(raw, self.cooldown_bars)

        return raw

    @staticmethod
    def _apply_cooldown(raw: pd.Series, cooldown: int) -> pd.Series:
        out = raw.values.copy()
        last_trigger = -cooldown - 1
        for i in range(len(out)):
            if out[i] != 0:
                if i - last_trigger <= cooldown:
                    out[i] = 0.0
                else:
                    last_trigger = i
        return pd.Series(out, index=raw.index)
