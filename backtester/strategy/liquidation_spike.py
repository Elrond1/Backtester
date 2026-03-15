"""
LiquidationSpike strategy.

Logic:
  - When hourly long liquidations exceed threshold → shorts got squeezed → go LONG
    (price pumped hard, momentum likely continues 1 bar)
  - When hourly short liquidations exceed threshold → longs got squeezed → go SHORT
    (price dumped hard, momentum likely continues 1 bar)

  Uses aux["liq"] DataFrame (liq_long, liq_short, liq_total) aligned with OHLCV.

Direction modes:
  "trend"       — trade in direction of the squeeze (default)
  "contrarian"  — fade the move (mean reversion)

Position sizing note:
  The 20% capital / 1% target is handled at run_backtest level via
  initial_capital and commission settings. The strategy only outputs signals.
"""

import pandas as pd
import numpy as np

from backtester.strategy.base import Strategy


class LiquidationSpike(Strategy):
    """
    Opens a position when a liquidation spike is detected.

    Parameters
    ----------
    threshold_usd   : Minimum liquidation volume in millions USD to trigger signal.
                      e.g. 50 = $50M in one hour  (Coinalyze data is in millions USD)
    side            : "both" | "long_only" | "short_only"
                      which direction of liquidation to trade
    direction       : "trend" | "contrarian"
    hold_bars       : How many bars to hold the position after entry
    zscore_mode     : If True, use Z-score > threshold instead of raw value
                      (more adaptive to market regime, threshold becomes z-score units)
    zscore_window   : Rolling window for Z-score calculation
    """

    def __init__(
        self,
        threshold_usd: float = 50,  # $50M in Coinalyze units (millions USD)
        side: str = "both",
        direction: str = "trend",
        hold_bars: int = 3,
        zscore_mode: bool = False,
        zscore_window: int = 168,   # 1 week of hourly bars
    ):
        self.threshold_usd = threshold_usd
        self.side = side
        self.direction = direction
        self.hold_bars = hold_bars
        self.zscore_mode = zscore_mode
        self.zscore_window = zscore_window

    def generate_signals(
        self,
        df: pd.DataFrame,
        aux: dict | None = None,
    ) -> pd.Series:
        if aux is None or "liq" not in aux:
            raise ValueError(
                "LiquidationSpike requires aux={'liq': liquidations_df}.\n"
                "Use: run_backtest(df, strategy, aux={'liq': liq_df})"
            )

        liq = aux["liq"].reindex(df.index, method="ffill")

        long_vol = liq["liq_long"].fillna(0)
        short_vol = liq["liq_short"].fillna(0)

        if self.zscore_mode:
            long_trigger = self._zscore(long_vol) > self.threshold_usd
            short_trigger = self._zscore(short_vol) > self.threshold_usd
        else:
            long_trigger = long_vol > self.threshold_usd
            short_trigger = short_vol > self.threshold_usd

        # Direction: who got liquidated, which way to trade
        # shorts squeezed (liq_short spike) → price pumped → LONG in trend mode
        # longs squeezed (liq_long spike)   → price dumped → SHORT in trend mode
        if self.direction == "trend":
            buy_signal = short_trigger   # shorts squeezed → follow pump
            sell_signal = long_trigger   # longs squeezed → follow dump
        else:  # contrarian
            buy_signal = long_trigger    # longs squeezed → fade the dump
            sell_signal = short_trigger  # shorts squeezed → fade the pump

        if self.side == "long_only":
            sell_signal = pd.Series(False, index=df.index)
        elif self.side == "short_only":
            buy_signal = pd.Series(False, index=df.index)

        # Convert spike → hold for N bars
        raw = pd.Series(0, index=df.index, dtype=float)
        raw[buy_signal] = 1.0
        raw[sell_signal] = -1.0

        # Extend signal for hold_bars (forward fill after each trigger)
        signal = self._extend_signal(raw, self.hold_bars)
        return signal

    @staticmethod
    def _zscore(series: pd.Series, window: int = 168) -> pd.Series:
        mean = series.rolling(window, min_periods=24).mean()
        std = series.rolling(window, min_periods=24).std()
        return (series - mean) / std.replace(0, np.nan)

    @staticmethod
    def _extend_signal(raw: pd.Series, hold_bars: int) -> pd.Series:
        """Hold position for hold_bars after each spike."""
        result = pd.Series(0.0, index=raw.index)
        vals = raw.values
        out = result.values.copy()
        last_val = 0.0
        bars_held = 0

        for i in range(len(vals)):
            if vals[i] != 0:
                last_val = vals[i]
                bars_held = hold_bars
            if bars_held > 0:
                out[i] = last_val
                bars_held -= 1
            else:
                out[i] = 0.0

        return pd.Series(out, index=raw.index)
