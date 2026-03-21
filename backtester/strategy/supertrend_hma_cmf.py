"""
SuperTrend + Hull Moving Average + Chaikin Money Flow strategy (v2).

Изменения vs v1:
- Выход ТОЛЬКО по смене SuperTrend (убран выход по CMF — он давал сотни лишних сделок)
- CMF используется только как фильтр на вход (значение, не пересечение нуля)
- Минимальное удержание позиции min_bars свечей перед проверкой выхода
- SuperTrend multiplier по умолчанию 4.0 (меньше ложных разворотов)

Вход Long:  SuperTrend перешёл в зелёный И цена выше HMA 55 И CMF > 0
Вход Short: SuperTrend перешёл в красный И цена ниже HMA 55 И CMF < 0
Выход:      SuperTrend меняет направление (после min_bars удержания)
"""

import pandas as pd

from backtester.strategy.base import Strategy
from backtester.strategy.indicators import supertrend, hma, cmf


class SupertrendHmaCmfStrategy(Strategy):
    """
    Parameters
    ----------
    st_period  : SuperTrend ATR period (default 10)
    st_mult    : SuperTrend ATR multiplier (default 4.0)
    hma_period : Hull MA period (default 55)
    cmf_period : Chaikin Money Flow period (default 20)
    min_bars   : Minimum bars to hold before allowing exit (default 4)
    """

    def __init__(
        self,
        st_period:  int   = 10,
        st_mult:    float = 4.0,
        hma_period: int   = 55,
        cmf_period: int   = 20,
        min_bars:   int   = 4,
    ):
        self.st_period  = st_period
        self.st_mult    = st_mult
        self.hma_period = hma_period
        self.cmf_period = cmf_period
        self.min_bars   = min_bars

    def generate_signals(self, df: pd.DataFrame, aux=None) -> pd.Series:
        close = df["close"]

        # ── Indicators ────────────────────────────────────────────────────────
        st_dir, _ = supertrend(df, period=self.st_period, multiplier=self.st_mult)
        hma_line  = hma(close, self.hma_period)
        cmf_line  = cmf(df, self.cmf_period)

        # SuperTrend flip: переход направления
        st_prev      = st_dir.shift(1)
        st_turned_up = (st_prev == -1) & (st_dir == 1)   # красный → зелёный
        st_turned_dn = (st_prev ==  1) & (st_dir == -1)  # зелёный → красный

        # ── Entry conditions ──────────────────────────────────────────────────
        long_entry  = st_turned_up & (close > hma_line) & (cmf_line > 0)
        short_entry = st_turned_dn & (close < hma_line) & (cmf_line < 0)

        # ── Exit: только по смене SuperTrend ─────────────────────────────────
        long_exit  = st_dir == -1   # ST красный → выход из лонга
        short_exit = st_dir ==  1   # ST зелёный → выход из шорта

        # ── Build position series bar-by-bar ──────────────────────────────────
        signal    = pd.Series(0, index=df.index, dtype=int)
        position  = 0
        bars_held = 0

        for i in range(len(df)):
            if position == 0:
                if long_entry.iloc[i]:
                    position  = 1
                    bars_held = 0
                elif short_entry.iloc[i]:
                    position  = -1
                    bars_held = 0
            elif position == 1:
                bars_held += 1
                if bars_held >= self.min_bars and long_exit.iloc[i]:
                    position  = 0
                    bars_held = 0
                    # немедленный разворот если есть сигнал шорта
                    if short_entry.iloc[i]:
                        position  = -1
                        bars_held = 0
            elif position == -1:
                bars_held += 1
                if bars_held >= self.min_bars and short_exit.iloc[i]:
                    position  = 0
                    bars_held = 0
                    if long_entry.iloc[i]:
                        position  = 1
                        bars_held = 0

            signal.iloc[i] = position

        return signal

    def get_indicators(self, df: pd.DataFrame) -> dict:
        close           = df["close"]
        _, st_line      = supertrend(df, period=self.st_period, multiplier=self.st_mult)
        hma_line        = hma(close, self.hma_period)
        return {
            f"HMA {self.hma_period}": hma_line,
            "SuperTrend":             st_line,
        }
