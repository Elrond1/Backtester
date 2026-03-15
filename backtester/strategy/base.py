"""Abstract Strategy base class."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Subclass this and implement `generate_signals`.

    Example
    -------
    class MyCrossover(Strategy):
        def __init__(self, fast=20, slow=50):
            self.fast = fast
            self.slow = slow

        def generate_signals(self, df, aux=None):
            from backtester.strategy.indicators import sma
            return (sma(df['close'], self.fast) > sma(df['close'], self.slow)).astype(int)
    """

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        aux: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.Series:
        """
        Generate trading signals from OHLCV data.

        Parameters
        ----------
        df  : Primary OHLCV DataFrame (DatetimeIndex, columns: open/high/low/close/volume)
        aux : Optional dict of additional DataFrames keyed by timeframe string,
              e.g. {"4h": df_4h, "1d": df_1d}. Used for multi-timeframe strategies.

        Returns
        -------
        pd.Series aligned with df.index:
            1  — long
           -1  — short
            0  — flat / no position
        """
        ...

    def get_params(self) -> dict:
        """Returns all constructor parameters (for optimizer)."""
        import inspect
        sig = inspect.signature(self.__class__.__init__)
        return {
            k: getattr(self, k)
            for k in sig.parameters
            if k != "self" and hasattr(self, k)
        }

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"
