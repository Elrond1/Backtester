"""
Vectorized backtesting engine.

No look-ahead bias: signals are shifted by 1 bar before computing returns.
Supports long (1), short (-1), and flat (0) positions.
Models commissions and slippage.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtester.strategy.base import Strategy


@dataclass
class BacktestResult:
    """Contains all outputs of a backtest run."""

    equity: pd.Series          # Equity curve (absolute value)
    returns: pd.Series         # Bar-by-bar strategy returns
    positions: pd.Series       # Position series (after shift)
    trades: pd.DataFrame       # Trade log
    metrics: dict              # Performance metrics dict
    params: dict               # Strategy parameters used
    symbol: str = ""
    timeframe: str = ""

    def report(self) -> str:
        lines = [
            f"{'='*46}",
            f"  Backtest Report — {self.symbol} {self.timeframe}",
            f"{'='*46}",
        ]
        for k, v in self.metrics.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                lines.append(f"  {label:<25} {v:>10.4f}")
            else:
                lines.append(f"  {label:<25} {str(v):>10}")
        lines.append(f"{'='*46}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.report()


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,   # 0.1% per trade side
    slippage: float = 0.0005,    # 0.05% price impact
    symbol: str = "",
    timeframe: str = "",
    aux: Optional[dict[str, pd.DataFrame]] = None,
) -> BacktestResult:
    """
    Run a vectorized backtest.

    Parameters
    ----------
    df              : OHLCV DataFrame (DatetimeIndex)
    strategy        : Strategy instance with generate_signals()
    initial_capital : Starting capital in quote currency
    commission      : Fraction charged per trade per side (e.g. 0.001 = 0.1%)
    slippage        : Price slippage fraction on entry/exit
    symbol          : For display only
    timeframe       : For display only
    aux             : Extra DataFrames for multi-timeframe strategies

    Returns
    -------
    BacktestResult
    """
    raw_signals = strategy.generate_signals(df, aux=aux).reindex(df.index).fillna(0)
    raw_signals = raw_signals.clip(-1, 1)

    # Shift 1 to avoid look-ahead: position is decided at bar close, active next bar
    positions = raw_signals.shift(1).fillna(0)

    close = df["close"]
    bar_returns = close.pct_change().fillna(0)

    # Position changes trigger commission + slippage
    pos_change = positions.diff().fillna(positions.iloc[0])
    trade_cost = pos_change.abs() * (commission + slippage)

    strategy_returns = positions * bar_returns - trade_cost

    equity = (1 + strategy_returns).cumprod() * initial_capital

    trades = _extract_trades(positions, close, commission, slippage, initial_capital)

    from backtester.engine.metrics import compute_metrics
    metrics = compute_metrics(strategy_returns, equity, trades, initial_capital)

    return BacktestResult(
        equity=equity,
        returns=strategy_returns,
        positions=positions,
        trades=trades,
        metrics=metrics,
        params=strategy.get_params(),
        symbol=symbol,
        timeframe=timeframe,
    )


def _extract_trades(
    positions: pd.Series,
    close: pd.Series,
    commission: float,
    slippage: float,
    initial_capital: float,
) -> pd.DataFrame:
    """Extract individual trade records from position series."""
    trades = []
    in_trade = False
    entry_time = None
    entry_price = None
    side = 0

    pos_vals = positions.values
    times = positions.index
    prices = close.values

    for i in range(len(pos_vals)):
        p = pos_vals[i]
        price = prices[i]

        if not in_trade and p != 0:
            in_trade = True
            side = int(p)
            entry_time = times[i]
            entry_price = price * (1 + side * slippage)

        elif in_trade and (p != side or i == len(pos_vals) - 1):
            exit_price = price * (1 - side * slippage)
            raw_pnl = side * (exit_price / entry_price - 1)
            net_pnl = raw_pnl - 2 * (commission + slippage)

            trades.append({
                "entry_time": entry_time,
                "exit_time": times[i],
                "side": "long" if side == 1 else "short",
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "pnl_pct": round(net_pnl * 100, 4),
                "duration": times[i] - entry_time,
            })

            in_trade = False
            if p != 0:
                in_trade = True
                side = int(p)
                entry_time = times[i]
                entry_price = price * (1 + side * slippage)

    return pd.DataFrame(trades)
