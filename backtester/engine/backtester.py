"""
Vectorized backtesting engine.

No look-ahead bias: signals are shifted by 1 bar before computing returns.
Supports long (1), short (-1), and flat (0) positions.
Models commissions and slippage.

When take_profit or stop_loss are specified, the engine switches to a
bar-by-bar simulation that checks intrabar high/low for TP/SL hits.
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
    take_profit: float = 0.0,    # fraction, e.g. 0.02 = 2% TP (0 = disabled)
    stop_loss: float = 0.0,      # fraction, e.g. 0.01 = 1% SL (0 = disabled)
) -> "BacktestResult":
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
    take_profit     : Take profit as fraction of entry price (0 = disabled)
    stop_loss       : Stop loss as fraction of entry price (0 = disabled)

    Returns
    -------
    BacktestResult
    """
    raw_signals = strategy.generate_signals(df, aux=aux).reindex(df.index).fillna(0)
    raw_signals = raw_signals.clip(-1, 1)

    if take_profit > 0 or stop_loss > 0:
        trades, equity, strategy_returns, positions = _simulate_tp_sl(
            raw_signals, df, commission, slippage, initial_capital, take_profit, stop_loss
        )
    else:
        # Fast vectorized path (no TP/SL)
        positions = raw_signals.shift(1).fillna(0)
        close = df["close"]
        bar_returns = close.pct_change().fillna(0)
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


# ── Vectorized trade extractor (no TP/SL) ────────────────────────────────────

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


# ── Bar-by-bar simulation with TP/SL ─────────────────────────────────────────

def _simulate_tp_sl(
    raw_signals: pd.Series,
    df: pd.DataFrame,
    commission: float,
    slippage: float,
    initial_capital: float,
    take_profit: float,
    stop_loss: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Bar-by-bar simulation that checks high/low for intrabar TP/SL hits.

    SL is checked before TP (conservative — assume worst case first).
    After a forced TP/SL exit, new entry is blocked until the next bar.

    Returns
    -------
    (trades_df, equity, returns, positions)
    """
    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    times     = df.index
    n         = len(times)

    signals = raw_signals.shift(1).fillna(0).values.astype(float)

    trades      = []
    capital     = initial_capital
    equity_arr  = np.empty(n)
    returns_arr = np.zeros(n)
    pos_arr     = np.zeros(n)

    in_trade    = False
    side        = 0
    entry_idx   = 0
    entry_price = 0.0
    block_entry = False   # stay flat on bar right after forced TP/SL exit

    for i in range(n):
        sig        = int(signals[i])
        bar_return = 0.0
        c, h, l    = close_arr[i], high_arr[i], low_arr[i]

        if in_trade:
            pos_arr[i] = side

            # ── Check TP/SL (SL has priority) ──
            exit_price = None
            forced     = False

            if side == 1:   # long
                if stop_loss  > 0 and l <= entry_price * (1 - stop_loss):
                    exit_price = entry_price * (1 - stop_loss)
                    forced = True
                elif take_profit > 0 and h >= entry_price * (1 + take_profit):
                    exit_price = entry_price * (1 + take_profit)
                    forced = True
            else:           # short
                if stop_loss  > 0 and h >= entry_price * (1 + stop_loss):
                    exit_price = entry_price * (1 + stop_loss)
                    forced = True
                elif take_profit > 0 and l <= entry_price * (1 - take_profit):
                    exit_price = entry_price * (1 - take_profit)
                    forced = True

            # ── Signal-based exit (or final bar) ──
            if not forced and (sig != side or i == n - 1):
                exit_price = c * (1 - side * slippage)

            if exit_price is not None:
                # Apply exit slippage on forced exits
                if forced:
                    exit_price = exit_price * (1 - side * slippage)

                net_pnl = side * (exit_price / entry_price - 1) - 2 * (commission + slippage)
                bar_return += net_pnl
                capital   *= (1 + net_pnl)

                trades.append({
                    "entry_time":  times[entry_idx],
                    "exit_time":   times[i],
                    "side":        "long" if side == 1 else "short",
                    "entry_price": round(entry_price, 6),
                    "exit_price":  round(exit_price, 6),
                    "pnl_pct":     round(net_pnl * 100, 4),
                    "duration":    times[i] - times[entry_idx],
                })

                in_trade = False
                if forced:
                    block_entry = True   # skip new entry this bar

                elif sig != 0 and sig != side:
                    # Immediate reversal on signal exit (not TP/SL)
                    in_trade    = True
                    side        = sig
                    entry_idx   = i
                    entry_price = c * (1 + side * slippage)
                    pos_arr[i]  = side
                    cost        = commission + slippage
                    bar_return -= cost
                    capital    *= (1 - cost)

        # ── Enter if flat and signal says so ──
        if not in_trade and not block_entry and sig != 0:
            in_trade    = True
            side        = sig
            entry_idx   = i
            entry_price = c * (1 + side * slippage)
            pos_arr[i]  = side
            cost        = commission + slippage
            bar_return -= cost
            capital    *= (1 - cost)

        block_entry     = False   # reset every bar
        returns_arr[i]  = bar_return
        equity_arr[i]   = capital

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "entry_price",
                 "exit_price", "pnl_pct", "duration"]
    )
    return (
        trades_df,
        pd.Series(equity_arr,  index=df.index),
        pd.Series(returns_arr, index=df.index),
        pd.Series(pos_arr,     index=df.index),
    )
