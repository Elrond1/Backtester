"""
MT4-style "Every Tick" backtesting engine using 1-second OHLCV bars.

Strategies generate signals on regular OHLCV bars (any timeframe).
Execution — entries, exits, TP and SL — is simulated on 1-second bars,
giving precision comparable to MT4's "Every Tick" mode without requiring
raw tick data.

1-second bars are ~15-30 MB/month per symbol vs ~2 GB/month for aggTrades,
making it practical to store years of history locally in DuckDB.

Usage
-----
    from backtester.data.manager import get_ohlcv, get_seconds
    from backtester.engine.tick_backtester import run_tick_backtest

    df    = get_ohlcv("BTC/USDT", "1h",  since="2020-01-01")
    df_1s = get_seconds("BTC/USDT",      since="2020-01-01")

    result = run_tick_backtest(
        df, df_1s, strategy,
        take_profit=0.02,
        stop_loss=0.01,
    )
    print(result)
"""

from typing import Optional

import numpy as np
import pandas as pd

from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.strategy.base import Strategy


def run_tick_backtest(
    df: pd.DataFrame,
    df_1s: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,   # 0.1% per trade side
    slippage: float = 0.0005,    # 0.05% price impact
    symbol: str = "",
    timeframe: str = "",
    aux: Optional[dict[str, pd.DataFrame]] = None,
    take_profit: float = 0.0,    # fraction, e.g. 0.02 = 2% TP (0 = disabled)
    stop_loss: float = 0.0,      # fraction, e.g. 0.01 = 1% SL (0 = disabled)
) -> BacktestResult:
    """
    Run a backtest with 1-second bar precision (MT4 Every Tick equivalent).

    Signals are generated on ``df`` (any timeframe) and shifted by 1 bar to
    avoid look-ahead bias.  Trade execution — entry price, TP, and SL — is
    resolved on ``df_1s`` (1-second bars).

    Within each 1-second bar the price path is inferred from the bar's
    direction, matching the MT4 "Control Points" heuristic:

    * Bullish 1s bar (close >= open)  →  O → Low → High → Close
    * Bearish 1s bar (close <  open)  →  O → High → Low → Close

    This determines which of TP or SL is checked first, eliminating the
    conservative "SL always first" bias of bar-by-bar engines.

    Parameters
    ----------
    df          : OHLCV DataFrame (any timeframe) — for signal generation
    df_1s       : 1-second OHLCV DataFrame — for trade execution precision
    strategy    : Strategy instance with generate_signals()
    take_profit : TP as fraction of entry price (0 = disabled)
    stop_loss   : SL as fraction of entry price (0 = disabled)

    Returns
    -------
    BacktestResult — same format as run_backtest(), fully compatible with
    all visualisation and metrics utilities.
    """
    raw_signals = strategy.generate_signals(df, aux=aux).reindex(df.index).fillna(0)
    raw_signals = raw_signals.clip(-1, 1)
    signals = raw_signals.shift(1).fillna(0)

    trades, equity, returns, positions = _simulate_1s(
        signals, df, df_1s,
        commission, slippage, initial_capital,
        take_profit, stop_loss,
    )

    metrics = compute_metrics(returns, equity, trades, initial_capital)

    return BacktestResult(
        equity=equity,
        returns=returns,
        positions=positions,
        trades=trades,
        metrics=metrics,
        params=strategy.get_params(),
        symbol=symbol,
        timeframe=timeframe,
    )


# ── Core simulation ───────────────────────────────────────────────────────────

def _simulate_1s(
    signals: pd.Series,
    df: pd.DataFrame,
    df_1s: pd.DataFrame,
    commission: float,
    slippage: float,
    initial_capital: float,
    take_profit: float,
    stop_loss: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Bar-by-bar simulation using 1-second sub-bars for TP/SL precision.

    Entry/exit at OHLCV bar boundaries uses the open of the first 1s sub-bar
    (or the OHLCV open when 1s data is absent for that period).

    TP/SL are checked on every 1s sub-bar using the bar-direction heuristic.
    SL and TP are never checked simultaneously on the same 1s bar — the order
    depends on whether the bar is bullish or bearish (see module docstring).
    """
    n = len(df)
    bar_times = df.index

    s1_times  = df_1s.index
    s1_open_  = df_1s["open"].values
    s1_high_  = df_1s["high"].values
    s1_low_   = df_1s["low"].values
    s1_close_ = df_1s["close"].values

    sig_arr  = signals.values.astype(int)
    df_open  = df["open"].values
    df_close = df["close"].values

    # Precompute 1s-bar slice boundaries for each OHLCV bar using searchsorted.
    # s1_starts[i] = first 1s index falling inside OHLCV bar i
    # s1_ends[i]   = first 1s index of OHLCV bar i+1  (exclusive upper bound)
    bar_ns   = bar_times.view("int64")
    s1_ns    = s1_times.view("int64")
    s1_starts            = np.searchsorted(s1_ns, bar_ns, side="left").astype(int)
    s1_ends              = np.empty(n, dtype=int)
    s1_ends[:-1]         = s1_starts[1:]
    s1_ends[-1]          = len(s1_times)

    trades      = []
    capital     = initial_capital
    equity_arr  = np.empty(n)
    returns_arr = np.zeros(n)
    pos_arr     = np.zeros(n)

    in_trade    = False
    side        = 0
    entry_price = 0.0
    entry_bar   = 0       # OHLCV bar index where position was entered
    block_entry = False   # True for the rest of the bar after a forced TP/SL exit

    for i in range(n):
        sig   = sig_arr[i]
        s1_s  = s1_starts[i]
        s1_e  = s1_ends[i]
        has1s = s1_s < s1_e
        bar_return = 0.0

        # Reference open for this OHLCV bar: first 1s open, or OHLCV open fallback
        ref_open = float(s1_open_[s1_s]) if has1s else float(df_open[i])

        # ── Signal exit / entry at the open of this bar ───────────────────────
        if in_trade and sig != side:
            ep  = ref_open * (1.0 - side * slippage)
            net = side * (ep / entry_price - 1.0) - 2.0 * (commission + slippage)
            bar_return += net
            capital    *= (1.0 + net)
            trades.append(_trade_row(bar_times[entry_bar], bar_times[i],
                                     side, entry_price, ep, net))
            in_trade = False

            # Immediate reversal on signal flip
            if sig != 0 and not block_entry:
                entry_price  = ref_open * (1.0 + sig * slippage)
                cost         = commission + slippage
                bar_return  -= cost
                capital     *= (1.0 - cost)
                in_trade     = True
                side         = sig
                entry_bar    = i

        elif not in_trade and sig != 0 and not block_entry:
            entry_price = ref_open * (1.0 + sig * slippage)
            cost        = commission + slippage
            bar_return -= cost
            capital    *= (1.0 - cost)
            in_trade    = True
            side        = sig
            entry_bar   = i

        # ── TP / SL scan on 1s sub-bars ──────────────────────────────────────
        if in_trade and has1s and (take_profit > 0.0 or stop_loss > 0.0):
            if side == 1:
                tp_lv = entry_price * (1.0 + take_profit) if take_profit > 0.0 else None
                sl_lv = entry_price * (1.0 - stop_loss)  if stop_loss  > 0.0 else None
            else:
                tp_lv = entry_price * (1.0 - take_profit) if take_profit > 0.0 else None
                sl_lv = entry_price * (1.0 + stop_loss)  if stop_loss  > 0.0 else None

            for j in range(s1_s, s1_e):
                h      = s1_high_[j]
                l      = s1_low_[j]
                o1     = s1_open_[j]
                c1     = s1_close_[j]
                bullish = c1 >= o1   # determines intrabar price path

                hit = None
                if side == 1:       # long position
                    if bullish:     # O → Low → High → C  (SL before TP)
                        if sl_lv is not None and l <= sl_lv:
                            hit = sl_lv
                        elif tp_lv is not None and h >= tp_lv:
                            hit = tp_lv
                    else:           # O → High → Low → C  (TP before SL)
                        if tp_lv is not None and h >= tp_lv:
                            hit = tp_lv
                        elif sl_lv is not None and l <= sl_lv:
                            hit = sl_lv
                else:               # short position
                    if bullish:     # O → Low → High → C  (TP(low) before SL(high))
                        if tp_lv is not None and l <= tp_lv:
                            hit = tp_lv
                        elif sl_lv is not None and h >= sl_lv:
                            hit = sl_lv
                    else:           # O → High → Low → C  (SL(high) before TP(low))
                        if sl_lv is not None and h >= sl_lv:
                            hit = sl_lv
                        elif tp_lv is not None and l <= tp_lv:
                            hit = tp_lv

                if hit is not None:
                    ep  = hit * (1.0 - side * slippage)
                    net = side * (ep / entry_price - 1.0) - 2.0 * (commission + slippage)
                    bar_return += net
                    capital    *= (1.0 + net)
                    trades.append(_trade_row(bar_times[entry_bar], s1_times[j],
                                             side, entry_price, ep, net))
                    in_trade    = False
                    block_entry = True
                    break

        # ── Force-close on the final OHLCV bar ───────────────────────────────
        if in_trade and i == n - 1:
            ref_close = float(s1_close_[s1_e - 1]) if has1s else float(df_close[i])
            ep  = ref_close * (1.0 - side * slippage)
            net = side * (ep / entry_price - 1.0) - 2.0 * (commission + slippage)
            bar_return += net
            capital    *= (1.0 + net)
            trades.append(_trade_row(bar_times[entry_bar], bar_times[i],
                                     side, entry_price, ep, net))
            in_trade = False

        pos_arr[i]     = float(side) if in_trade else 0.0
        returns_arr[i] = bar_return
        equity_arr[i]  = capital
        block_entry    = False   # reset at end of each OHLCV bar

    cols = ["entry_time", "exit_time", "side",
            "entry_price", "exit_price", "pnl_pct", "duration"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=cols)

    return (
        trades_df,
        pd.Series(equity_arr,  index=df.index),
        pd.Series(returns_arr, index=df.index),
        pd.Series(pos_arr,     index=df.index),
    )


def _trade_row(
    entry_time, exit_time, side: int,
    entry_price: float, exit_price: float, net_pnl: float,
) -> dict:
    return {
        "entry_time":  entry_time,
        "exit_time":   exit_time,
        "side":        "long" if side == 1 else "short",
        "entry_price": round(entry_price, 6),
        "exit_price":  round(exit_price, 6),
        "pnl_pct":     round(net_pnl * 100, 4),
        "duration":    exit_time - entry_time,
    }
