"""
S/R Grid backtesting engine with MT4-level tick precision.

Strategy logic:
  1. Compute support/resistance level = (max_high + min_low) / 2
     over the last `lookback_d1` D1 bars (updated daily, no look-ahead).
  2. Enter LONG when 1s bar close is within `entry_tolerance` of S/R.
  3. Average down at fixed intervals from entry price:
       orders 1-2: every `first_avg_step` / `second_avg_step` (default 4%)
       orders 3+:  every `subsequent_step` (default 2%)
     Each averaging order uses `order_size_pct` of initial capital.
  4. Close ALL orders when price reaches avg_entry × (1 + take_profit_pct).
  5. No stop-loss — `max_orders` caps the maximum grid depth.

Simulation runs on 1-second OHLCV bars for MT4 "Every Tick" precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SRGridResult:
    equity: pd.Series      # equity at each trade close (+ initial point)
    trades: pd.DataFrame   # trade log
    metrics: dict          # performance metrics
    sr_levels: pd.Series   # daily S/R midpoint level
    symbol: str = ""

    def report(self) -> str:
        lines = [
            f"{'='*54}",
            f"  S/R Grid Backtest — {self.symbol}",
            f"{'='*54}",
        ]
        for k, v in self.metrics.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                lines.append(f"  {label:<32} {v:>10.4f}")
            else:
                lines.append(f"  {label:<32} {str(v):>10}")
        lines.append(f"{'='*54}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.report()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _grid_trade(entry_time, exit_time, side, avg_entry, exit_price,
                n_orders, invested_usd, pnl_usd, pnl_pct) -> dict:
    return {
        "entry_time":   entry_time,
        "exit_time":    exit_time,
        "side":         side,
        "avg_entry":    round(avg_entry, 4),
        "exit_price":   round(exit_price, 4),
        "n_orders":     n_orders,
        "invested_usd": round(invested_usd, 2),
        "pnl_usd":      round(pnl_usd, 2),
        "pnl_pct":      round(pnl_pct, 4),
    }


# ── S/R calculation ────────────────────────────────────────────────────────────

def _build_sr(df_d1: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Rolling midpoint of the D1 high/low range.
    Shifted by 1 bar so today's value is based on data up to yesterday only
    (no look-ahead bias).
    """
    high_roll = df_d1["high"].rolling(lookback, min_periods=lookback).max()
    low_roll  = df_d1["low"].rolling(lookback,  min_periods=lookback).min()
    return ((high_roll + low_roll) / 2).shift(1)


# ── Main backtest function ─────────────────────────────────────────────────────

def run_sr_grid_backtest(
    df_1s: pd.DataFrame,
    df_d1: pd.DataFrame,
    initial_capital: float = 10_000.0,
    order_size_pct: float = 0.03,      # 3% of initial capital per order = $300
    first_avg_step: float = 0.04,      # first averaging after 4% drop
    second_avg_step: float = 0.04,     # second averaging after another 4%
    subsequent_step: float = 0.02,     # each subsequent averaging every 2%
    take_profit_pct: float = 0.03,     # TP: 3% above weighted avg entry
    max_orders: int = 10,
    lookback_d1: int = 30,
    entry_tolerance: float = 0.005,    # 0.5% around S/R to trigger entry
    commission: float = 0.001,         # 0.1% per side
    slippage: float = 0.0005,          # 0.05% price impact
    symbol: str = "",
) -> SRGridResult:
    """
    Run S/R grid backtest on 1-second OHLCV bars.

    Parameters
    ----------
    df_1s         : 1-second OHLCV DataFrame (DatetimeIndex UTC)
    df_d1         : Daily OHLCV DataFrame (DatetimeIndex UTC)
    """
    # ── 1. Daily S/R → forward-fill onto 1s bar timestamps ──
    sr_d1 = _build_sr(df_d1, lookback_d1).dropna()
    sr_1s = sr_d1.reindex(df_1s.index, method="ffill").dropna()

    df_sim    = df_1s.reindex(sr_1s.index)
    high_arr  = df_sim["high"].values.astype(np.float64)
    low_arr   = df_sim["low"].values.astype(np.float64)
    close_arr = df_sim["close"].values.astype(np.float64)
    sr_arr    = sr_1s.values.astype(np.float64)
    times     = sr_1s.index
    n         = len(times)

    # ── 2. Pre-compute grid drop factors from entry price ──
    # factors[k] = grid_price_k / entry_price
    # e.g. max_orders=10: [1.0, 0.96, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78]
    factors = np.ones(max_orders, dtype=np.float64)
    if max_orders > 1:
        factors[1] = 1.0 - first_avg_step
    if max_orders > 2:
        factors[2] = factors[1] - second_avg_step
    for k in range(3, max_orders):
        factors[k] = factors[k - 1] - subsequent_step

    order_usd = initial_capital * order_size_pct  # USD per order = $300

    # ── 3. Simulation ──
    trades         = []
    capital        = initial_capital

    in_grid        = False
    n_orders       = 0          # orders placed so far (index of NEXT grid level to watch)
    total_qty      = 0.0        # total coins held
    total_invested = 0.0        # total USD invested in open orders (excl. fees)
    grid_prices    = np.zeros(max_orders)
    entry_time_val = None

    equity_events = [(times[0], initial_capital)]

    for i in range(n):
        sr = sr_arr[i]
        h  = high_arr[i]
        l  = low_arr[i]
        c  = close_arr[i]

        if in_grid:
            avg_cost = total_invested / total_qty
            tp_price = avg_cost * (1.0 + take_profit_pct)

            # ── Check TP hit on this 1s bar ──
            if h >= tp_price:
                exit_price = tp_price * (1.0 - slippage)
                proceeds   = total_qty * exit_price
                exit_fee   = proceeds * commission
                capital   += proceeds - exit_fee

                pnl_usd = (exit_price - avg_cost) * total_qty - exit_fee
                pnl_pct = pnl_usd / total_invested * 100.0

                trades.append({
                    "entry_time":   entry_time_val,
                    "exit_time":    times[i],
                    "side":         "long",
                    "avg_entry":    round(avg_cost, 4),
                    "exit_price":   round(exit_price, 4),
                    "n_orders":     n_orders,
                    "invested_usd": round(total_invested, 2),
                    "pnl_usd":      round(pnl_usd, 2),
                    "pnl_pct":      round(pnl_pct, 4),
                })
                equity_events.append((times[i], round(capital, 2)))

                in_grid        = False
                n_orders       = 0
                total_qty      = 0.0
                total_invested = 0.0
                entry_time_val = None

            # ── Check next averaging level(s) — handle price gaps ──
            elif n_orders < max_orders:
                while n_orders < max_orders and l <= grid_prices[n_orders]:
                    fill_price      = grid_prices[n_orders] * (1.0 + slippage)
                    qty             = order_usd / fill_price
                    entry_fee       = order_usd * commission
                    capital        -= (order_usd + entry_fee)
                    total_qty      += qty
                    total_invested += order_usd
                    n_orders       += 1

        else:
            # ── Check entry: close price within tolerance of S/R ──
            if sr > 0.0 and abs(c - sr) / sr <= entry_tolerance:
                fill_price      = c * (1.0 + slippage)
                qty             = order_usd / fill_price
                entry_fee       = order_usd * commission
                capital        -= (order_usd + entry_fee)
                total_qty       = qty
                total_invested  = order_usd
                n_orders        = 1          # first order placed; watching level[1] next
                entry_time_val  = times[i]
                in_grid         = True
                grid_prices     = fill_price * factors   # absolute levels

    # ── Close open grid at end of data ──
    if in_grid and total_qty > 0.0:
        exit_price = close_arr[-1] * (1.0 - slippage)
        avg_cost   = total_invested / total_qty
        proceeds   = total_qty * exit_price
        exit_fee   = proceeds * commission
        capital   += proceeds - exit_fee

        pnl_usd = (exit_price - avg_cost) * total_qty - exit_fee
        pnl_pct = pnl_usd / total_invested * 100.0

        trades.append({
            "entry_time":   entry_time_val,
            "exit_time":    times[-1],
            "side":         "long",
            "avg_entry":    round(avg_cost, 4),
            "exit_price":   round(exit_price, 4),
            "n_orders":     n_orders,
            "invested_usd": round(total_invested, 2),
            "pnl_usd":      round(pnl_usd, 2),
            "pnl_pct":      round(pnl_pct, 4),
        })
        equity_events.append((times[-1], round(capital, 2)))

    # ── Build outputs ──
    cols = ["entry_time", "exit_time", "side", "avg_entry",
            "exit_price", "n_orders", "invested_usd", "pnl_usd", "pnl_pct"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=cols)

    eq_idx   = pd.DatetimeIndex([e[0] for e in equity_events])
    eq_vals  = [e[1] for e in equity_events]
    equity   = pd.Series(eq_vals, index=eq_idx, name="equity")
    equity   = equity[~equity.index.duplicated(keep="last")]

    from backtester.engine.metrics import compute_metrics
    returns = equity.pct_change().fillna(0)
    metrics = compute_metrics(returns, equity, trades_df, initial_capital)

    return SRGridResult(
        equity=equity,
        trades=trades_df,
        metrics=metrics,
        sr_levels=sr_d1,
        symbol=symbol,
    )


# ── Memory-efficient chunked version ──────────────────────────────────────────

def run_sr_grid_backtest_chunked(
    cache: Any,
    df_d1: pd.DataFrame,
    symbol: str,
    start: "datetime",
    end: "datetime",
    initial_capital: float = 10_000.0,
    order_size_pct: float = 0.03,
    first_avg_step: float = 0.04,
    second_avg_step: float = 0.04,
    subsequent_step: float = 0.02,
    take_profit_pct: float = 0.03,
    max_orders: int = 10,
    lookback_d1: int = 30,
    entry_tolerance: float = 0.005,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> SRGridResult:
    """
    Memory-efficient S/R grid backtest: loads 1s bars one month at a time
    from DuckDB instead of keeping all data in RAM.

    Parameters
    ----------
    cache   : DataCache instance
    df_d1   : Daily OHLCV DataFrame (DatetimeIndex UTC)
    symbol  : e.g. "BTC/USDT"
    start   : backtest start datetime (UTC)
    end     : backtest end datetime (UTC)
    """
    from datetime import datetime

    sr_d1     = _build_sr(df_d1, lookback_d1).dropna()
    order_usd = initial_capital * order_size_pct

    # LONG grid factors: price drops from entry  [1.0, 0.96, 0.92, 0.90, ...]
    factors_long = np.ones(max_orders, dtype=np.float64)
    if max_orders > 1:
        factors_long[1] = 1.0 - first_avg_step
    if max_orders > 2:
        factors_long[2] = factors_long[1] - second_avg_step
    for k in range(3, max_orders):
        factors_long[k] = factors_long[k - 1] - subsequent_step

    # SHORT grid factors: price rises from entry [1.0, 1.04, 1.08, 1.10, ...]
    factors_short = np.ones(max_orders, dtype=np.float64)
    if max_orders > 1:
        factors_short[1] = 1.0 + first_avg_step
    if max_orders > 2:
        factors_short[2] = factors_short[1] + second_avg_step
    for k in range(3, max_orders):
        factors_short[k] = factors_short[k - 1] + subsequent_step

    # ── Persistent simulation state across months ──
    trades         = []
    capital        = initial_capital
    in_grid        = False
    side           = 0        # 1 = long, -1 = short
    n_orders       = 0
    total_qty      = 0.0
    total_invested = 0.0      # sum of USD put in (cost basis, excl. fees)
    grid_prices    = np.zeros(max_orders)
    entry_time_val = None
    equity_events  = []
    first_ts       = None
    last_close     = 0.0
    times          = None

    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    while current < end:
        next_month = (current + timedelta(days=32)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        chunk_end = min(next_month, end)

        print(f"  Simulating {current.strftime('%Y-%m')}...", end="\r")

        from backtester.data.manager import load_1s_month
        df_chunk = load_1s_month(symbol, current.year, current.month)
        if df_chunk.empty:
            current = next_month
            continue

        # Filter to requested date range within the month
        ts_start = pd.Timestamp(current).tz_localize("UTC") if current.tzinfo is None else pd.Timestamp(current)
        ts_end   = pd.Timestamp(chunk_end).tz_localize("UTC") if chunk_end.tzinfo is None else pd.Timestamp(chunk_end)
        df_chunk = df_chunk.loc[df_chunk.index >= ts_start]
        if chunk_end < end:
            df_chunk = df_chunk.loc[df_chunk.index < ts_end]

        if df_chunk.empty:
            current = next_month
            continue

        sr_chunk = sr_d1.reindex(df_chunk.index, method="ffill").dropna()
        if sr_chunk.empty:
            current = next_month
            continue

        df_sim    = df_chunk.reindex(sr_chunk.index)
        high_arr  = df_sim["high"].values.astype(np.float64)
        low_arr   = df_sim["low"].values.astype(np.float64)
        close_arr = df_sim["close"].values.astype(np.float64)
        sr_arr    = sr_chunk.values.astype(np.float64)
        times     = sr_chunk.index
        n         = len(times)

        if first_ts is None and n > 0:
            first_ts = times[0]
            equity_events.append((first_ts, initial_capital))

        for i in range(n):
            sr = sr_arr[i]
            h  = high_arr[i]
            l  = low_arr[i]
            c  = close_arr[i]

            if in_grid:
                avg_cost = total_invested / total_qty

                if side == 1:   # ── LONG ──
                    tp_price = avg_cost * (1.0 + take_profit_pct)
                    if h >= tp_price:
                        exit_price  = tp_price * (1.0 - slippage)
                        proceeds    = total_qty * exit_price
                        exit_fee    = proceeds * commission
                        capital    += proceeds - exit_fee
                        pnl_usd     = (exit_price - avg_cost) * total_qty - exit_fee
                        pnl_pct     = pnl_usd / total_invested * 100.0
                        trades.append(_grid_trade(entry_time_val, times[i], "long",
                                                   avg_cost, exit_price, n_orders,
                                                   total_invested, pnl_usd, pnl_pct))
                        equity_events.append((times[i], round(capital, 2)))
                        in_grid = False; n_orders = 0; total_qty = 0.0
                        total_invested = 0.0; entry_time_val = None; side = 0
                    elif n_orders < max_orders:
                        while n_orders < max_orders and l <= grid_prices[n_orders]:
                            fp   = grid_prices[n_orders] * (1.0 + slippage)
                            qty  = order_usd / fp
                            fee  = order_usd * commission
                            capital -= (order_usd + fee)
                            total_qty += qty; total_invested += order_usd; n_orders += 1

                else:           # ── SHORT ──
                    tp_price = avg_cost * (1.0 - take_profit_pct)
                    if l <= tp_price:
                        exit_price  = tp_price * (1.0 + slippage)
                        proceeds    = total_qty * (avg_cost - exit_price)
                        exit_fee    = total_invested * commission
                        capital    += total_invested + proceeds - exit_fee
                        pnl_usd     = proceeds - exit_fee
                        pnl_pct     = pnl_usd / total_invested * 100.0
                        trades.append(_grid_trade(entry_time_val, times[i], "short",
                                                   avg_cost, exit_price, n_orders,
                                                   total_invested, pnl_usd, pnl_pct))
                        equity_events.append((times[i], round(capital, 2)))
                        in_grid = False; n_orders = 0; total_qty = 0.0
                        total_invested = 0.0; entry_time_val = None; side = 0
                    elif n_orders < max_orders:
                        while n_orders < max_orders and h >= grid_prices[n_orders]:
                            fp   = grid_prices[n_orders] * (1.0 - slippage)
                            qty  = order_usd / fp
                            fee  = order_usd * commission
                            capital -= (order_usd + fee)
                            total_qty += qty; total_invested += order_usd; n_orders += 1

            else:
                # ── Entry when price is within tolerance of S/R ──
                if sr > 0.0 and abs(c - sr) / sr <= entry_tolerance:
                    if c >= sr:                     # price at/above S/R → SHORT
                        fp              = c * (1.0 - slippage)
                        qty             = order_usd / fp
                        fee             = order_usd * commission
                        capital        -= (order_usd + fee)
                        total_qty       = qty
                        total_invested  = order_usd
                        n_orders        = 1
                        entry_time_val  = times[i]
                        in_grid         = True
                        side            = -1
                        grid_prices     = fp * factors_short
                    else:                           # price below S/R → LONG
                        fp              = c * (1.0 + slippage)
                        qty             = order_usd / fp
                        fee             = order_usd * commission
                        capital        -= (order_usd + fee)
                        total_qty       = qty
                        total_invested  = order_usd
                        n_orders        = 1
                        entry_time_val  = times[i]
                        in_grid         = True
                        side            = 1
                        grid_prices     = fp * factors_long

            last_close = c

        current = next_month

    print()  # newline after progress

    # ── Close any open position at end of data ──
    if in_grid and total_qty > 0.0 and last_close > 0.0:
        avg_cost = total_invested / total_qty
        if side == 1:
            exit_price = last_close * (1.0 - slippage)
            proceeds   = total_qty * exit_price
            exit_fee   = proceeds * commission
            capital   += proceeds - exit_fee
            pnl_usd    = (exit_price - avg_cost) * total_qty - exit_fee
        else:
            exit_price = last_close * (1.0 + slippage)
            proceeds   = total_qty * (avg_cost - exit_price)
            exit_fee   = total_invested * commission
            capital   += total_invested + proceeds - exit_fee
            pnl_usd    = proceeds - exit_fee

        pnl_pct = pnl_usd / total_invested * 100.0
        t_end   = times[-1] if times is not None else end
        trades.append(_grid_trade(entry_time_val, t_end,
                                  "long" if side == 1 else "short",
                                  avg_cost, exit_price, n_orders,
                                  total_invested, pnl_usd, pnl_pct))
        equity_events.append((t_end, round(capital, 2)))

    # ── Build outputs ──
    cols = ["entry_time", "exit_time", "side", "avg_entry",
            "exit_price", "n_orders", "invested_usd", "pnl_usd", "pnl_pct"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=cols)

    if equity_events:
        eq_idx = pd.DatetimeIndex([e[0] for e in equity_events])
        eq_vals = [e[1] for e in equity_events]
        equity = pd.Series(eq_vals, index=eq_idx, name="equity")
        equity = equity[~equity.index.duplicated(keep="last")]
    else:
        equity = pd.Series([initial_capital], index=pd.DatetimeIndex([start]))

    from backtester.engine.metrics import compute_metrics
    returns = equity.pct_change().fillna(0)
    metrics = compute_metrics(returns, equity, trades_df, initial_capital)

    return SRGridResult(
        equity=equity,
        trades=trades_df,
        metrics=metrics,
        sr_levels=sr_d1,
        symbol=symbol,
    )
