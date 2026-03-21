"""
Grid backtester — bar-by-bar simulation for grid entry strategies.

Grid mechanics:
  - Trigger bar: Level 1 filled immediately at bar close (market order).
  - Levels 2-4:  limit orders placed below (long) or above (short).
      L2: step1 away from L1
      L3: step2 away from L2
      L4: step2 away from L3
  - Each level: size_pct of capital at time of trigger.
  - Exit: TP when price reaches avg_entry ± tp_pct (all levels at once).
  - Safety exit: close at market after max_hold_bars if TP not reached.
  - Only one grid open at a time.

Equity tracked mark-to-market every bar (unrealized P&L visible in equity curve).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtester.engine.metrics import compute_metrics


@dataclass
class GridBacktestResult:
    equity:    pd.Series
    returns:   pd.Series
    positions: pd.Series
    trades:    pd.DataFrame
    metrics:   dict
    params:    dict
    symbol:    str = ""
    timeframe: str = ""

    def report(self) -> str:
        lines = [
            f"{'='*46}",
            f"  Grid Backtest Report — {self.symbol} {self.timeframe}",
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


def run_grid_backtest(
    df: pd.DataFrame,
    trigger_signals: pd.Series,
    initial_capital:    float = 10_000.0,
    commission:         float = 0.001,
    slippage:           float = 0.0005,
    size_l1:            float = 0.01,    # level 1 (trigger): 1% of capital
    size_l2:            float = 0.01,    # level 2: 1% of capital
    size_rest:          float = 0.012,   # levels 3-25: 1.2% of capital each
    step:               float = 0.012,   # 1.2% gap between ALL levels
    n_levels:           int   = 25,
    tp_pct:             float = 0.013,   # exit at +1.3% from avg entry
    sl_pct:             float = 0.0,     # SL from avg entry (0 = disabled)
    max_hold_bars:      int   = 500,
    max_grid_loss_pct:  float = 0.0,     # close grid if unrealized loss > X% of initial_capital (0=off)
    symbol:             str   = "",
    timeframe:          str   = "",
) -> GridBacktestResult:
    """
    Bar-by-bar grid backtest with mark-to-market equity tracking.

    Grid sizing:
      Level 1 (trigger)  : size_l1  of capital
      Level 2            : size_l2  of capital
      Levels 3..n_levels : size_rest of capital each
    All levels spaced `step` apart.
    Exit: TP when price reaches avg_entry × (1 ± tp_pct).
    Safety: close after max_hold_bars if TP/SL not hit.
    """
    signals = trigger_signals.reindex(df.index).fillna(0).values
    close   = df["close"].values
    high    = df["high"].values
    low     = df["low"].values
    times   = df.index
    n       = len(times)

    capital     = initial_capital   # realised cash (not counting open positions)
    equity_arr  = np.empty(n)
    returns_arr = np.zeros(n)
    pos_arr     = np.zeros(n)
    trades      = []

    # Grid state
    in_grid        = False
    grid_side      = 0
    grid_entry_idx = 0
    filled_levels  = []   # list of (fill_price, usd_size)
    pending        = []   # list of (limit_price, usd_size)
    avg_entry      = 0.0
    tp_price       = 0.0
    total_invested = 0.0  # sum of USD committed to filled levels

    def _avg_fill(filled):
        total_cost = sum(p * s for p, s in filled)
        total_size = sum(s for _, s in filled)
        return total_cost / total_size if total_size > 0 else 0.0

    def _close_all(exit_price: float, exit_idx: int, reason: str):
        nonlocal capital, in_grid, grid_side, filled_levels, pending
        nonlocal avg_entry, tp_price, total_invested

        avg_p    = _avg_fill(filled_levels)
        invested = sum(s for _, s in filled_levels)
        raw_pnl  = grid_side * (exit_price / avg_p - 1)
        net_pnl  = raw_pnl - 2 * (commission + slippage)
        pnl_usd  = net_pnl * invested

        capital += invested + pnl_usd   # return principal + profit

        trades.append({
            "entry_time":    times[grid_entry_idx],
            "exit_time":     times[exit_idx],
            "side":          "long" if grid_side == 1 else "short",
            "avg_entry":     round(avg_p, 4),
            "exit_price":    round(exit_price, 4),
            "levels_filled": len(filled_levels),
            "invested_usd":  round(invested, 2),
            "pnl_pct":       round(net_pnl * 100, 4),
            "pnl_usd":       round(pnl_usd, 2),
            "reason":        reason,
            "duration":      times[exit_idx] - times[grid_entry_idx],
        })

        in_grid        = False
        grid_side      = 0
        filled_levels  = []
        pending        = []
        avg_entry      = 0.0
        tp_price       = 0.0
        total_invested = 0.0

    prev_equity = initial_capital

    for i in range(n):
        c, h, l = close[i], high[i], low[i]

        if in_grid:
            pos_arr[i] = grid_side
            bars_held  = i - grid_entry_idx

            # ── Check TP ────────────────────────────────────────────────
            sl_price = avg_entry * (1 - grid_side * sl_pct) if sl_pct > 0 else None
            tp_hit   = (grid_side == 1 and h >= tp_price) or \
                       (grid_side == -1 and l <= tp_price)
            sl_hit   = sl_pct > 0 and (
                (grid_side == 1  and l <= sl_price) or
                (grid_side == -1 and h >= sl_price)
            )

            if tp_hit:
                exit_p = tp_price * (1 - grid_side * slippage)
                _close_all(exit_p, i, "TP")

            elif sl_hit:
                exit_p = sl_price * (1 - grid_side * slippage)
                _close_all(exit_p, i, "SL")

            elif bars_held >= max_hold_bars:
                # Safety exit at bar close
                exit_p = c * (1 - grid_side * slippage)
                _close_all(exit_p, i, "timeout")

            elif max_grid_loss_pct > 0 and filled_levels:
                # Equity-based stop: close if unrealized loss > threshold
                avg_p    = _avg_fill(filled_levels)
                invested = sum(s for _, s in filled_levels)
                unreal   = grid_side * (c / avg_p - 1) * invested
                if unreal < -max_grid_loss_pct * initial_capital:
                    exit_p = c * (1 - grid_side * slippage)
                    _close_all(exit_p, i, "grid_stop")

            else:
                # ── Check pending limit orders ───────────────────────
                still_pending = []
                for (lim_p, usd_sz) in pending:
                    filled = (grid_side == 1  and l <= lim_p) or \
                             (grid_side == -1 and h >= lim_p)
                    if filled:
                        fill_p = lim_p * (1 + grid_side * slippage)
                        cost   = usd_sz * (commission + slippage)
                        capital -= (usd_sz + cost)
                        total_invested += usd_sz
                        filled_levels.append((fill_p, usd_sz))
                        avg_entry = _avg_fill(filled_levels)
                        tp_price  = avg_entry * (1 + grid_side * tp_pct)
                    else:
                        still_pending.append((lim_p, usd_sz))
                pending = still_pending

        # ── New trigger (only when flat) ─────────────────────────────
        if not in_grid and signals[i] != 0:
            sig    = int(signals[i])
            fill_p = c * (1 + sig * slippage)
            usd_l1 = capital * size_l1
            cost   = usd_l1 * (commission + slippage)

            capital        -= (usd_l1 + cost)
            total_invested  = usd_l1
            filled_levels   = [(fill_p, usd_l1)]
            avg_entry       = fill_p
            tp_price        = avg_entry * (1 + sig * tp_pct)

            # Build pending levels 2..n_levels
            # L2: size_l2,  L3+: size_rest, all spaced `step` apart
            prev_p  = fill_p
            pending = []
            for lvl in range(1, n_levels):
                usd_sz = capital * size_l2 if lvl == 1 else capital * size_rest
                lim_p  = prev_p * (1 - step) if sig == 1 else prev_p * (1 + step)
                pending.append((lim_p, usd_sz))
                prev_p = lim_p

            in_grid        = True
            grid_side      = sig
            grid_entry_idx = i
            pos_arr[i]     = sig

        # ── Mark-to-market equity ────────────────────────────────────
        if in_grid and filled_levels:
            avg_p    = _avg_fill(filled_levels)
            invested = sum(s for _, s in filled_levels)
            unreal   = grid_side * (c / avg_p - 1) * invested
            equity_arr[i] = capital + invested + unreal
        else:
            equity_arr[i] = capital

        # Per-bar return
        returns_arr[i] = equity_arr[i] / prev_equity - 1 if prev_equity > 0 else 0.0
        prev_equity    = equity_arr[i]

    # Force-close any open grid at the last bar
    if in_grid and filled_levels:
        exit_p = close[-1] * (1 - grid_side * slippage)
        _close_all(exit_p, n - 1, "end")
        equity_arr[-1]  = capital
        returns_arr[-1] = equity_arr[-1] / (equity_arr[-2] if n > 1 else initial_capital) - 1

    equity  = pd.Series(equity_arr,  index=df.index)
    returns = pd.Series(returns_arr, index=df.index)
    pos     = pd.Series(pos_arr,     index=df.index)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "avg_entry", "exit_price",
                 "levels_filled", "invested_usd", "pnl_pct", "pnl_usd",
                 "reason", "duration"]
    )

    metrics = compute_metrics(returns, equity, trades_df, initial_capital)

    params = {
        "size_l1": size_l1, "size_l2": size_l2, "size_rest": size_rest,
        "step": step, "n_levels": n_levels, "tp_pct": tp_pct,
        "sl_pct": sl_pct, "max_hold_bars": max_hold_bars,
        "max_grid_loss_pct": max_grid_loss_pct,
    }

    return GridBacktestResult(
        equity=equity, returns=returns, positions=pos,
        trades=trades_df, metrics=metrics, params=params,
        symbol=symbol, timeframe=timeframe,
    )
