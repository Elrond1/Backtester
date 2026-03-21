"""
Optimization: Anchored VWAP + ADX + ATR — BTC/USDT 4h
------------------------------------------------------
Grid-searches the following parameters:

  adx_threshold      : [25, 30, 35]
  swing_period       : [50, 100, 200]
  cooldown_bars      : [10, 20, 30]
  min_vwap_dist_pct  : [0.002, 0.005, 0.01]
  pos_frac_cap       : [0.15, 0.20, 0.25]   (max position size vs capital)
  ema_filter         : [True, False]         (long only > EMA200, short only < EMA200)

Fixed:
  atr_sl_mult = 1.5  |  rr_ratio = 3.0  |  weekly_dd_limit = 5%
  risk_pct = 1%      |  commission = 0.1%  |  slippage = 0.05%

Sorted by Sharpe ratio. Top-20 printed + saved to vwap_opt_results.csv.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from backtester.data import get_ohlcv
from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.strategy.vwap_adx_atr import VwapAdxAtr
from backtester.strategy.indicators import ema as _ema


# ── Runner (same as vwap_adx_atr_backtest.py + pos_frac_cap + ema_filter) ─────

def _pos_fraction(atr_val: float, entry_price: float,
                  risk_pct: float, atr_sl_mult: float,
                  pos_frac_cap: float) -> float:
    sl_dist = atr_sl_mult * atr_val
    if sl_dist <= 0 or entry_price <= 0:
        return 0.0
    sl_frac = sl_dist / entry_price
    return min(risk_pct / sl_frac, pos_frac_cap)


def run_backtest_opt(
    df: pd.DataFrame,
    strategy: VwapAdxAtr,
    ema200: pd.Series,
    initial_capital: float = 10_000.0,
    risk_pct: float = 0.01,
    rr_ratio: float = 3.0,
    weekly_dd_limit: float = 0.05,
    commission: float = 0.001,
    slippage: float = 0.0005,
    pos_frac_cap: float = 0.20,
    ema_filter: bool = True,
) -> dict:
    raw_signals = strategy.generate_signals(df).reindex(df.index).fillna(0)

    atr_arr    = strategy.atr_line.values
    ema200_arr = ema200.values
    signals    = raw_signals.shift(1).fillna(0).values.astype(int)

    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    times     = df.index
    n         = len(times)

    trades      = []
    capital     = initial_capital
    equity_arr  = np.empty(n)
    returns_arr = np.zeros(n)

    in_trade    = False
    side        = 0
    entry_idx   = 0
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    pos_frac    = 0.0
    block_entry = False

    def _iso_week(ts: pd.Timestamp) -> int:
        iso = ts.isocalendar()
        return iso[0] * 100 + iso[1]

    current_week      = _iso_week(times[0]) if n > 0 else 0
    week_start_equity = capital
    halted_until      = None

    for i in range(n):
        ts  = times[i]
        sig = signals[i]
        c   = close_arr[i]
        h   = high_arr[i]
        l   = low_arr[i]
        bar_return = 0.0

        wk = _iso_week(ts)
        if wk != current_week:
            current_week      = wk
            week_start_equity = capital

        if halted_until is not None and ts >= halted_until:
            halted_until = None

        trading_allowed = (halted_until is None)

        # ── EMA filter ──
        if ema_filter and not np.isnan(ema200_arr[i]):
            if sig == 1 and c < ema200_arr[i]:
                sig = 0
            elif sig == -1 and c > ema200_arr[i]:
                sig = 0

        if in_trade:
            exit_price = None
            forced     = False
            exit_tag   = "signal"

            if side == 1:
                if l <= sl_price:
                    exit_price = sl_price * (1 - slippage)
                    forced = True; exit_tag = "sl"
                elif h >= tp_price:
                    exit_price = tp_price * (1 - slippage)
                    forced = True; exit_tag = "tp"
            else:
                if h >= sl_price:
                    exit_price = sl_price * (1 + slippage)
                    forced = True; exit_tag = "sl"
                elif l <= tp_price:
                    exit_price = tp_price * (1 + slippage)
                    forced = True; exit_tag = "tp"

            if not forced and (sig == -side or i == n - 1):
                exit_price = c * (1 - side * slippage)

            if exit_price is not None:
                raw_ret = side * (exit_price / entry_price - 1)
                net_pnl = pos_frac * raw_ret - 2 * (commission + slippage)
                bar_return += net_pnl
                capital   *= (1 + net_pnl)
                trades.append({"exit_reason": exit_tag, "pnl_pct": net_pnl * 100})
                in_trade    = False
                block_entry = forced

                weekly_dd = (capital - week_start_equity) / week_start_equity
                if weekly_dd <= -weekly_dd_limit and halted_until is None:
                    halted_until    = (ts + pd.offsets.Week(weekday=0)).normalize()
                    trading_allowed = False

        if not in_trade and not block_entry and sig != 0 and trading_allowed:
            atr = atr_arr[i]
            if np.isnan(atr) or atr <= 0:
                block_entry = False
                equity_arr[i]  = capital
                returns_arr[i] = bar_return
                continue

            ep      = c * (1 + sig * slippage)
            sl_dist = strategy.atr_sl_mult * atr
            if sl_dist / ep > 0.15:
                block_entry = False
                equity_arr[i]  = capital
                returns_arr[i] = bar_return
                continue

            sl = ep - sig * sl_dist
            tp = ep + sig * rr_ratio * sl_dist
            pf = _pos_fraction(atr, ep, risk_pct, strategy.atr_sl_mult, pos_frac_cap)
            if pf <= 0:
                block_entry = False
                equity_arr[i]  = capital
                returns_arr[i] = bar_return
                continue

            cost = pf * (commission + slippage)
            bar_return -= cost
            capital    *= (1 - cost)

            in_trade    = True
            side        = sig
            entry_idx   = i
            entry_price = ep
            sl_price    = sl
            tp_price    = tp
            pos_frac    = pf

        block_entry    = False
        returns_arr[i] = bar_return
        equity_arr[i]  = capital

    equity  = pd.Series(equity_arr,  index=df.index)
    returns = pd.Series(returns_arr, index=df.index)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["exit_reason", "pnl_pct"]
    )
    m = compute_metrics(returns, equity, trades_df, initial_capital)

    win_rate = (
        (trades_df["pnl_pct"] > 0).mean() * 100
        if not trades_df.empty else 0.0
    )
    sl_rate = (
        (trades_df["exit_reason"] == "sl").mean() * 100
        if "exit_reason" in trades_df.columns and not trades_df.empty else 0.0
    )

    return {
        **m,
        "total_trades":    len(trades_df),
        "win_rate_pct":    round(win_rate, 2),
        "sl_rate_pct":     round(sl_rate, 2),
        "final_capital":   round(capital, 2),
    }


# ── Grid search ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SYMBOL  = "BTC/USDT"
    SINCE   = "2020-01-01"
    TF      = "4h"

    print(f"Downloading {SYMBOL} {TF} from {SINCE} ...")
    df = get_ohlcv(SYMBOL, TF, since=SINCE)
    print(f"Loaded {len(df):,} candles  ({df.index[0].date()} -> {df.index[-1].date()})")

    ema200 = _ema(df["close"], 200)

    param_grid = {
        "adx_threshold":     [25, 30, 35],
        "swing_period":      [50, 100, 200],
        "cooldown_bars":     [10, 20, 30],
        "min_vwap_dist_pct": [0.002, 0.005, 0.01],
        "pos_frac_cap":      [0.15, 0.20, 0.25],
        "ema_filter":        [True, False],
    }

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"\nGrid size: {len(combos)} combinations\n")

    rows = []
    for combo in tqdm(combos, desc="Optimizing"):
        params = dict(zip(keys, combo))

        strategy = VwapAdxAtr(
            swing_period      = params["swing_period"],
            adx_period        = 14,
            adx_threshold     = params["adx_threshold"],
            adx_exit_level    = 20.0,
            atr_period        = 14,
            atr_sl_mult       = 1.5,
            cooldown_bars     = params["cooldown_bars"],
            min_vwap_dist_pct = params["min_vwap_dist_pct"],
        )

        metrics = run_backtest_opt(
            df,
            strategy,
            ema200,
            initial_capital  = 10_000.0,
            risk_pct         = 0.01,
            rr_ratio         = 3.0,
            weekly_dd_limit  = 0.05,
            commission       = 0.001,
            slippage         = 0.0005,
            pos_frac_cap     = params["pos_frac_cap"],
            ema_filter       = params["ema_filter"],
        )

        rows.append({**params, **metrics})

    results = pd.DataFrame(rows)

    # ── Sort and display ──────────────────────────────────────────────────────
    results = results.sort_values("sharpe_ratio", ascending=False)

    display_cols = [
        "adx_threshold", "swing_period", "cooldown_bars",
        "min_vwap_dist_pct", "pos_frac_cap", "ema_filter",
        "sharpe_ratio", "total_return_pct", "max_drawdown_pct",
        "win_rate_pct", "profit_factor", "total_trades", "final_capital",
    ]
    display_cols = [c for c in display_cols if c in results.columns]

    print(f"\n{'═' * 80}")
    print("  TOP-20 by Sharpe Ratio")
    print(f"{'═' * 80}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(results[display_cols].head(20).to_string(index=False))

    # ── Also show top-5 by Total Return ──────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("  TOP-5 by Total Return %")
    print(f"{'═' * 80}")
    print(results.sort_values("total_return_pct", ascending=False)[display_cols].head(5).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_csv = "vwap_opt_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nFull results saved -> {out_csv}")
    print(f"Total profitable combos: {(results['total_return_pct'] > 0).sum()} / {len(results)}")
