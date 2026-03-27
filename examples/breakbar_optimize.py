"""
BreakBar: parameter optimization + fractional position sizing (MM).

Money Management
----------------
Instead of risking 100% of capital per trade, we risk `risk_per_trade` fraction.
Position size scales so that if SL is hit, we lose exactly risk_per_trade × capital.

    position_weight = risk_per_trade / (|entry - SL| / entry)
    net_pnl         = position_weight × trade_raw_return - costs

This lets us:
  - Control max loss per trade independently of the strategy parameters
  - Compound faster when risk_per_trade is higher (but also blow up faster)

Grid search over: min_bar_size, min_bars_in, rr_ratio, risk_per_trade
Sorted by: total_return_pct (max drawdown shown alongside for sanity check)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from backtester.data import get_ohlcv
from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.strategy.breakbar import BreakBar
from backtester.visualization.charts import plot_backtest


# ── Backtest with fractional position sizing ──────────────────────────────────

def run_mm_backtest(
    df: pd.DataFrame,
    strategy: BreakBar,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    risk_per_trade: float = 0.02,   # fraction of capital risked per trade (MM)
    symbol: str = "",
    timeframe: str = "",
) -> BacktestResult:
    """
    BreakBar backtest with risk-based position sizing.

    risk_per_trade = 0.02 means: each trade risks 2% of current equity.
    If SL is 1% away, position size = 2× capital (i.e. 2:1 leverage).
    """
    raw_signals = strategy.generate_signals(df).reindex(df.index).fillna(0)
    zone_sl     = strategy.zone_sl_levels(df)
    sw_low, sw_high = strategy.swing_levels(df)

    signals = raw_signals.shift(1).fillna(0).values.astype(int)
    zone_sl = zone_sl.shift(1).fillna(np.nan).values

    close_arr  = df["close"].values
    high_arr   = df["high"].values
    low_arr    = df["low"].values
    sl_low_arr = sw_low.shift(1).fillna(np.nan).values
    sl_hi_arr  = sw_high.shift(1).fillna(np.nan).values
    times      = df.index
    n          = len(times)

    trades      = []
    capital     = initial_capital
    equity_arr  = np.empty(n)
    returns_arr = np.zeros(n)
    pos_arr     = np.zeros(n)

    in_trade       = False
    side           = 0
    entry_idx      = 0
    entry_price    = 0.0
    sl_price       = 0.0
    tp_price       = 0.0
    pos_weight     = 0.0   # fraction of capital in the trade
    block_entry    = False

    for i in range(n):
        sig        = signals[i]
        bar_return = 0.0
        c, h, l    = close_arr[i], high_arr[i], low_arr[i]

        if in_trade:
            pos_arr[i] = side
            exit_price    = None
            forced        = False
            forced_reason = ""

            if side == 1:
                if l <= sl_price:
                    exit_price = sl_price; forced = True; forced_reason = "sl"
                elif h >= tp_price:
                    exit_price = tp_price; forced = True; forced_reason = "tp"
            else:
                if h >= sl_price:
                    exit_price = sl_price; forced = True; forced_reason = "sl"
                elif l <= tp_price:
                    exit_price = tp_price; forced = True; forced_reason = "tp"

            if not forced and (sig == -side or i == n - 1):
                exit_price = c * (1 - side * slippage)

            if exit_price is not None:
                if forced:
                    exit_price *= (1 - side * slippage)

                raw_return = side * (exit_price / entry_price - 1)
                costs      = pos_weight * 2 * (commission + slippage)
                net_pnl    = pos_weight * raw_return - costs

                bar_return += net_pnl
                capital   *= (1 + net_pnl)

                trades.append({
                    "entry_time":   times[entry_idx],
                    "exit_time":    times[i],
                    "side":         "long" if side == 1 else "short",
                    "entry_price":  round(entry_price, 2),
                    "exit_price":   round(exit_price, 2),
                    "sl_price":     round(sl_price, 2),
                    "tp_price":     round(tp_price, 2),
                    "exit_reason":  forced_reason if forced else "signal",
                    "pos_weight":   round(pos_weight, 4),
                    "pnl_pct":      round(net_pnl * 100, 4),
                    "duration":     times[i] - times[entry_idx],
                })
                in_trade = False
                if forced:
                    block_entry = True

        if not in_trade and not block_entry and sig != 0:
            ep = c * (1 + sig * slippage)

            z_sl = zone_sl[i]
            if sig == 1:
                sl = z_sl if not np.isnan(z_sl) else sl_low_arr[i]
            else:
                sl = z_sl if not np.isnan(z_sl) else sl_hi_arr[i]

            if np.isnan(sl):
                equity_arr[i] = capital; returns_arr[i] = bar_return
                block_entry = False; continue

            dist = (ep - sl) if sig == 1 else (sl - ep)
            if dist <= 0 or dist / ep > 0.30:
                equity_arr[i] = capital; returns_arr[i] = bar_return
                block_entry = False; continue

            # ── MM: size the position so SL loss = risk_per_trade × capital ──
            sl_pct     = dist / ep
            pw         = risk_per_trade / sl_pct        # position weight (can be > 1 = leverage)
            pw         = min(pw, 5.0)                   # cap at 5× leverage

            tp = ep + sig * strategy.rr_ratio * dist

            in_trade    = True
            side        = sig
            entry_idx   = i
            entry_price = ep
            sl_price    = sl
            tp_price    = tp
            pos_weight  = pw
            pos_arr[i]  = side
            entry_cost  = pw * (commission + slippage)
            bar_return -= entry_cost
            capital    *= (1 - entry_cost)

        block_entry    = False
        returns_arr[i] = bar_return
        equity_arr[i]  = capital

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "entry_price", "exit_price",
                 "sl_price", "tp_price", "exit_reason", "pos_weight", "pnl_pct", "duration"]
    )

    equity    = pd.Series(equity_arr,  index=df.index)
    returns   = pd.Series(returns_arr, index=df.index)
    positions = pd.Series(pos_arr,     index=df.index)
    metrics   = compute_metrics(returns, equity, trades_df, initial_capital)

    return BacktestResult(
        equity=equity, returns=returns, positions=positions,
        trades=trades_df, metrics=metrics,
        params={**strategy.get_params(), "risk_per_trade": risk_per_trade},
        symbol=symbol, timeframe=timeframe,
    )


# ── Grid search ───────────────────────────────────────────────────────────────

def optimize(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    param_grid = {
        "min_size_pct":   [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0],
        "min_bars_in":    [2, 3, 4, 5, 6, 8, 10],
        "rr_ratio":       [1.5, 2.0, 2.5, 3.0, 4.0],
        "risk_per_trade": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    }

    keys  = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    print(f"\n  Grid: {len(combos)} combinations on {symbol} {timeframe}")

    rows = []
    for combo in tqdm(combos, desc="Optimizing"):
        params = dict(zip(keys, combo))
        rpt    = params.pop("risk_per_trade")
        try:
            strat = BreakBar(**params)
            result = run_mm_backtest(df, strat, risk_per_trade=rpt,
                                     symbol=symbol, timeframe=timeframe)
            rows.append({**params, "risk_per_trade": rpt, **result.metrics})
        except Exception as e:
            rows.append({**params, "risk_per_trade": rpt, "error": str(e)})

    results = pd.DataFrame(rows)
    if "total_return_pct" in results.columns:
        results = results.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SYMBOL     = "BTC/USDT"
    SINCE      = "2020-01-01"
    CAPITAL    = 10_000.0
    COMMISSION = 0.001
    SLIPPAGE   = 0.0005
    TF         = "4h"

    print(f"  Downloading {SYMBOL} {TF} from {SINCE} …")
    df = get_ohlcv(SYMBOL, TF, since=SINCE)
    print(f"  Loaded {len(df):,} candles  ({df.index[0].date()} → {df.index[-1].date()})")

    results = optimize(df, SYMBOL, TF)

    # Save full results
    csv_path = "breakbar_opt_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n  Full results saved → {csv_path}")

    # Show top 20
    cols = ["min_size_pct", "min_bars_in", "rr_ratio", "risk_per_trade",
            "total_return_pct", "max_drawdown_pct", "win_rate_pct",
            "profit_factor", "sharpe_ratio", "total_trades"]
    show_cols = [c for c in cols if c in results.columns]
    print(f"\n  TOP 20 by Total Return:")
    print(results[show_cols].head(20).to_string(index=False))

    # Run and chart the best config
    best = results.iloc[0]
    print(f"\n  Best config: {best[show_cols[:4]].to_dict()}")

    best_strat = BreakBar(
        min_size_pct = float(best["min_size_pct"]),
        min_bars_in  = int(best["min_bars_in"]),
        rr_ratio     = float(best["rr_ratio"]),
    )
    best_result = run_mm_backtest(
        df, best_strat,
        initial_capital=CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        risk_per_trade=float(best["risk_per_trade"]),
        symbol=SYMBOL,
        timeframe=TF,
    )

    print(f"\n{best_result.report()}")

    html = "breakbar_best.html"
    plot_backtest(
        best_result, df,
        title=(f"{SYMBOL} {TF} — BreakBar BEST  "
               f"min_size={best['min_size_pct']}%  "
               f"bars_in={int(best['min_bars_in'])}  "
               f"RR={best['rr_ratio']}  "
               f"risk={best['risk_per_trade']:.0%}  | {SINCE}→"),
        save_html=html,
        show=False,
    )
    print(f"\n  Chart → {html}")

    import subprocess
    subprocess.Popen(["open", "-a", "Google Chrome", html])
