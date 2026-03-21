"""
Backtest: «Институциональный поток» — Anchored VWAP + ADX(14) + ATR(14)
on BTC/USDT

Capital       : $10 000
Period        : 2020-01-01 → now
Timeframes    : 4h and 1h

Risk Management
---------------
- Position size  : 1% of current balance per trade (risk-based sizing)
- Stop-Loss      : 1.5 × ATR(14) from entry price
- Take-Profit    : 3 × SL distance from entry  (minimum RR 1:3)
- Weekly DD gate : if intra-week equity loss >= 5%, trading is halted
                   and resumes only on the following Monday (00:00 UTC)

Entry conditions
----------------
Long  : candle low <= VWAP AND close > VWAP  AND  ADX > 25  (bounce from below)
Short : candle high >= VWAP AND close < VWAP  AND  ADX > 25  (rejection from above)

Fees  : 0.1% commission + 0.05% slippage per side (Binance taker)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import get_ohlcv
from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.strategy.vwap_adx_atr import VwapAdxAtr
from backtester.visualization.charts import plot_backtest


# ── Risk-based position sizing ────────────────────────────────────────────────

def _pos_fraction(atr_val: float, entry_price: float,
                  risk_pct: float, atr_sl_mult: float) -> float:
    """
    Fraction of capital to commit so that a full SL hit costs exactly risk_pct.
    fraction = risk_pct / (sl_distance / entry_price). Capped at 1.0.
    """
    sl_dist = atr_sl_mult * atr_val
    if sl_dist <= 0 or entry_price <= 0:
        return 0.0
    sl_frac = sl_dist / entry_price
    return min(risk_pct / sl_frac, 1.0)


# ── Bar-by-bar simulation ─────────────────────────────────────────────────────

def run_vwap_backtest(
    df: pd.DataFrame,
    strategy: VwapAdxAtr,
    initial_capital: float = 10_000.0,
    risk_pct: float = 0.01,          # 1% risk per trade
    rr_ratio: float = 3.0,           # TP = rr_ratio × SL distance
    weekly_dd_limit: float = 0.05,   # 5% weekly drawdown gate
    commission: float = 0.001,
    slippage: float = 0.0005,
    symbol: str = "",
    timeframe: str = "",
) -> BacktestResult:
    """
    Bar-by-bar simulation:
      - No look-ahead: signal at bar i triggers entry at bar i+1
      - Position size  = 1% risk / (ATR-based SL fraction)
      - SL  = entry +/- 1.5 x ATR
      - TP  = entry +/- 3 x SL distance  (RR 1:3)
      - Weekly 5% drawdown gate
    """
    raw_signals = strategy.generate_signals(df).reindex(df.index).fillna(0)

    atr_arr  = strategy.atr_line.values
    signals  = raw_signals.shift(1).fillna(0).values.astype(int)

    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    times     = df.index
    n         = len(times)

    trades      = []
    capital     = initial_capital
    equity_arr  = np.empty(n)
    returns_arr = np.zeros(n)
    pos_arr     = np.zeros(n)

    in_trade    = False
    side        = 0
    entry_idx   = 0
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    pos_frac    = 0.0
    block_entry = False

    # ── Weekly drawdown gate ──────────────────────────────────────────────────
    def _iso_week(ts: pd.Timestamp) -> int:
        iso = ts.isocalendar()
        return iso[0] * 100 + iso[1]

    current_week      = _iso_week(times[0]) if n > 0 else 0
    week_start_equity = capital
    halted_until      = None   # pd.Timestamp or None

    for i in range(n):
        ts  = times[i]
        sig = signals[i]
        c   = close_arr[i]
        h   = high_arr[i]
        l   = low_arr[i]
        bar_return = 0.0

        # ── Reset weekly anchor on new ISO-week ──
        wk = _iso_week(ts)
        if wk != current_week:
            current_week      = wk
            week_start_equity = capital

        # ── Lift halt when we've reached Monday ──
        if halted_until is not None and ts >= halted_until:
            halted_until = None

        trading_allowed = (halted_until is None)

        # ── Manage open trade ────────────────────────────────────────────────
        if in_trade:
            pos_arr[i] = side
            exit_price = None
            forced     = False
            exit_tag   = "signal"

            if side == 1:   # long: SL below, TP above
                if l <= sl_price:
                    exit_price = sl_price * (1 - slippage)
                    forced     = True
                    exit_tag   = "sl"
                elif h >= tp_price:
                    exit_price = tp_price * (1 - slippage)
                    forced     = True
                    exit_tag   = "tp"
            else:           # short: SL above, TP below
                if h >= sl_price:
                    exit_price = sl_price * (1 + slippage)
                    forced     = True
                    exit_tag   = "sl"
                elif l <= tp_price:
                    exit_price = tp_price * (1 + slippage)
                    forced     = True
                    exit_tag   = "tp"

            # Signal reversal or final bar
            if not forced and (sig == -side or i == n - 1):
                exit_price = c * (1 - side * slippage)

            if exit_price is not None:
                raw_ret = side * (exit_price / entry_price - 1)
                net_pnl = pos_frac * raw_ret - 2 * (commission + slippage)
                bar_return += net_pnl
                capital   *= (1 + net_pnl)

                trades.append({
                    "entry_time":  times[entry_idx],
                    "exit_time":   ts,
                    "side":        "long" if side == 1 else "short",
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(exit_price, 2),
                    "sl_price":    round(sl_price, 2),
                    "tp_price":    round(tp_price, 2),
                    "exit_reason": exit_tag,
                    "pos_frac":    round(pos_frac, 4),
                    "pnl_pct":     round(net_pnl * 100, 4),
                    "duration":    ts - times[entry_idx],
                })

                in_trade    = False
                block_entry = forced

                # Check weekly drawdown after close
                weekly_dd = (capital - week_start_equity) / week_start_equity
                if weekly_dd <= -weekly_dd_limit and halted_until is None:
                    next_monday     = (ts + pd.offsets.Week(weekday=0)).normalize()
                    halted_until    = next_monday
                    trading_allowed = False

        # ── New entry ────────────────────────────────────────────────────────
        if not in_trade and not block_entry and sig != 0 and trading_allowed:
            atr = atr_arr[i]
            if np.isnan(atr) or atr <= 0:
                block_entry = False
                equity_arr[i]  = capital
                returns_arr[i] = bar_return
                continue

            ep      = c * (1 + sig * slippage)
            sl_dist = strategy.atr_sl_mult * atr

            if sig == 1:
                sl = ep - sl_dist
                tp = ep + rr_ratio * sl_dist
            else:
                sl = ep + sl_dist
                tp = ep - rr_ratio * sl_dist

            # Sanity: skip data spikes wider than 15%
            if sl_dist / ep > 0.15 or np.isnan(sl):
                block_entry = False
                equity_arr[i]  = capital
                returns_arr[i] = bar_return
                continue

            pf = _pos_fraction(atr, ep, risk_pct, strategy.atr_sl_mult)
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
            pos_arr[i]  = side

        block_entry    = False
        returns_arr[i] = bar_return
        equity_arr[i]  = capital

    # ── Build result ─────────────────────────────────────────────────────────
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "entry_price", "exit_price",
                 "sl_price", "tp_price", "exit_reason", "pos_frac", "pnl_pct", "duration"]
    )

    equity    = pd.Series(equity_arr,  index=df.index)
    returns   = pd.Series(returns_arr, index=df.index)
    positions = pd.Series(pos_arr,     index=df.index)
    metrics   = compute_metrics(returns, equity, trades_df, initial_capital)

    return BacktestResult(
        equity=equity,
        returns=returns,
        positions=positions,
        trades=trades_df,
        metrics=metrics,
        params=strategy.get_params(),
        symbol=symbol,
        timeframe=timeframe,
    )


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_trade_breakdown(result: BacktestResult) -> None:
    trades = result.trades
    if trades.empty:
        print("  No trades.")
        return

    by_side = trades.groupby("side")["pnl_pct"].agg(
        ["count", "mean", lambda x: (x > 0).mean() * 100]
    )
    by_side.columns = ["trades", "avg_pnl%", "win%"]
    print(f"\n  By side:\n{by_side.to_string()}")

    if "exit_reason" in trades.columns:
        by_exit = trades.groupby("exit_reason")["pnl_pct"].agg(["count", "mean"])
        by_exit.columns = ["trades", "avg_pnl%"]
        print(f"\n  By exit reason:\n{by_exit.to_string()}")

    print(f"\n  Best trade  : {trades['pnl_pct'].max():.2f}%")
    print(f"  Worst trade : {trades['pnl_pct'].min():.2f}%")
    if "duration" in trades.columns:
        print(f"  Avg hold    : {trades['duration'].mean()}")
    if "pos_frac" in trades.columns:
        print(f"  Avg pos frac: {trades['pos_frac'].mean():.2%}")


def print_weekly_dd_summary(result: BacktestResult) -> None:
    trades = result.trades
    if trades.empty:
        return
    t = trades.copy()
    t["week"] = pd.to_datetime(t["exit_time"]).apply(
        lambda ts: ts.isocalendar()[0] * 100 + ts.isocalendar()[1]
    )
    weekly_pnl = t.groupby("week")["pnl_pct"].sum()
    bad_weeks  = weekly_pnl[weekly_pnl <= -4.5]
    if not bad_weeks.empty:
        print(f"\n  Weeks near 5% weekly DD gate ({len(bad_weeks)} weeks):")
        print(bad_weeks.to_string())
    else:
        print("\n  Weekly 5% drawdown gate was never triggered.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SYMBOL      = "BTC/USDT"
    SINCE       = "2020-01-01"
    CAPITAL     = 10_000.0
    COMMISSION  = 0.001     # 0.1% Binance taker
    SLIPPAGE    = 0.0005    # 0.05%
    RISK_PCT    = 0.01      # 1% balance at risk per trade
    RR_RATIO    = 3.0       # TP = 3 x SL distance
    WEEKLY_DD   = 0.05      # 5% weekly drawdown gate

    strategy = VwapAdxAtr(
        swing_period   = 20,
        adx_period     = 14,
        adx_threshold  = 25.0,
        adx_exit_level = 20.0,
        atr_period     = 14,
        atr_sl_mult    = 1.5,   # SL = 1.5 x ATR
    )

    results = {}

    for tf in ["4h", "1h"]:
        print(f"\n{'═' * 60}")
        print(f"  Downloading {SYMBOL} {tf}  from {SINCE} ...")
        df = get_ohlcv(SYMBOL, tf, since=SINCE)
        print(f"  Loaded {len(df):,} candles  ({df.index[0].date()} -> {df.index[-1].date()})")

        result = run_vwap_backtest(
            df,
            strategy,
            initial_capital = CAPITAL,
            risk_pct        = RISK_PCT,
            rr_ratio        = RR_RATIO,
            weekly_dd_limit = WEEKLY_DD,
            commission      = COMMISSION,
            slippage        = SLIPPAGE,
            symbol          = SYMBOL,
            timeframe       = tf,
        )
        results[tf] = (result, df)

        print(f"\n{result.report()}")
        print_trade_breakdown(result)
        print_weekly_dd_summary(result)

        if not result.trades.empty:
            cols = ["entry_time", "exit_time", "side",
                    "entry_price", "exit_price", "exit_reason", "pnl_pct"]
            print(f"\n  First 10 trades:\n{result.trades[cols].head(10).to_string(index=False)}")

        html = f"vwap_adx_atr_{tf}.html"
        plot_backtest(
            result, df,
            indicators={"Anchored VWAP": strategy.vwap_line},
            title=(
                f"{SYMBOL} {tf} - Anchored VWAP + ADX(14,>=25) + ATR(14) | "
                f"SL=1.5xATR  TP=1:3  Risk=1%  WeeklyDD<=5% | 2020->"
            ),
            save_html=html,
        )
        print(f"\n  Chart -> {html}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  COMPARISON  4h  vs  1h")
    print(f"{'═' * 60}")
    keys = [
        "total_return_pct", "cagr_pct", "sharpe_ratio",
        "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades",
    ]
    print(f"  {'Metric':<28} {'4h':>12} {'1h':>12}")
    print(f"  {'-' * 52}")
    for k in keys:
        v4 = results["4h"][0].metrics.get(k, "-")
        v1 = results["1h"][0].metrics.get(k, "-")
        label = k.replace("_", " ").title()
        print(f"  {label:<28} {str(v4):>12} {str(v1):>12}")
    print(f"{'═' * 60}")
