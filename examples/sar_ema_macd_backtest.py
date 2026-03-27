"""
Backtest: SAR + EMA 200 + MACD with rebound entry on BTC/USDT
- Capital : $10 000
- Period  : 2020-01-01 → now
- TF      : 1h  and  4h
- SL      : nearest swing low / high (rolling 20 bars)
- TP      : 1 : 2 risk/reward
- Fees    : 0.1% commission + 0.05% slippage per side
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import get_ohlcv
from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.strategy.sar_ema_macd import SarEmaMacd
from backtester.strategy.indicators import ema as _ema, parabolic_sar
from backtester.visualization.charts import plot_backtest


def run_dynamic_backtest(
    df: pd.DataFrame,
    strategy: SarEmaMacd,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    symbol: str = "",
    timeframe: str = "",
) -> BacktestResult:
    raw_signals = strategy.generate_signals(df).reindex(df.index).fillna(0)
    sw_low, sw_high = strategy.swing_levels(df)

    # Shift by 1 — enter on next bar open after signal
    signals = raw_signals.shift(1).fillna(0).values.astype(int)

    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    sl_low    = sw_low.values
    sl_high   = sw_high.values
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
    block_entry = False

    for i in range(n):
        sig        = signals[i]
        bar_return = 0.0
        c, h, l    = close_arr[i], high_arr[i], low_arr[i]

        if in_trade:
            pos_arr[i] = side
            exit_price = None
            forced     = False

            if side == 1:
                if l <= sl_price:
                    exit_price = sl_price; forced = True
                elif h >= tp_price:
                    exit_price = tp_price; forced = True
            else:
                if h >= sl_price:
                    exit_price = sl_price; forced = True
                elif l <= tp_price:
                    exit_price = tp_price; forced = True

            if not forced and (sig == -side or i == n - 1):
                exit_price = c * (1 - side * slippage)

            if exit_price is not None:
                if forced:
                    exit_price *= (1 - side * slippage)

                net_pnl    = side * (exit_price / entry_price - 1) - 2 * (commission + slippage)
                bar_return += net_pnl
                capital   *= (1 + net_pnl)

                trades.append({
                    "entry_time":  times[entry_idx],
                    "exit_time":   times[i],
                    "side":        "long" if side == 1 else "short",
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(exit_price, 2),
                    "sl_price":    round(sl_price, 2),
                    "tp_price":    round(tp_price, 2),
                    "exit_reason": "sl/tp" if forced else "signal",
                    "pnl_pct":     round(net_pnl * 100, 4),
                    "duration":    times[i] - times[entry_idx],
                })
                in_trade = False
                if forced:
                    block_entry = True

        if not in_trade and not block_entry and sig != 0:
            ep   = c * (1 + sig * slippage)
            sl   = sl_low[i] if sig == 1 else sl_high[i]
            dist = (ep - sl) if sig == 1 else (sl - ep)

            if dist <= 0 or dist / ep > 0.15 or np.isnan(sl):
                block_entry = False
                equity_arr[i]  = capital
                returns_arr[i] = bar_return
                continue

            tp = ep + sig * strategy.rr_ratio * dist

            in_trade    = True
            side        = sig
            entry_idx   = i
            entry_price = ep
            sl_price    = sl
            tp_price    = tp
            pos_arr[i]  = side
            cost        = commission + slippage
            bar_return -= cost
            capital    *= (1 - cost)

        block_entry    = False
        returns_arr[i] = bar_return
        equity_arr[i]  = capital

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "entry_price", "exit_price",
                 "sl_price", "tp_price", "exit_reason", "pnl_pct", "duration"]
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

    by_exit = trades.groupby("exit_reason")["pnl_pct"].agg(["count", "mean"])
    by_exit.columns = ["trades", "avg_pnl%"]
    print(f"\n  By exit reason:\n{by_exit.to_string()}")

    print(f"\n  Best trade : {trades['pnl_pct'].max():.2f}%")
    print(f"  Worst trade: {trades['pnl_pct'].min():.2f}%")
    print(f"  Avg hold   : {trades['duration'].mean()}")


if __name__ == "__main__":
    SYMBOL     = "BTC/USDT"
    SINCE      = "2020-01-01"
    CAPITAL    = 10_000.0
    COMMISSION = 0.001
    SLIPPAGE   = 0.0005

    strategy = SarEmaMacd(
        ema_period=200,
        sar_start=0.02,
        sar_increment=0.02,
        sar_maximum=0.2,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rebound_candles=3,
        swing_period=20,
        rr_ratio=2.0,
    )

    results = {}

    for tf in ["1h", "4h"]:
        print(f"\n{'═' * 55}")
        print(f"  Downloading {SYMBOL} {tf}  from {SINCE} …")
        df = get_ohlcv(SYMBOL, tf, since=SINCE)
        print(f"  Loaded {len(df):,} candles  ({df.index[0].date()} → {df.index[-1].date()})")

        result = run_dynamic_backtest(
            df, strategy,
            initial_capital=CAPITAL,
            commission=COMMISSION,
            slippage=SLIPPAGE,
            symbol=SYMBOL,
            timeframe=tf,
        )
        results[tf] = (result, df)

        print(f"\n{result.report()}")
        print_trade_breakdown(result)

        if not result.trades.empty:
            cols = ["entry_time", "exit_time", "side",
                    "entry_price", "exit_price", "exit_reason", "pnl_pct"]
            print(f"\n  First 10 trades:\n{result.trades[cols].head(10).to_string(index=False)}")

        ema200 = _ema(df["close"], 200)
        sar    = parabolic_sar(df)
        html   = f"sar_ema_macd_{tf}.html"
        plot_backtest(
            result, df,
            indicators={"EMA 200": ema200, "SAR": sar},
            title=f"{SYMBOL} {tf} — SAR+EMA200+MACD rebound | 2020→ | SL swing / TP 1:2",
            save_html=html,
        )
        print(f"\n  Chart → {html}")

    print(f"\n{'═' * 55}")
    print(f"  COMPARISON  1h  vs  4h")
    print(f"{'═' * 55}")
    keys = [
        "total_return_pct", "cagr_pct", "sharpe_ratio",
        "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades",
    ]
    header = f"  {'Metric':<25} {'1h':>12} {'4h':>12}"
    print(header)
    print(f"  {'-' * 49}")
    for k in keys:
        v1 = results["1h"][0].metrics.get(k, "—")
        v2 = results["4h"][0].metrics.get(k, "—")
        label = k.replace("_", " ").title()
        print(f"  {label:<25} {str(v1):>12} {str(v2):>12}")
    print(f"{'═' * 55}")
