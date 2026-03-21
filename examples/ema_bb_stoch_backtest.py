"""
Backtest v3: «Среднесрочный Трендолов» — EMA 50/200 + BB(20,2) + Stochastic(14,3,3)
──────────────────────────────────────────────────────────────────────────────────────
Capital       : $10 000
Period        : 2020-01-01 → now
Timeframes    : 1h, 4h, 1d  (long+short и long_only)
Risk per trade: 1.5% от текущего баланса
SL            : swing low/high (10 баров), минимум 1.0 × ATR
TP            : 1 : 2.0 risk/reward
Breakeven     : SL → entry при 1:1 RR
Cooldown      : пропуск входа N баров после выхода из позиции
Fees          : 0.1% commission + 0.05% slippage per side

v3 изменения:
  1. Только точное касание BB телом свечи (close <= lower_bb)
  2. Stoch кроссовер: текущий или предыдущий бар
  3. Stoch растёт/падает 2 бара подряд
  4. RSI фильтр (< 60 для лонгов)
  5. Тренд устойчив 5 свечей
  6. Cooldown между сделками
  7. Блокировка входа только после жёсткого SL
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import get_ohlcv
from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.strategy.ema_bb_stoch import EmaBbStoch
from backtester.strategy.indicators import ema as _ema
from backtester.visualization.charts import plot_backtest


# ══════════════════════════════════════════════════════════════════════════════
# Bar-by-bar симуляция
# ══════════════════════════════════════════════════════════════════════════════

def run_dynamic_backtest(
    df: pd.DataFrame,
    strategy: EmaBbStoch,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    cooldown_bars: int = 0,
    symbol: str = "",
    timeframe: str = "",
) -> BacktestResult:
    """
    cooldown_bars : не открывать новую сделку N баров после закрытия предыдущей.
    """
    raw_signals = strategy.generate_signals(df).reindex(df.index).fillna(0)
    sw_low, sw_high = strategy.swing_levels(df)

    signals   = raw_signals.shift(1).fillna(0).values.astype(int)
    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    sl_low    = sw_low.values
    sl_high   = sw_high.values
    times     = df.index
    n         = len(times)

    trades        = []
    capital       = initial_capital
    equity_arr    = np.empty(n)
    returns_arr   = np.zeros(n)
    pos_arr       = np.zeros(n)

    in_trade      = False
    side          = 0
    entry_idx     = 0
    entry_price   = 0.0
    sl_price      = 0.0
    tp_price      = 0.0
    be_level      = 0.0
    be_triggered  = False
    position_size = 1.0
    block_entry   = False       # блок после жёсткого SL
    cooldown_left = 0           # осталось баров до разрешения входа

    for i in range(n):
        sig        = signals[i]
        c, h, l    = close_arr[i], high_arr[i], low_arr[i]
        bar_return = 0.0

        if cooldown_left > 0:
            cooldown_left -= 1

        if in_trade:
            pos_arr[i] = side

            # ── Безубыток ─────────────────────────────────────────────────────
            if not be_triggered:
                if side == 1 and h >= be_level:
                    sl_price    = entry_price
                    be_triggered = True
                elif side == -1 and l <= be_level:
                    sl_price    = entry_price
                    be_triggered = True

            # ── SL / TP ───────────────────────────────────────────────────────
            exit_price  = None
            forced      = False
            is_hard_sl  = False
            exit_reason = ""

            if side == 1:
                if l <= sl_price:
                    exit_price  = sl_price
                    forced      = True
                    is_hard_sl  = not be_triggered
                    exit_reason = "breakeven" if be_triggered else "sl"
                elif h >= tp_price:
                    exit_price  = tp_price
                    forced      = True
                    exit_reason = "tp"
            else:
                if h >= sl_price:
                    exit_price  = sl_price
                    forced      = True
                    is_hard_sl  = not be_triggered
                    exit_reason = "breakeven" if be_triggered else "sl"
                elif l <= tp_price:
                    exit_price  = tp_price
                    forced      = True
                    exit_reason = "tp"

            if not forced and (sig == -side or i == n - 1):
                exit_price  = c * (1 - side * slippage)
                exit_reason = "signal"

            if exit_price is not None:
                if forced:
                    exit_price *= (1 - side * slippage)

                raw_ret    = side * (exit_price / entry_price - 1)
                net_pnl    = raw_ret * position_size - 2 * (commission + slippage)
                bar_return += net_pnl
                capital   *= (1 + net_pnl)

                trades.append({
                    "entry_time":    times[entry_idx],
                    "exit_time":     times[i],
                    "side":          "long" if side == 1 else "short",
                    "entry_price":   round(entry_price, 4),
                    "exit_price":    round(exit_price, 4),
                    "sl_price":      round(sl_price, 4),
                    "tp_price":      round(tp_price, 4),
                    "exit_reason":   exit_reason,
                    "breakeven_hit": be_triggered,
                    "pnl_pct":       round(net_pnl * 100, 4),
                    "duration":      times[i] - times[entry_idx],
                })
                in_trade = False
                if forced and is_hard_sl:
                    block_entry = True
                if cooldown_bars > 0:
                    cooldown_left = cooldown_bars

        # ── Новая сделка ──────────────────────────────────────────────────────
        if not in_trade and not block_entry and cooldown_left == 0 and sig != 0:
            ep = c * (1 + sig * slippage)

            sl   = sl_low[i]  if sig == 1 else sl_high[i]
            dist = (ep - sl)  if sig == 1 else (sl - ep)

            if dist <= 0 or dist / ep > 0.20 or np.isnan(sl):
                block_entry = False
                equity_arr[i]  = capital
                returns_arr[i] = bar_return
                continue

            tp = ep + sig * strategy.rr_ratio * dist
            sl_pct        = dist / ep
            position_size = min(strategy.risk_pct / sl_pct, 5.0)
            be_level      = ep + sig * dist

            in_trade     = True
            be_triggered = False
            side         = sig
            entry_idx    = i
            entry_price  = ep
            sl_price     = sl
            tp_price     = tp
            pos_arr[i]   = side
            cost         = commission + slippage
            bar_return  -= cost
            capital     *= (1 - cost)

        block_entry    = False
        returns_arr[i] = bar_return
        equity_arr[i]  = capital

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "entry_price", "exit_price",
                 "sl_price", "tp_price", "exit_reason", "breakeven_hit",
                 "pnl_pct", "duration"]
    )
    equity    = pd.Series(equity_arr,  index=df.index)
    returns   = pd.Series(returns_arr, index=df.index)
    positions = pd.Series(pos_arr,     index=df.index)
    metrics   = compute_metrics(returns, equity, trades_df, initial_capital)

    return BacktestResult(
        equity=equity, returns=returns, positions=positions,
        trades=trades_df, metrics=metrics,
        params=strategy.get_params(), symbol=symbol, timeframe=timeframe,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Отчёты
# ══════════════════════════════════════════════════════════════════════════════

def print_trade_breakdown(result: BacktestResult) -> None:
    t = result.trades
    if t.empty:
        print("  Нет сделок.")
        return
    by_side = t.groupby("side")["pnl_pct"].agg(
        trades="count",
        avg_pnl=lambda x: round(x.mean(), 3),
        win_rate=lambda x: round((x > 0).mean() * 100, 1),
    )
    print(f"\n  По направлению:\n{by_side.to_string()}")
    by_exit = t.groupby("exit_reason")["pnl_pct"].agg(
        trades="count",
        avg_pnl=lambda x: round(x.mean(), 3),
    )
    print(f"\n  По причине выхода:\n{by_exit.to_string()}")
    be = t[t["breakeven_hit"]]
    print(f"\n  Безубыток: {len(be)} активаций  (прибыльных: {(be['pnl_pct'] > 0).sum()})")
    print(f"  Лучшая:  {t['pnl_pct'].max():.2f}%   Худшая: {t['pnl_pct'].min():.2f}%")
    print(f"  Ср. длительность: {t['duration'].mean()}")


def print_comparison(results: dict) -> None:
    keys = [
        "total_return_pct", "cagr_pct", "sharpe_ratio", "sortino_ratio",
        "calmar_ratio", "max_drawdown_pct", "win_rate_pct",
        "profit_factor", "total_trades", "final_capital",
    ]
    tfs = list(results.keys())
    w = 17
    print(f"\n{'═' * 72}")
    print(f"  СРАВНЕНИЕ: {' | '.join(tfs)}")
    print(f"{'═' * 72}")
    print(f"  {'Метрика':<30}" + "".join(f"{tf:>{w}}" for tf in tfs))
    print(f"  {'-' * 68}")
    for k in keys:
        row = f"  {k.replace('_',' ').title():<30}"
        for tf in tfs:
            v = results[tf].metrics.get(k, "—")
            row += f"{str(v):>{w}}"
        print(row)
    print(f"{'═' * 72}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SYMBOL     = "BTC/USDT"
    SINCE      = "2020-01-01"
    CAPITAL    = 10_000.0
    COMMISSION = 0.001
    SLIPPAGE   = 0.0005

    # Cooldown баров по таймфрейму (≈ 12 часов «отдыха» после сделки)
    COOLDOWN = {"1h": 12, "4h": 3, "1d": 2}

    # Запускаем ДВА варианта: long+short и long_only
    CONFIGS = {
        "long+short": dict(long_only=False),
        "long_only":  dict(long_only=True),
    }

    all_results = {}

    for tf in ["1h", "4h", "1d"]:
        print(f"\n{'═' * 70}")
        print(f"  Загрузка {SYMBOL} [{tf}]  с {SINCE}")
        df = get_ohlcv(SYMBOL, tf, since=SINCE)
        print(f"  {len(df):,} свечей  ({df.index[0].date()} → {df.index[-1].date()})")

        for cfg_name, cfg_params in CONFIGS.items():
            tag = f"{tf} {cfg_name}"
            strategy = EmaBbStoch(
                ema_fast=50, ema_slow=200,
                bb_period=20, bb_std=2.0,
                stoch_k=14, stoch_smooth=3, stoch_d=3,
                stoch_oversold=20, stoch_overbought=80,
                trend_bars=5,
                rsi_period=14, rsi_max_long=60, rsi_min_short=40,
                swing_period=10,
                atr_period=14, atr_sl_mult=1.0,
                rr_ratio=2.0,
                risk_pct=0.015,
                **cfg_params,
            )

            result = run_dynamic_backtest(
                df, strategy,
                initial_capital=CAPITAL,
                commission=COMMISSION,
                slippage=SLIPPAGE,
                cooldown_bars=COOLDOWN[tf],
                symbol=SYMBOL,
                timeframe=tf,
            )
            all_results[tag] = result

            print(f"\n{'─' * 50}")
            print(f"  [{tag}]")
            print(f"{result.report()}")
            print_trade_breakdown(result)

            if not result.trades.empty:
                cols = ["entry_time", "exit_time", "side",
                        "entry_price", "exit_price",
                        "exit_reason", "breakeven_hit", "pnl_pct"]
                print(f"\n  Первые 15 сделок:\n"
                      f"{result.trades[cols].head(15).to_string(index=False)}")

            # График только для лучших конфигураций (не засорять папку)
            if tf in ("4h", "1d"):
                ema50  = _ema(df["close"], 50)
                ema200 = _ema(df["close"], 200)
                safe   = cfg_name.replace("+", "_")
                html   = f"ema_bb_stoch_v3_{tf}_{safe}.html"
                plot_backtest(
                    result, df,
                    indicators={"EMA 50": ema50, "EMA 200": ema200},
                    title=f"{SYMBOL} {tf} v3 [{cfg_name}] | Risk 1.5% | RR 1:2 | BE 1:1",
                    save_html=html,
                )
                print(f"\n  График → {html}")

    # ── Итоговая таблица ───────────────────────────────────────────────────────
    print_comparison(all_results)
