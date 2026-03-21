"""
«Смарт-Риск» — SuperTrend (10, 3) + HMA (55) + CMF (20)
=========================================================
Параметры стратегии:
  SuperTrend : period=10, multiplier=3
  HMA        : period=55
  CMF        : period=20

Risk Management:
  Риск на сделку : 2% от текущего баланса
  Макс. плечо    : 5x
  Фильтр волат.  : пропустить вход если свеча сигнала > 3 ATR
  Stop-Loss      : по текущей линии SuperTrend (динамический трейлинг)
  Take-Profit    : 50% позиции при RR 1:2 от SL-дистанции на входе
  Trailing exit  : оставшиеся 50% — при смене цвета SuperTrend

Условия входа:
  Long  : ST перешёл в зелёный + цена > HMA + наклон HMA вверх + CMF > 0 + фильтр волат.
  Short : ST перешёл в красный + цена < HMA + наклон HMA вниз  + CMF < 0 + фильтр волат.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import get_ohlcv
from backtester.strategy.indicators import supertrend, hma, cmf, atr
from backtester.engine.metrics import compute_metrics
from backtester.engine.backtester import BacktestResult
from backtester.visualization.charts import plot_backtest

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL     = "BTC/USDT"
SINCE      = "2020-01-01"
CAPITAL    = 10_000.0
COMMISSION = 0.001    # 0.1% per side (taker)
SLIPPAGE   = 0.0005   # 0.05% per side
MAX_LEV    = 5.0      # max leverage cap

ST_PERIOD  = 10
ST_MULT    = 3.0
HMA_PERIOD = 55
CMF_PERIOD = 20
ATR_PERIOD = 10       # same period as SuperTrend
ATR_FILTER = 3.0      # max candle size (in ATR units) to allow entry


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, initial_capital: float) -> BacktestResult:
    """
    Bar-by-bar simulation with partial TP and dynamic SuperTrend trailing stop.

    Exit logic (priority order):
      1. Stop-Loss  : price touches SuperTrend line intrabar
      2. Take-Profit: 50% closed at RR 1:2 (2× initial SL distance from entry)
      3. Trail exit : remaining 50% closed on SuperTrend color flip
    """
    # ── Indicators ────────────────────────────────────────────────────────────
    st_dir_s, st_line_s = supertrend(df, period=ST_PERIOD, multiplier=ST_MULT)
    hma_s    = hma(df["close"], HMA_PERIOD)
    cmf_s    = cmf(df, CMF_PERIOD)
    atr_s    = atr(df, ATR_PERIOD)

    n      = len(df)
    hi     = df["high"].values
    lo     = df["low"].values
    cl     = df["close"].values
    st_d   = st_dir_s.values.astype(float)
    st_l   = st_line_s.values
    hma_v  = hma_s.values
    cmf_v  = cmf_s.values
    atr_v  = atr_s.values
    times  = df.index

    # ── State ─────────────────────────────────────────────────────────────────
    capital       = initial_capital
    in_trade      = False
    side          = 0         # 1=long, -1=short
    entry_px      = 0.0
    tp1_px        = 0.0       # 50% TP target price
    init_units    = 0.0       # units opened at entry
    units_rem     = 0.0       # units still open
    partial_done  = False     # True after first 50% was closed at TP1
    entry_idx     = 0

    equity_arr  = np.zeros(n)
    returns_arr = np.zeros(n)
    pos_arr     = np.zeros(n)
    prev_cap    = capital
    trades      = []

    # ── Helper ────────────────────────────────────────────────────────────────
    def _record(etype, exit_time, ex_px, units):
        """Append a trade record (gross P&L → net after commission)."""
        gross = side * (ex_px / entry_px - 1)
        net   = gross - COMMISSION          # exit-side cost only (entry already deducted)
        trades.append({
            "entry_time":  times[entry_idx],
            "exit_time":   exit_time,
            "side":        "long" if side == 1 else "short",
            "entry_price": round(entry_px, 2),
            "exit_price":  round(ex_px, 2),
            "exit_type":   etype,
            "units":       round(units, 8),
            "pnl_pct":     round(net * 100, 4),
            "pnl_usd":     round(net * units * entry_px, 2),
        })

    def _close(etype, ex_px_raw, units):
        """Apply slippage + commission, update capital, write trade record."""
        nonlocal capital
        ex_px = ex_px_raw * (1.0 - side * SLIPPAGE)
        gross_usd = units * side * (ex_px - entry_px)
        cost_usd  = COMMISSION * units * ex_px
        capital  += gross_usd - cost_usd
        _record(etype, times[i], ex_px, units)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i in range(n):
        h_i   = hi[i];  l_i  = lo[i];  c_i  = cl[i]
        st_di = st_d[i]; st_li = st_l[i]
        hma_i = hma_v[i]; cmf_i = cmf_v[i]; atr_i = atr_v[i]

        # Skip until all indicators are ready
        if any(np.isnan(v) for v in [st_li, hma_i, cmf_i, atr_i]):
            equity_arr[i]  = capital
            returns_arr[i] = 0.0
            prev_cap = capital
            continue

        # ── Exit logic ────────────────────────────────────────────────────────
        if in_trade:
            pos_arr[i] = side

            # ST flip: close пересёк активную линию → смена цвета индикатора.
            # Требует st_di ≠ st_d[i-1], значит st_di != side_dir.
            # wick_sl и st_flip взаимно исключают друг друга (разные st_di).
            st_flip = False
            if i > 0 and not np.isnan(st_d[i - 1]):
                st_flip = (side == 1 and st_d[i-1] == 1 and st_di == -1) or \
                          (side == -1 and st_d[i-1] == -1 and st_di == 1)

            # Intrabar wick SL: ST ещё НЕ перевернулся (st_di совпадает с side).
            # В этом случае st_li — ПРАВИЛЬНАЯ полоса (нижняя для лонга, верхняя для шорта).
            wick_sl = (side == 1  and st_di == 1  and l_i <= st_li) or \
                      (side == -1 and st_di == -1 and h_i >= st_li)

            # Partial TP: 50% at RR 1:2 (только если стоп не сработал)
            tp_hit = (not partial_done) and (not st_flip) and (not wick_sl) and (
                (side == 1 and h_i >= tp1_px) or
                (side == -1 and l_i <= tp1_px)
            )

            if st_flip:
                # Выход по смене цвета: цена закрытия (close-based stop)
                etype = "st_flip" if partial_done else "sl"
                _close(etype, c_i, units_rem)
                in_trade = False; partial_done = False; units_rem = 0.0

            elif wick_sl:
                # Intrabar wick коснулся ST-линии (правильная полоса)
                _close("sl", st_li, units_rem)
                in_trade = False; partial_done = False; units_rem = 0.0

            elif tp_hit:
                # Close 50% at TP1
                half = init_units * 0.5
                _close("tp1", tp1_px, half)
                units_rem    = init_units * 0.5
                partial_done = True

        # ── Entry logic (only when flat) ──────────────────────────────────────
        if not in_trade and i > 0:
            prev_st = st_d[i - 1]
            if np.isnan(prev_st):
                equity_arr[i] = capital
                returns_arr[i] = (capital - prev_cap) / prev_cap if prev_cap else 0.0
                prev_cap = capital
                continue

            st_up = (prev_st == -1) and (st_di == 1)   # ST turned green
            st_dn = (prev_st ==  1) and (st_di == -1)  # ST turned red

            hma_up = hma_v[i] > hma_v[i-1] if not np.isnan(hma_v[i-1]) else False
            hma_dn = hma_v[i] < hma_v[i-1] if not np.isnan(hma_v[i-1]) else False

            vol_ok = (h_i - l_i) <= ATR_FILTER * atr_i  # volatility filter

            if st_up and c_i > hma_i and cmf_i > 0 and hma_up and vol_ok:
                ep     = c_i * (1.0 + SLIPPAGE)
                sl_d   = ep - st_li
                if sl_d > 0:
                    iu = min(capital * 0.02 / sl_d, capital * MAX_LEV / ep)
                    capital -= COMMISSION * iu * ep     # entry commission
                    in_trade  = True;  side  = 1
                    entry_px  = ep;    tp1_px = ep + 2.0 * sl_d
                    init_units = iu;   units_rem = iu
                    partial_done = False;  entry_idx = i
                    pos_arr[i] = 1

            elif st_dn and c_i < hma_i and cmf_i < 0 and hma_dn and vol_ok:
                ep   = c_i * (1.0 - SLIPPAGE)
                sl_d = st_li - ep
                if sl_d > 0:
                    iu = min(capital * 0.02 / sl_d, capital * MAX_LEV / ep)
                    capital -= COMMISSION * iu * ep
                    in_trade  = True;  side  = -1
                    entry_px  = ep;    tp1_px = ep - 2.0 * sl_d
                    init_units = iu;   units_rem = iu
                    partial_done = False;  entry_idx = i
                    pos_arr[i] = -1

        # ── Bar close ─────────────────────────────────────────────────────────
        equity_arr[i]  = capital
        returns_arr[i] = (capital - prev_cap) / prev_cap if prev_cap else 0.0
        prev_cap = capital

    # ── Build result ──────────────────────────────────────────────────────────
    equity_s  = pd.Series(equity_arr,  index=df.index)
    returns_s = pd.Series(returns_arr, index=df.index)
    pos_s     = pd.Series(pos_arr,     index=df.index)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
        "entry_time", "exit_time", "side", "entry_price",
        "exit_price", "exit_type", "units", "pnl_pct", "pnl_usd",
    ])

    metrics = compute_metrics(returns_s, equity_s, trades_df, initial_capital)

    # Extra breakdown by exit type
    if not trades_df.empty:
        for etype in ("sl", "tp1", "st_flip"):
            metrics[f"exits_{etype}"] = int((trades_df["exit_type"] == etype).sum())
        metrics["total_pnl_usd"] = round(trades_df["pnl_usd"].sum(), 2)
        metrics["avg_pnl_usd"]   = round(trades_df["pnl_usd"].mean(), 2)

    return BacktestResult(
        equity=equity_s,
        returns=returns_s,
        positions=pos_s,
        trades=trades_df,
        metrics=metrics,
        params={
            "st_period": ST_PERIOD, "st_mult": ST_MULT,
            "hma_period": HMA_PERIOD, "cmf_period": CMF_PERIOD,
            "risk_pct": 2.0, "max_leverage": MAX_LEV,
            "atr_filter": ATR_FILTER,
        },
        symbol=SYMBOL,
    )


# ── Runner ────────────────────────────────────────────────────────────────────

for tf in ["1h", "4h"]:
    print(f"\n{'#'*60}")
    print(f"  «Смарт-Риск»  {SYMBOL}  {tf}  |  {SINCE} → сегодня")
    print(f"  Баланс: ${CAPITAL:,.0f}  |  Риск: 2%  |  Макс. плечо: {MAX_LEV}x")
    print(f"{'#'*60}")

    df = get_ohlcv(SYMBOL, tf, since=SINCE)
    print(f"  Загружено {len(df):,} свечей  ({df.index[0]} → {df.index[-1]})\n")

    result = simulate(df, CAPITAL)

    # Extended report
    print(result.report())

    if not result.trades.empty:
        td = result.trades
        print(f"\n  Разбивка выходов:")
        print(f"    Stop-Loss (SL)        : {result.metrics.get('exits_sl', 0)}")
        print(f"    Take-Profit 50% (TP1) : {result.metrics.get('exits_tp1', 0)}")
        print(f"    SuperTrend flip       : {result.metrics.get('exits_st_flip', 0)}")
        print(f"    Avg P&L per exit, $   : {result.metrics.get('avg_pnl_usd', 0):.2f}")

        long_t  = td[td["side"] == "long"]
        short_t = td[td["side"] == "short"]
        print(f"\n  Long  trades: {len(long_t)}  | "
              f"WR: {(long_t['pnl_pct'] > 0).mean()*100:.1f}%  | "
              f"Net $: {long_t['pnl_usd'].sum():.0f}")
        print(f"  Short trades: {len(short_t)}  | "
              f"WR: {(short_t['pnl_pct'] > 0).mean()*100:.1f}%  | "
              f"Net $: {short_t['pnl_usd'].sum():.0f}")

        print(f"\n  Последние 10 закрытий:")
        cols = ["entry_time", "exit_time", "side", "entry_price",
                "exit_price", "exit_type", "pnl_usd", "pnl_pct"]
        print(td[cols].tail(10).to_string(index=False))

    # Chart
    st_dir_s, st_line_s = supertrend(df, period=ST_PERIOD, multiplier=ST_MULT)
    hma_line = hma(df["close"], HMA_PERIOD)
    out_file = f"smart_risk_{tf.replace('h', 'H')}.html"
    plot_backtest(
        result, df,
        indicators={"HMA 55": hma_line, "SuperTrend": st_line_s},
        title=f"«Смарт-Риск» {SYMBOL} {tf} — ST(10,3) + HMA55 + CMF20",
        save_html=out_file,
    )
    print(f"\n  График сохранён → {out_file}")
