"""
Backtest: «S/R Bounce — Grid DCA»

Capital       : $10 000
Timeframe     : 1d (BTC/USDT)
Period        : 2020-01-01 → now

Money Management — сетка DCA (только лонги)
-------------------------------------------
Сигнал: цена касается уровня поддержки (2–4 недельной давности) и закрывается выше.

При сигнале открывается сетка из 7 равных ордеров:
  Ордер 1: ep           — по рыночной при сигнале
  Ордер 2: ep × 0.97   — лимит −3%
  Ордер 3: ep × 0.96   — лимит −4%
  Ордер 4: ep × 0.95   — лимит −5%
  Ордер 5: ep × 0.94   — лимит −6%
  Ордер 6: ep × 0.93   — лимит −7%
  Ордер 7: ep × 0.92   — лимит −8%

TP  : avg_entry × 1.01 (пересчитывается после каждого DCA-ордера)
SL  : последний уровень DCA × 0.98 (−2% буфер ниже последнего уровня)

Комиссия 0.1% per side, slippage 0.05%.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.data import get_ohlcv
from backtester.engine.backtester import BacktestResult
from backtester.engine.metrics import compute_metrics
from backtester.strategy.sr_bounce import SRBounce


# ── Grid simulation ───────────────────────────────────────────────────────────

def run_grid_backtest(
    df: pd.DataFrame,
    strategy: SRBounce,
    initial_capital: float = 10_000.0,
    n_orders:        int   = 7,
    order_size_pct:  float = 0.10,    # доля капитала на каждый ордер
    first_dca_drop:  float = 0.03,    # −3% для 2-го ордера (лонг) / +3% (шорт)
    step_dca_drop:   float = 0.01,    # −1% для каждого следующего
    tp_pct:          float = 0.01,    # TP = avg_entry × 1.01 (лонг) / × 0.99 (шорт)
    commission:      float = 0.001,
    slippage:        float = 0.0005,
    symbol:          str   = "",
    timeframe:       str   = "",
) -> BacktestResult:
    """Bar-by-bar симуляция сетки DCA от уровней поддержки (только лонги)."""

    raw_sig  = strategy.generate_signals(df).reindex(df.index).fillna(0)
    # Сигнал на баре i → вход на баре i+1 (no lookahead)
    signals  = raw_sig.shift(1).fillna(0).values.astype(int)
    # лонг (от поддержки) и шорт (от сопротивления)

    open_arr  = df["open"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    close_arr = df["close"].values
    times     = df.index
    n         = len(times)

    trades      = []
    capital     = initial_capital
    equity_arr  = np.empty(n)
    returns_arr = np.zeros(n)

    # Grid state
    in_grid        = False
    grid_side      = 0        # 1=long, -1=short
    total_coins    = 0.0      # монеты (лонг) или проданные монеты (шорт)
    total_cost     = 0.0      # суммарный USD вложен
    avg_entry      = 0.0
    tp_price       = 0.0
    filled_count   = 0
    dca_levels     = []       # список (цена_уровня | None если исполнен)
    grid_idx       = 0
    order_size_usd = 0.0
    dca_fills      = []

    for i in range(n):
        ts   = times[i]
        sig  = signals[i]
        h    = high_arr[i]
        l    = low_arr[i]
        c    = close_arr[i]
        o    = open_arr[i]
        bar_ret = 0.0

        if in_grid:
            # 1. Проверяем DCA лимиты
            for j, lvl in enumerate(dca_levels):
                if lvl is None or filled_count >= n_orders:
                    continue
                triggered = (grid_side == 1 and l <= lvl) or \
                            (grid_side == -1 and h >= lvl)
                if triggered:
                    fill_px  = lvl
                    cost_usd = order_size_usd
                    comm     = cost_usd * commission
                    if capital < cost_usd + comm:
                        continue
                    if grid_side == 1:
                        coins = cost_usd / fill_px
                    else:
                        coins = cost_usd / fill_px   # те же монеты для шорта
                    capital     -= cost_usd + comm
                    total_coins += coins
                    total_cost  += cost_usd
                    avg_entry    = total_cost / total_coins
                    tp_price     = avg_entry * (1.0 - tp_pct) if grid_side == -1 \
                                   else avg_entry * (1.0 + tp_pct)
                    filled_count += 1
                    dca_levels[j] = None
                    dca_fills.append((i, fill_px, order_size_usd))

            # 2. Проверяем только TP (без SL)
            exit_price = None
            if grid_side == 1 and h >= tp_price:
                exit_price = tp_price
            elif grid_side == -1 and l <= tp_price:
                exit_price = tp_price

            if exit_price is not None:
                if grid_side == 1:
                    proceeds = total_coins * exit_price * (1.0 - commission)
                    pnl_usd  = proceeds - total_cost
                else:
                    # шорт: мы продавали монеты, теперь выкупаем
                    buyback  = total_coins * exit_price * (1.0 + commission)
                    pnl_usd  = total_cost - buyback
                    proceeds = total_cost + pnl_usd   # возврат в капитал

                pnl_pct  = pnl_usd / total_cost * 100.0
                bar_ret  = pnl_usd / (capital + total_cost)
                capital += proceeds if grid_side == 1 else (total_cost + pnl_usd)

                trades.append({
                    "entry_time":    times[grid_idx],
                    "exit_time":     ts,
                    "side":          "long" if grid_side == 1 else "short",
                    "entry_price":   round(avg_entry, 2),
                    "exit_price":    round(exit_price, 2),
                    "avg_entry":     round(avg_entry, 2),
                    "tp_price":      round(tp_price, 2),
                    "sl_price":      0.0,
                    "exit_reason":   "tp",
                    "orders_filled": filled_count,
                    "pnl_usd":       round(pnl_usd, 2),
                    "pnl_pct":       round(pnl_pct, 4),
                    "duration":      ts - times[grid_idx],
                })

                in_grid      = False
                total_coins  = 0.0
                total_cost   = 0.0
                avg_entry    = 0.0
                filled_count = 0
                dca_levels   = []
                grid_side    = 0

        # 3. Новый вход (если не в позиции и есть сигнал)
        if not in_grid and sig != 0:
            ep             = o * (1.0 + sig * slippage)
            order_size_usd = min(capital * order_size_pct, capital)
            if order_size_usd <= 0:
                equity_arr[i]  = capital
                returns_arr[i] = bar_ret
                continue

            coins  = order_size_usd / ep
            comm   = order_size_usd * commission
            capital -= order_size_usd + comm

            total_coins  = coins
            total_cost   = order_size_usd
            avg_entry    = ep
            grid_side    = sig
            tp_price     = ep * (1.0 + tp_pct) if sig == 1 else ep * (1.0 - tp_pct)
            filled_count = 1
            grid_idx     = i

            # DCA уровни: для лонга — вниз, для шорта — вверх
            lvl = ep * (1.0 - sig * first_dca_drop)
            dca_levels = [lvl]
            for _ in range(n_orders - 2):
                lvl = lvl * (1.0 - sig * step_dca_drop)
                dca_levels.append(lvl)

            in_grid = True

        equity_arr[i]  = capital + total_coins * c
        returns_arr[i] = bar_ret

    # Закрываем незакрытую позицию по последнему close
    if in_grid and total_coins > 0:
        exit_price = close_arr[-1]
        if grid_side == 1:
            proceeds = total_coins * exit_price * (1.0 - commission)
            pnl_usd  = proceeds - total_cost
            capital += proceeds
        else:
            buyback  = total_coins * exit_price * (1.0 + commission)
            pnl_usd  = total_cost - buyback
            capital += total_cost + pnl_usd
        pnl_pct    = pnl_usd / total_cost * 100.0
        trades.append({
            "entry_time":    times[grid_idx],
            "exit_time":     times[-1],
            "side":          "long",
            "entry_price":   round(avg_entry, 2),
            "exit_price":    round(exit_price, 2),
            "avg_entry":     round(avg_entry, 2),
            "tp_price":      round(tp_price, 2),
            "sl_price":      0.0,
            "exit_reason":   "end",
            "orders_filled": filled_count,
            "pnl_usd":       round(pnl_usd, 2),
            "pnl_pct":       round(pnl_pct, 4),
            "duration":      times[-1] - times[grid_idx],
        })
        equity_arr[-1] = capital

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "entry_price", "exit_price",
                 "avg_entry", "tp_price", "sl_price", "exit_reason",
                 "orders_filled", "pnl_usd", "pnl_pct", "duration"]
    )
    # Для совместимости с compute_metrics добавляем pnl_pct как долю (не проценты)
    pnl_col = trades_df["pnl_pct"] / 100.0 if not trades_df.empty else pd.Series(dtype=float)

    equity    = pd.Series(equity_arr,  index=df.index)
    returns   = pd.Series(returns_arr, index=df.index)
    positions = pd.Series(np.zeros(n), index=df.index)
    metrics   = compute_metrics(returns, equity, trades_df, initial_capital)

    result = BacktestResult(
        equity=equity, returns=returns, positions=positions,
        trades=trades_df, metrics=metrics,
        params=strategy.get_params(), symbol=symbol, timeframe=timeframe,
    )
    result._dca_fills = dca_fills   # сохраняем для графика
    return result


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_grid_backtest(
    result,
    df: pd.DataFrame,
    strategy: SRBounce,
    title: str = "",
    save_html: str = "",
) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.30, 0.15],
        vertical_spacing=0.03,
        subplot_titles=("Price & S/R Levels", "Equity Curve", "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    sup = strategy.support_line
    res = strategy.resistance_line

    # S/R зона
    fig.add_trace(go.Scatter(
        x=pd.concat([pd.Series(sup.index), pd.Series(res.index[::-1])]),
        y=pd.concat([sup, res.iloc[::-1]]),
        fill="toself", fillcolor="rgba(150,150,150,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sup.index, y=sup.values, mode="lines",
        name="Support (2–4w)", line=dict(color="#26a69a", width=1.5, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=res.index, y=res.values, mode="lines",
        name="Resistance (2–4w)", line=dict(color="#ef5350", width=1.5, dash="dot"),
    ), row=1, col=1)

    trades = result.trades
    if not trades.empty:
        def _bar_price(times_list, field, pct):
            return [df[field].iloc[df.index.get_indexer([t], method="nearest")[0]] * (1 + pct)
                    for t in times_list]

        # Входы (первый ордер)
        fig.add_trace(go.Scatter(
            x=trades["entry_time"],
            y=_bar_price(trades["entry_time"], "low", -0.015),
            mode="markers", name="Grid entry",
            marker=dict(symbol="triangle-up", size=12, color="#26a69a",
                        line=dict(width=1, color="white")),
            customdata=trades[["entry_price", "exit_price", "pnl_pct",
                                "exit_reason", "orders_filled", "avg_entry"]].values,
            hovertemplate=(
                "<b>Grid Long Entry</b><br>"
                "1st entry: %{customdata[0]:.2f}<br>"
                "Exit:      %{customdata[1]:.2f}<br>"
                "PnL:       %{customdata[2]:.3f}%<br>"
                "Reason:    %{customdata[3]}<br>"
                "Orders:    %{customdata[4]}/7<br>"
                "Avg entry: %{customdata[5]:.2f}<extra></extra>"
            ),
        ), row=1, col=1)

        # Выходы
        exit_colors = ["#ef5350" if r == "sl" else "#26a69a"
                       for r in trades["exit_reason"]]
        fig.add_trace(go.Scatter(
            x=trades["exit_time"],
            y=_bar_price(trades["exit_time"], "high", 0.015),
            mode="markers", name="Exit", showlegend=False,
            marker=dict(symbol="triangle-down", size=10,
                        color=exit_colors, line=dict(width=1, color="white")),
        ), row=1, col=1)

    # Equity
    initial = result.equity.iloc[0]
    fig.add_trace(go.Scatter(
        x=result.equity.index, y=result.equity.values,
        mode="lines", name="Strategy",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
    ), row=2, col=1)

    bnh = (df["close"] / df["close"].iloc[0]) * initial
    fig.add_trace(go.Scatter(
        x=bnh.index, y=bnh.values, mode="lines", name="Buy & Hold",
        line=dict(color="#9E9E9E", width=1.5, dash="dot"),
    ), row=2, col=1)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, showlegend=False,
    ), row=3, col=1)

    # Stats
    m       = result.metrics
    final   = result.equity.iloc[-1]
    ret_pct = m.get("total_return_pct", 0)
    rc      = "#26a69a" if ret_pct >= 0 else "#ef5350"

    avg_orders = (result.trades["orders_filled"].mean()
                  if not result.trades.empty and "orders_filled" in result.trades else 0)
    tp_cnt = (result.trades["exit_reason"] == "tp").sum() if not result.trades.empty else 0
    sl_cnt = (result.trades["exit_reason"] == "sl").sum() if not result.trades.empty else 0

    stats = "<br>".join([
        f"<b>Capital: ${initial:,.0f} → ${final:,.0f}</b>",
        f"Return:     <span style='color:{rc}'>{ret_pct:+.2f}%</span>",
        f"Win Rate:   {m.get('win_rate_pct', 0):.1f}%",
        f"Sharpe:     {m.get('sharpe_ratio', 0):.2f}",
        f"Max DD:     {m.get('max_drawdown_pct', 0):.2f}%",
        f"Trades:     {int(m.get('total_trades', 0))}  (TP:{tp_cnt} SL:{sl_cnt})",
        f"Avg orders: {avg_orders:.1f}/7",
        f"Pft Factor: {m.get('profit_factor', 0):.2f}",
    ])
    fig.add_annotation(
        text=stats, xref="paper", yref="paper",
        x=0.01, y=0.01, xanchor="left", yanchor="bottom",
        showarrow=False, align="left",
        font=dict(size=12, color="#e0e0e0", family="monospace"),
        bgcolor="rgba(30,30,40,0.85)",
        bordercolor="#555", borderwidth=1, borderpad=8,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_dark", height=920,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(row=1, col=1, title_text="Price")
    fig.update_yaxes(row=2, col=1, title_text="Capital")
    fig.update_yaxes(row=3, col=1, title_text="Volume")

    if save_html:
        fig.write_html(save_html)
        print(f"  Chart saved → {save_html}")
    fig.show()
    return fig


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_breakdown(result) -> None:
    trades = result.trades
    if trades.empty:
        print("  No trades.")
        return
    print(f"\n  Exit reason breakdown:")
    if "exit_reason" in trades.columns:
        gr = trades.groupby("exit_reason").agg(
            count=("pnl_pct", "count"),
            avg_pnl=("pnl_pct", "mean"),
        )
        print(f"{gr.to_string()}")
    print(f"\n  Avg orders filled : {trades['orders_filled'].mean():.2f} / 7")
    print(f"  Best trade PnL    : {trades['pnl_pct'].max():.3f}%")
    print(f"  Worst trade PnL   : {trades['pnl_pct'].min():.3f}%")
    print(f"  Avg PnL           : {trades['pnl_pct'].mean():.3f}%")
    print(f"  Avg hold          : {trades['duration'].mean()}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SYMBOL    = "BTC/USDT"
    SINCE     = "2020-01-01"
    TIMEFRAME = "1d"
    CAPITAL   = 10_000.0

    strategy = SRBounce(
        weeks_near    = 2,
        weeks_far     = 4,
        zone_pct      = 0.015,   # 1.5% зона касания поддержки
        ema_period    = 50,
        rsi_period    = 14,
        rsi_long_max  = 70.0,
        rsi_short_min = 30.0,
        vol_sma_period= 20,
        atr_period    = 14,
        atr_sl_mult   = 1.5,
        min_rr        = 0.5,     # низкий порог — геометрия определяется сеткой, не RR
        cooldown_bars = 0,
    )

    print(f"\n{'═' * 60}")
    print(f"  Загружаю {SYMBOL} {TIMEFRAME} с {SINCE} ...")
    df = get_ohlcv(SYMBOL, TIMEFRAME, since=SINCE)
    print(f"  Загружено {len(df):,} свечей  ({df.index[0].date()} → {df.index[-1].date()})")

    result = run_grid_backtest(
        df,
        strategy,
        initial_capital = CAPITAL,
        n_orders        = 7,
        order_size_pct  = 0.10,    # 10% капитала на каждый из 7 ордеров
        first_dca_drop  = 0.05,    # 2-й ордер на −5%
        step_dca_drop   = 0.015,   # 3–7-й ордера каждый −1.5%
        tp_pct          = 0.03,    # TP = avg_entry × 1.03 (лонг) / × 0.97 (шорт)
        commission      = 0.001,
        slippage        = 0.0005,
        symbol          = SYMBOL,
        timeframe       = TIMEFRAME,
    )

    print(f"\n{result.report()}")
    print_breakdown(result)

    if not result.trades.empty:
        cols = ["entry_time", "exit_time", "entry_price", "avg_entry",
                "exit_price", "exit_reason", "orders_filled", "pnl_pct"]
        print(f"\n  Первые 15 сделок:\n{result.trades[cols].head(15).to_string(index=False)}")

    html = "sr_bounce_grid_1d.html"
    plot_grid_backtest(
        result, df, strategy,
        title=(
            f"{SYMBOL} {TIMEFRAME} | S/R Bounce Grid DCA | "
            f"7 ордеров: вход → ±5% → 5×(±1.5%) | TP=3% avg | без SL | лонг+шорт | 2020→"
        ),
        save_html=html,
    )
