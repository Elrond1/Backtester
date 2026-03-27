"""
BreakBar — бинарное предсказание направления следующей свечи.

Логика:
  Пробой зоны вверх (long)  → следующая свеча бычья (close > open) = WIN, иначе LOSS
  Пробой зоны вниз (short) → следующая свеча медвежья (close < open) = WIN, иначе LOSS

Таймфреймы: M15, M30, H1, H4 — каждый отдельный расчёт и отдельный график.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.data import get_ohlcv
from backtester.strategy.breakbar import BreakBar


# ──────────────────────────────────────────────────────────────────────────────
def analyse_next_candle(df: pd.DataFrame, strategy: BreakBar) -> pd.DataFrame:
    """
    Для каждого пробоя смотрим на следующую свечу:
      Long  → WIN если close > open, иначе LOSS
      Short → WIN если close < open, иначе LOSS
    """
    signals = strategy.generate_signals(df)

    op    = df["open"].values
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    times = df.index
    n     = len(df)

    rows = []
    for i in range(n - 1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        j = i + 1
        if sig == 1:
            result = "win" if close[j] > op[j] else "loss"
        else:
            result = "win" if close[j] < op[j] else "loss"

        rows.append({
            "breakout_time": times[i],
            "next_bar_time": times[j],
            "side":          "long" if sig == 1 else "short",
            "bar_high":      high[i],
            "bar_low":       low[i],
            "result":        result,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
def print_stats(tf: str, df_sig: pd.DataFrame) -> None:
    if df_sig.empty:
        print(f"  [{tf}] Нет сигналов.")
        return

    total = len(df_sig)
    w = (df_sig["result"] == "win").sum()
    l = (df_sig["result"] == "loss").sum()

    print(f"\n{'═'*52}")
    print(f"  {tf}  —  всего пробоев: {total}")
    print(f"{'─'*52}")
    print(f"  WIN  : {w:>4}  ({w/total*100:5.1f}%)")
    print(f"  LOSS : {l:>4}  ({l/total*100:5.1f}%)")
    print(f"{'─'*52}")

    for side_name in ["long", "short"]:
        s = df_sig[df_sig["side"] == side_name]
        if s.empty:
            continue
        s_w = (s["result"] == "win").sum()
        s_l = (s["result"] == "loss").sum()
        s_t = len(s)
        print(f"  {side_name.upper():5s}  WIN: {s_w:>3} ({s_w/s_t*100:4.1f}%)  "
              f"LOSS: {s_l:>3} ({s_l/s_t*100:4.1f}%)  всего: {s_t}")


# ──────────────────────────────────────────────────────────────────────────────
def plot_breakbar(
    df: pd.DataFrame,
    df_sig: pd.DataFrame,
    tf: str,
    symbol: str,
    save_html: str,
) -> None:
    """
    График:
      Row 1 (большой): свечи + маркеры пробоев (WIN=зелёный, LOSS=красный)
      Row 2 (малый):   накопленный win rate по времени
    """
    if df_sig.empty:
        print(f"  [{tf}] Нет данных для графика.")
        return

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
        subplot_titles=(
            f"{symbol} {tf} — BreakBar сигналы",
            "Накопленный Win Rate %",
        ),
    )

    # ── свечи ──────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── маркеры пробоев ────────────────────────────────────────────────────
    # Long WIN  — зелёный треугольник вверх (под баром)
    # Long LOSS — красный треугольник вверх (под баром)
    # Short WIN  — красный треугольник вниз (над баром)  ← цвет по направлению
    # Short LOSS — зелёный треугольник вниз (над баром)
    # Заливка: WIN=полная, LOSS=контур

    marker_cfg = {
        #              symbol            color      opacity  outline
        ("long",  "win"):  ("triangle-up",   "#00e676", 1.0,    False),
        ("long",  "loss"): ("triangle-up",   "#ef5350", 1.0,    False),
        ("short", "win"):  ("triangle-down", "#ef5350", 1.0,    False),
        ("short", "loss"): ("triangle-down", "#00e676", 1.0,    False),
    }

    for (side, res), (sym, color, opacity, _) in marker_cfg.items():
        mask = (df_sig["side"] == side) & (df_sig["result"] == res)
        sub  = df_sig[mask]
        if sub.empty:
            continue

        # позиция маркера — чуть ниже low (long) или выше high (short)
        if side == "long":
            y_vals = sub["bar_low"] * 0.9985
        else:
            y_vals = sub["bar_high"] * 1.0015

        label = f"{'↑' if side=='long' else '↓'} {side.upper()} {'WIN' if res=='win' else 'LOSS'}"

        fig.add_trace(
            go.Scatter(
                x=sub["breakout_time"],
                y=y_vals,
                mode="markers",
                name=label,
                marker=dict(
                    symbol=sym,
                    size=10,
                    color=color,
                    opacity=opacity,
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "%{x}<br>"
                    "Цена: %{y:.2f}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # ── накопленный win rate ───────────────────────────────────────────────
    df_sig_sorted = df_sig.sort_values("breakout_time")
    is_win        = (df_sig_sorted["result"] == "win").astype(int)
    cumwin        = is_win.cumsum()
    cumtotal      = pd.Series(range(1, len(df_sig_sorted) + 1), index=df_sig_sorted.index)
    cumwr         = cumwin / cumtotal * 100

    fig.add_trace(
        go.Scatter(
            x=df_sig_sorted["breakout_time"],
            y=cumwr.values,
            mode="lines",
            name="Win Rate %",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.10)",
        ),
        row=2, col=1,
    )
    # линия 50%
    fig.add_hline(
        y=50, row=2, col=1,
        line=dict(color="#888", width=1, dash="dot"),
        annotation_text="50%",
        annotation_position="right",
    )

    # ── итоговая статистика (текстовый блок) ──────────────────────────────
    total = len(df_sig)
    w = (df_sig["result"] == "win").sum()
    l = total - w
    final_wr = w / total * 100

    stats_lines = [
        f"<b>Пробоев: {total}</b>",
        f"WIN  : {w}  ({w/total*100:.1f}%)",
        f"LOSS : {l}  ({l/total*100:.1f}%)",
        f"Win Rate: <b>{final_wr:.1f}%</b>",
    ]
    # по направлению
    for side_name in ["long", "short"]:
        s = df_sig[df_sig["side"] == side_name]
        if not s.empty:
            sw = (s["result"] == "win").sum()
            stats_lines.append(
                f"{side_name.upper()}: {sw}/{len(s)}  ({sw/len(s)*100:.1f}%)"
            )

    fig.add_annotation(
        text="<br>".join(stats_lines),
        xref="paper", yref="paper",
        x=0.01, y=0.27,
        xanchor="left", yanchor="top",
        showarrow=False,
        align="left",
        font=dict(size=12, color="#e0e0e0", family="monospace"),
        bgcolor="rgba(30,30,40,0.88)",
        bordercolor="#555",
        borderwidth=1,
        borderpad=8,
    )

    # ── стиль ─────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=800,
        title=dict(
            text=f"{symbol} {tf} — BreakBar | предсказание следующей свечи",
            font=dict(size=15),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(row=1, col=1, title_text="Price")
    fig.update_yaxes(row=2, col=1, title_text="Win Rate %", range=[0, 100])

    fig.write_html(save_html)
    fig.show()
    print(f"  [{tf}] График → {save_html}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SYMBOL = "BTC/USDT"
    SINCE  = "2020-01-01"

    strategy = BreakBar(
        min_size_pct=2.0,   # минимальный размер импульсного бара = 2% от цены
        dist_hl=0,
        min_bars_in=5,
        zone_in_zone=True,
    )

    TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]
    all_results = {}

    for tf in TIMEFRAMES:
        print(f"\n  Загрузка {SYMBOL} {tf} с {SINCE} …")
        df = get_ohlcv(SYMBOL, tf, since=SINCE)
        print(f"  Загружено {len(df):,} свечей  ({df.index[0].date()} → {df.index[-1].date()})")

        df_sig = analyse_next_candle(df, strategy)
        all_results[tf] = df_sig

        print_stats(tf, df_sig)

        plot_breakbar(
            df=df,
            df_sig=df_sig,
            tf=tf,
            symbol=SYMBOL,
            save_html=f"breakbar_{tf.replace('m','M').replace('h','H').replace('1d','1D')}.html",
        )

    # ── сводная таблица ────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  СВОДКА ПО ТАЙМФРЕЙМАМ")
    print(f"{'─'*60}")
    header = f"  {'ТФ':<6} {'Всего':>6} {'WIN':>6} {'LOSS':>6} {'Win%':>8}  {'Long%':>7}  {'Short%':>7}"
    print(header)
    print(f"  {'-'*58}")

    for tf in TIMEFRAMES:
        d = all_results[tf]
        if d.empty:
            print(f"  {tf:<6}  — нет данных —")
            continue
        total = len(d)
        w  = (d["result"] == "win").sum()
        l  = total - w
        wr = w / total * 100

        lng = d[d["side"] == "long"]
        sht = d[d["side"] == "short"]
        lng_wr = (lng["result"] == "win").mean() * 100 if not lng.empty else float("nan")
        sht_wr = (sht["result"] == "win").mean() * 100 if not sht.empty else float("nan")

        print(f"  {tf:<6} {total:>6} {w:>6} {l:>6} {wr:>7.1f}%  {lng_wr:>6.1f}%  {sht_wr:>6.1f}%")

    print(f"{'═'*60}")
