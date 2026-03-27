"""
Deep analysis of exclusive streak=2 signal across all pairs.
Streak=2: exactly 2 same-color 15m candles before hour start (not 3+),
first 15m of new hour confirms reversal → enter 1h Polymarket contract.

Shows: monthly win%, max consecutive losses, losing months, P&L simulation.
"""

import sys
sys.path.insert(0, "/Users/arturbagian/Python_proj/Backtester")

import pandas as pd
from backtester.data.manager import get_ohlcv

PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
SINCE = "2020-01-01"
ENTRY_PRICE = 0.52   # realistic Polymarket entry
BET = 100            # $100 per trade for simulation


def get_trades(df15: pd.DataFrame, df1h: pd.DataFrame):
    df15 = df15.copy()
    df15["green"] = df15["close"] > df15["open"]
    trades = []

    for i in range(3, len(df15) - 1):
        ts = df15.index[i]
        if ts.minute != 0:
            continue
        if ts not in df1h.index:
            continue

        # Exactly 2 same-color candles (positions i-2, i-1)
        streak = df15["green"].iloc[i-2:i].values
        if not (all(streak) or not any(streak)):
            continue

        streak_color = bool(streak[0])

        # Candle before streak (i-3) must be opposite → streak is exactly 2
        candle_before = bool(df15["green"].iloc[i-3])
        if candle_before == streak_color:
            continue

        # First 15m of new hour must confirm reversal
        c1_green = bool(df15["green"].iloc[i])
        if c1_green == streak_color:
            continue

        # Outcome
        h = df1h.loc[ts]
        hour_green = bool(h["close"] > h["open"])
        win = (hour_green == c1_green)

        pnl = BET * (1 / ENTRY_PRICE - 1) if win else -BET

        trades.append({
            "ts": ts,
            "month": ts.to_period("M"),
            "win": win,
            "pnl": pnl,
        })

    return pd.DataFrame(trades)


def analyze(pair: str, df: pd.DataFrame):
    if df.empty:
        print(f"  {pair}: no trades")
        return

    symbol = pair.split("/")[0]
    months_total = (df["ts"].max() - df["ts"].min()).days / 30.44
    total = len(df)
    wins = df["win"].sum()
    win_pct = wins / total * 100
    per_month = total / months_total

    # Max consecutive losses
    max_loss_streak = 0
    cur = 0
    for w in df["win"]:
        if not w:
            cur += 1
            max_loss_streak = max(max_loss_streak, cur)
        else:
            cur = 0

    # Max consecutive wins
    max_win_streak = 0
    cur = 0
    for w in df["win"]:
        if w:
            cur += 1
            max_win_streak = max(max_win_streak, cur)
        else:
            cur = 0

    # Monthly stats
    monthly = df.groupby("month").agg(
        trades=("win", "count"),
        wins=("win", "sum"),
        pnl=("pnl", "sum"),
    )
    monthly["win%"] = monthly["wins"] / monthly["trades"] * 100
    losing_months = (monthly["pnl"] < 0).sum()
    best_month = monthly["pnl"].max()
    worst_month = monthly["pnl"].min()

    # Cumulative P&L
    df["cum_pnl"] = df["pnl"].cumsum()
    max_drawdown = 0
    peak = 0
    for v in df["cum_pnl"]:
        if v > peak:
            peak = v
        dd = (peak - v)
        if dd > max_drawdown:
            max_drawdown = dd

    total_pnl = df["pnl"].sum()
    ev = win_pct / 100 * (1 / ENTRY_PRICE - 1) - (1 - win_pct / 100)

    print(f"\n{'='*60}")
    print(f"  {symbol}  |  streak=2 exclusive  |  $100/trade @ {ENTRY_PRICE}¢")
    print(f"{'='*60}")
    print(f"  Период       : {df['ts'].min().date()} — {df['ts'].max().date()}")
    print(f"  Сделок       : {total:,}  ({per_month:.1f}/мес)")
    print(f"  Win rate     : {win_pct:.1f}%")
    print(f"  EV на $1     : {ev*100:+.1f}%")
    print(f"  Макс минусов подряд : {max_loss_streak}")
    print(f"  Макс плюсов подряд  : {max_win_streak}")
    print(f"  Убыточных месяцев   : {losing_months} из {len(monthly)}")
    print(f"  Лучший месяц : ${best_month:,.0f}")
    print(f"  Худший месяц : ${worst_month:,.0f}")
    print(f"  Макс просадка: ${max_drawdown:,.0f}  ({max_drawdown / (10000) * 100:.1f}% от $10k)")
    print(f"  Итого P&L    : ${total_pnl:,.0f}  за {months_total:.0f} мес")

    # Monthly win% distribution
    print(f"\n  Помесячный win%:")
    print(f"  {'Месяц':<10} {'Сделок':>7} {'Win%':>6} {'P&L':>8}")
    print(f"  {'-'*35}")
    for m, row in monthly.iterrows():
        marker = " ←" if row["pnl"] < 0 else ""
        print(f"  {str(m):<10} {row['trades']:>7.0f} {row['win%']:>5.1f}% ${row['pnl']:>7,.0f}{marker}")


def main():
    print("Loading data...")
    for pair in PAIRS:
        symbol = pair.split("/")[0]
        print(f"  {symbol}...", end=" ", flush=True)
        df15 = get_ohlcv(pair, "15m", since=SINCE)
        df1h = get_ohlcv(pair, "1h", since=SINCE)
        print(f"{len(df15):,} bars")
        trades = get_trades(df15, df1h)
        analyze(pair, trades)

    print(f"\n{'='*60}")
    print("  Done.")


if __name__ == "__main__":
    main()
