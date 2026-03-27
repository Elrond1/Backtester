"""
Strategy #2 realistic bets:
  BTC:        B=$200, C=$300, D=$200, D_C=$300
  DOGE/BNB:   B=$75,  C=$100, D=$75,  D_C=$100

Also tests different entry prices (52¢, 54¢, 56¢, 58¢) to show sensitivity.
"""

import sys
sys.path.insert(0, "/Users/arturbagian/Python_proj/Backtester")

import pandas as pd
from backtester.data.manager import get_ohlcv

PAIRS   = ["BTC/USDT", "DOGE/USDT", "BNB/USDT"]
SINCE   = "2020-01-01"
CAPITAL = 10_000

BETS = {
    "BTC/USDT":  {"B": 200, "C": 300, "D": 200, "D_C": 300},
    "DOGE/USDT": {"B":  75, "C": 100, "D":  75, "D_C": 100},
    "BNB/USDT":  {"B":  75, "C": 100, "D":  75, "D_C": 100},
}


def get_trades(pair, df15, df1h):
    df15 = df15.copy()
    df15["green"] = df15["close"] > df15["open"]
    trades = []
    prev_b_win = None
    prev_d_win = None

    for i in range(6, len(df15) - 1):
        ts = df15.index[i]
        if ts.minute != 0:
            continue
        if ts not in df1h.index:
            continue

        c1_green = bool(df15["green"].iloc[i])
        h = df1h.loc[ts]
        hour_green = bool(h["close"] > h["open"])
        win_hour = (hour_green == c1_green)

        # B/C: streak=5 exclusive
        streak5 = df15["green"].iloc[i-5:i].values
        if all(streak5) or not any(streak5):
            s5c = bool(streak5[0])
            if bool(df15["green"].iloc[i-6]) != s5c and c1_green != s5c:
                sig = "C" if prev_b_win is True else "B"
                trades.append({"ts": ts, "pair": pair, "signal": sig, "win": win_hour})
                prev_b_win = bool(win_hour)
                continue

        # D/D_C: last 3 candles same color + confirmation (not B)
        # Matches original Strategy #2 logic: streak=3 or 4 (anything that's not exclusive-5)
        streak3 = df15["green"].iloc[i-3:i].values
        if all(streak3) or not any(streak3):
            s3c = bool(streak3[0])
            if c1_green != s3c:
                sig = "D_C" if prev_d_win is True else "D"
                trades.append({"ts": ts, "pair": pair, "signal": sig, "win": win_hour})
                prev_d_win = bool(win_hour)

    return trades


def simulate(all_trades, entry_price):
    df = pd.DataFrame(all_trades).sort_values("ts").reset_index(drop=True)
    bets_col = []
    for _, row in df.iterrows():
        b = BETS[row["pair"]][row["signal"]]
        bets_col.append(b)
    df["bet"] = bets_col
    df["pnl"] = df.apply(
        lambda r: r["bet"] * (1 / entry_price - 1) if r["win"] else -r["bet"], axis=1
    )
    df["cum_pnl"] = df["pnl"].cumsum()

    peak = 0
    max_dd = 0
    for v in df["cum_pnl"]:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    months = (df["ts"].max() - df["ts"].min()).days / 30.44
    total_pnl = df["pnl"].sum()
    win_rate = df["win"].mean() * 100

    df["month"] = df["ts"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()

    return {
        "entry_price": entry_price,
        "win_rate": win_rate,
        "monthly_pnl": total_pnl / months,
        "annual_pnl": total_pnl / months * 12,
        "max_dd": max_dd,
        "max_dd_pct": max_dd / CAPITAL * 100,
        "losing_months": (monthly < 0).sum(),
        "worst_month": monthly.min(),
        "best_month": monthly.max(),
        "total_months": len(monthly),
    }


def main():
    print("Loading data...")
    all_trades = []
    for pair in PAIRS:
        symbol = pair.split("/")[0]
        print(f"  {symbol}...", end=" ", flush=True)
        df15 = get_ohlcv(pair, "15m", since=SINCE)
        df1h = get_ohlcv(pair, "1h", since=SINCE)
        trades = get_trades(pair, df15, df1h)
        all_trades.extend(trades)
        print(f"{len(trades):,} trades")

    print(f"\n{'='*65}")
    print(f"  Strategy #2 REALISTIC  |  BTC $200/$300, DOGE+BNB $75/$100")
    print(f"{'='*65}")
    print(f"\n  Чувствительность к цене входа:")
    print(f"  {'Цена':>6} {'Win%':>6} {'P&L/мес':>10} {'12 мес':>12} {'Макс DD':>10} {'Убыт мес':>10}")
    print(f"  {'-'*58}")

    for price in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
        r = simulate(all_trades, price)
        losing = f"{r['losing_months']}/{r['total_months']}"
        marker = " ← реалист." if price == 0.52 else (" ← осторожно" if price == 0.56 else "")
        print(f"  {price:.2f}¢  {r['win_rate']:>5.1f}%  ${r['monthly_pnl']:>8,.0f}  ${r['annual_pnl']:>10,.0f}  {r['max_dd_pct']:>8.1f}%  {losing:>8}{marker}")

    # Detailed view at 52¢
    r52 = simulate(all_trades, 0.52)
    print(f"\n  При 52¢ (базовый сценарий):")
    print(f"    P&L/мес       : ${r52['monthly_pnl']:,.0f}")
    print(f"    За 12 мес     : ${r52['annual_pnl']:,.0f}  ({r52['annual_pnl']/CAPITAL*100:.0f}%)")
    print(f"    Макс просадка : ${r52['max_dd']:,.0f}  ({r52['max_dd_pct']:.1f}% от $10k)")
    print(f"    Худший месяц  : ${r52['worst_month']:,.0f}")
    print(f"    Лучший месяц  : ${r52['best_month']:,.0f}")
    print(f"    Убыточных мес : {r52['losing_months']} из {r52['total_months']}")

    # Breakeven price
    print(f"\n  При 60¢ стратегия всё ещё в плюсе: {simulate(all_trades, 0.60)['losing_months']==0}")


if __name__ == "__main__":
    main()
