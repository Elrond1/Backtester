"""
Strategy #2 aggressive bets (B/C/D/D_C unchanged) + E/E_C at small bets ($50/$75)
Pairs: BTC + DOGE + BNB
"""

import sys
sys.path.insert(0, "/Users/arturbagian/Python_proj/Backtester")

import pandas as pd
from backtester.data.manager import get_ohlcv

PAIRS = ["BTC/USDT", "DOGE/USDT", "BNB/USDT"]
SINCE = "2020-01-01"
ENTRY_PRICE = 0.52
CAPITAL = 10_000

BETS = {
    "B":   400,   # Strategy #2 aggressive unchanged
    "C":   600,
    "D":   400,
    "D_C": 600,
    "E":    50,   # New, small
    "E_C":  75,
}


def get_trades(pair, df15, df1h):
    df15 = df15.copy()
    df15["green"] = df15["close"] > df15["open"]
    trades = []
    prev_b_win = None
    prev_d_win = None
    prev_e_win = None

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

        # Signal B/C: streak=5 exclusive
        streak5 = df15["green"].iloc[i-5:i].values
        if all(streak5) or not any(streak5):
            streak5_color = bool(streak5[0])
            if bool(df15["green"].iloc[i-6]) != streak5_color and c1_green != streak5_color:
                sig = "C" if prev_b_win is True else "B"
                bet = BETS[sig]
                pnl = bet * (1 / ENTRY_PRICE - 1) if win_hour else -bet
                trades.append({"ts": ts, "pair": pair, "signal": sig, "win": win_hour, "pnl": pnl})
                prev_b_win = bool(win_hour)
                continue

        # Signal D/D_C: streak=3 exclusive
        streak3 = df15["green"].iloc[i-3:i].values
        if all(streak3) or not any(streak3):
            streak3_color = bool(streak3[0])
            if bool(df15["green"].iloc[i-4]) != streak3_color and c1_green != streak3_color:
                sig = "D_C" if prev_d_win is True else "D"
                bet = BETS[sig]
                pnl = bet * (1 / ENTRY_PRICE - 1) if win_hour else -bet
                trades.append({"ts": ts, "pair": pair, "signal": sig, "win": win_hour, "pnl": pnl})
                prev_d_win = bool(win_hour)
                continue

        # Signal E/E_C: streak=2 exclusive
        streak2 = df15["green"].iloc[i-2:i].values
        if all(streak2) or not any(streak2):
            streak2_color = bool(streak2[0])
            if bool(df15["green"].iloc[i-3]) != streak2_color and c1_green != streak2_color:
                sig = "E_C" if prev_e_win is True else "E"
                bet = BETS[sig]
                pnl = bet * (1 / ENTRY_PRICE - 1) if win_hour else -bet
                trades.append({"ts": ts, "pair": pair, "signal": sig, "win": win_hour, "pnl": pnl})
                prev_e_win = bool(win_hour)

    return trades


def max_drawdown(cum_series):
    peak = 0
    max_dd = 0
    for v in cum_series:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


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

    df = pd.DataFrame(all_trades).sort_values("ts").reset_index(drop=True)
    df["cum_pnl"] = df["pnl"].cumsum()
    months = (df["ts"].max() - df["ts"].min()).days / 30.44

    dd = max_drawdown(df["cum_pnl"])
    total_pnl = df["pnl"].sum()

    # Separate E contribution
    df_s2 = df[df["signal"].isin(["B","C","D","D_C"])]
    df_e  = df[df["signal"].isin(["E","E_C"])]

    print(f"\n{'='*60}")
    print(f"  Strategy #2 агрессивный + E/E_C ($50/$75)")
    print(f"{'='*60}")
    print(f"  Всего сделок  : {len(df):,}  ({len(df)/months:.0f}/мес)")
    print(f"  Win rate      : {df['win'].mean()*100:.1f}%")
    print(f"  P&L/мес       : ${total_pnl/months:,.0f}")
    print(f"  Макс просадка : ${dd:,.0f}  ({dd/CAPITAL*100:.1f}% от $10k)")

    print(f"\n  По сигналам:")
    print(f"  {'Сигнал':<8} {'Сделок':>7} {'Win%':>6} {'Ставка':>8} {'P&L':>10} {'/мес':>8}")
    print(f"  {'-'*52}")
    for sig in ["B","C","D","D_C","E","E_C"]:
        sub = df[df["signal"] == sig]
        if sub.empty: continue
        w = sub["win"].mean() * 100
        p = sub["pnl"].sum()
        print(f"  {sig:<8} {len(sub):>7,} {w:>5.1f}% ${BETS[sig]:>7} ${p:>9,.0f} ${p/months:>7,.0f}")

    print(f"\n  Только Strategy #2 (B/C/D/D_C):")
    p2 = df_s2["pnl"].sum()
    dd2 = max_drawdown(df_s2.sort_values("ts")["pnl"].cumsum())
    print(f"    P&L/мес: ${p2/months:,.0f}  |  DD: ${dd2:,.0f} ({dd2/CAPITAL*100:.1f}%)")

    print(f"\n  Только E/E_C добавка:")
    pe = df_e["pnl"].sum()
    print(f"    P&L/мес: ${pe/months:,.0f}  |  Сделок: {len(df_e)/months:.0f}/мес")

    df["month"] = df["ts"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    print(f"\n  Убыточных месяцев: {(monthly<0).sum()} из {len(monthly)}")
    print(f"  Лучший месяц     : ${monthly.max():,.0f}")
    print(f"  Худший месяц     : ${monthly.min():,.0f}")

    print(f"\n  Проекция:")
    m = total_pnl / months
    for label, n in [("1 мес", 1),("3 мес", 3),("6 мес", 6),("12 мес", 12)]:
        print(f"    {label}: ${m*n:,.0f}  ({m*n/CAPITAL*100:.0f}%)")


if __name__ == "__main__":
    main()
