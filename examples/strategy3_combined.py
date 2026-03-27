"""
Strategy #3 = Strategy #2 (B/C/D/D_C streaks 3+5) + streak=2 exclusive (signal E/E_C)
Pairs: BTC + DOGE + BNB
Target: keep max drawdown ~10% of $10,000 capital

Signals:
  B/C   — streak=5 exclusive + 1h confirmation
  D/D_C — streak=3 exclusive + 1h confirmation
  E/E_C — streak=2 exclusive + 1h confirmation (NEW)
"""

import sys
sys.path.insert(0, "/Users/arturbagian/Python_proj/Backtester")

import pandas as pd
from backtester.data.manager import get_ohlcv

PAIRS = ["BTC/USDT", "DOGE/USDT", "BNB/USDT"]
SINCE = "2020-01-01"
ENTRY_PRICE = 0.52

# Bet sizes — scaled to keep ~10% max drawdown on $10k
# Strategy #2 aggressive had ~$400/$600 per signal on 3 pairs
# Adding streak=2 roughly triples trade volume → scale all down ~3x
BETS = {
    "B":   133,
    "C":   200,
    "D":   133,
    "D_C": 200,
    "E":   100,
    "E_C": 150,
}

CAPITAL = 10_000


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

        # --- Signal B/C: streak=5 exclusive ---
        streak5 = df15["green"].iloc[i-5:i].values
        is_5 = (all(streak5) or not any(streak5))
        if is_5:
            streak5_color = bool(streak5[0])
            candle_before5 = bool(df15["green"].iloc[i-6])
            exclusive5 = (candle_before5 != streak5_color)
            confirm5 = (c1_green != streak5_color)
            if exclusive5 and confirm5:
                sig = "C" if prev_b_win is True else "B"
                bet = BETS[sig]
                pnl = bet * (1 / ENTRY_PRICE - 1) if win_hour else -bet
                trades.append({"ts": ts, "pair": pair, "signal": sig, "win": win_hour, "pnl": pnl})
                prev_b_win = bool(win_hour)
                continue  # exclusive — skip D and E checks

        # --- Signal D/D_C: streak=3 exclusive (not 5) ---
        streak3 = df15["green"].iloc[i-3:i].values
        is_3 = (all(streak3) or not any(streak3))
        if is_3:
            streak3_color = bool(streak3[0])
            candle_before3 = bool(df15["green"].iloc[i-4])
            exclusive3 = (candle_before3 != streak3_color)
            confirm3 = (c1_green != streak3_color)
            if exclusive3 and confirm3:
                sig = "D_C" if prev_d_win is True else "D"
                bet = BETS[sig]
                pnl = bet * (1 / ENTRY_PRICE - 1) if win_hour else -bet
                trades.append({"ts": ts, "pair": pair, "signal": sig, "win": win_hour, "pnl": pnl})
                prev_d_win = bool(win_hour)
                continue

        # --- Signal E/E_C: streak=2 exclusive (not 3, not 5) ---
        streak2 = df15["green"].iloc[i-2:i].values
        is_2 = (all(streak2) or not any(streak2))
        if is_2:
            streak2_color = bool(streak2[0])
            candle_before2 = bool(df15["green"].iloc[i-3])
            exclusive2 = (candle_before2 != streak2_color)
            confirm2 = (c1_green != streak2_color)
            if exclusive2 and confirm2:
                sig = "E_C" if prev_e_win is True else "E"
                bet = BETS[sig]
                pnl = bet * (1 / ENTRY_PRICE - 1) if win_hour else -bet
                trades.append({"ts": ts, "pair": pair, "signal": sig, "win": win_hour, "pnl": pnl})
                prev_e_win = bool(win_hour)

    return trades


def max_drawdown_pct(cum_pnl_series, capital):
    peak = 0
    max_dd = 0
    for v in cum_pnl_series:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd, max_dd / capital * 100


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

    dd_abs, dd_pct = max_drawdown_pct(df["cum_pnl"], CAPITAL)
    total_pnl = df["pnl"].sum()
    win_rate = df["win"].mean() * 100

    print(f"\n{'='*60}")
    print(f"  СТРАТЕГИЯ №3: B/C + D/D_C + E/E_C  |  BTC+DOGE+BNB")
    print(f"{'='*60}")
    print(f"  Всего сделок  : {len(df):,}  ({len(df)/months:.0f}/мес)")
    print(f"  Win rate      : {win_rate:.1f}%")
    print(f"  Итого P&L     : ${total_pnl:,.0f}  за {months:.0f} мес")
    print(f"  P&L/мес       : ${total_pnl/months:,.0f}")
    print(f"  Макс просадка : ${dd_abs:,.0f}  ({dd_pct:.1f}% от $10k)")

    # By signal
    print(f"\n  По сигналам:")
    print(f"  {'Сигнал':<8} {'Сделок':>7} {'Win%':>6} {'Ставка':>8} {'P&L':>10} {'/мес':>8}")
    print(f"  {'-'*52}")
    for sig in ["B", "C", "D", "D_C", "E", "E_C"]:
        sub = df[df["signal"] == sig]
        if sub.empty:
            continue
        w = sub["win"].mean() * 100
        p = sub["pnl"].sum()
        print(f"  {sig:<8} {len(sub):>7,} {w:>5.1f}% ${BETS[sig]:>7} ${p:>9,.0f} ${p/months:>7,.0f}")

    # Monthly
    df["month"] = df["ts"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    losing = (monthly < 0).sum()

    print(f"\n  Убыточных месяцев: {losing} из {len(monthly)}")
    print(f"  Лучший месяц     : ${monthly.max():,.0f}")
    print(f"  Худший месяц     : ${monthly.min():,.0f}")

    # Projection
    monthly_avg = total_pnl / months
    print(f"\n  Проекция (фиксированные ставки):")
    print(f"  {'Период':<10} {'P&L':>12} {'% от $10k':>10}")
    print(f"  {'-'*35}")
    for label, m in [("1 мес", 1), ("3 мес", 3), ("6 мес", 6), ("12 мес", 12)]:
        pnl = monthly_avg * m
        pct = pnl / CAPITAL * 100
        print(f"  {label:<10} ${pnl:>10,.0f} {pct:>9.0f}%")


if __name__ == "__main__":
    main()
