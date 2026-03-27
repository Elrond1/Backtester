"""
Grid test: all pairs × all streak lengths (2–7)
Signal logic: streak N×15m + first 15m of new hour confirms reversal → enter 1h contract
Win = 1h candle closes in direction of first 15m (reversal direction)

Pairs: BTC, ETH, BNB, SOL, XRP, DOGE
Streaks: 2, 3, 4, 5, 6, 7
Period: 2020-01-01 → now
"""

import sys
sys.path.insert(0, "/Users/arturbagian/Python_proj/Backtester")

import pandas as pd
from backtester.data.manager import get_ohlcv

PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
STREAKS = [2, 3, 4, 5, 6, 7]
SINCE = "2020-01-01"


def test_streak(df15: pd.DataFrame, df1h: pd.DataFrame, streak_len: int):
    """
    Exclusive streak: exactly streak_len same-color candles before hour start.
    The candle at position i-streak_len-1 must be DIFFERENT color (breaks the streak).
    """
    df15 = df15.copy()
    df15["green"] = df15["close"] > df15["open"]

    signals = 0
    wins = 0

    for i in range(streak_len + 1, len(df15) - 1):
        ts = df15.index[i]

        # Must be start of a new hour
        if ts.minute != 0:
            continue

        # Must have corresponding 1h candle
        if ts not in df1h.index:
            continue

        # Check streak: last N candles before this one (i-streak_len : i)
        streak = df15["green"].iloc[i - streak_len:i].values
        if not (all(streak) or not any(streak)):
            continue

        streak_color = bool(streak[0])

        # EXCLUSIVE: candle before the streak must be opposite color
        candle_before = bool(df15["green"].iloc[i - streak_len - 1])
        if candle_before == streak_color:
            continue  # Streak is actually longer — skip

        # First 15m of new hour (index i) must be opposite direction (confirmation)
        c1_green = bool(df15["green"].iloc[i])
        if c1_green == streak_color:
            continue

        # Signal! Win = 1h candle closes in same direction as c1
        h = df1h.loc[ts]
        hour_green = bool(h["close"] > h["open"])

        signals += 1
        if hour_green == c1_green:
            wins += 1

    months = (df15.index[-1] - df15.index[0]).days / 30.44
    win_pct = wins / signals * 100 if signals else 0
    per_month = signals / months if months else 0

    return {
        "signals": signals,
        "wins": wins,
        "win%": round(win_pct, 1),
        "per_month": round(per_month, 1),
    }


def main():
    # Pre-load all data
    print("Loading data...")
    data = {}
    for pair in PAIRS:
        symbol = pair.split("/")[0]
        print(f"  {pair}...", end=" ", flush=True)
        df15 = get_ohlcv(pair, "15m", since=SINCE)
        df1h = get_ohlcv(pair, "1h", since=SINCE)
        data[pair] = (df15, df1h)
        print(f"{len(df15):,} bars")

    print("\n" + "="*90)
    print(f"  {'Pair':<12} | " + " | ".join(f"streak={s}  " for s in STREAKS))
    print(f"  {'':12} | " + " | ".join("win%  /mo " for _ in STREAKS))
    print("="*90)

    # Results matrix for summary
    results = {}

    for pair in PAIRS:
        df15, df1h = data[pair]
        row = {}
        parts = []
        for streak_len in STREAKS:
            r = test_streak(df15, df1h, streak_len)
            row[streak_len] = r
            parts.append(f"{r['win%']:>4.1f}% {r['per_month']:>4.1f}")
        results[pair] = row
        symbol = pair.split("/")[0]
        print(f"  {symbol:<12} | " + " | ".join(parts))

    print("="*90)

    # Find best streak per pair (by win% with min 5 signals/month)
    print("\n  Best streak per pair (min 5 signals/month):")
    print(f"  {'Pair':<12} | {'Best streak':<12} | {'Win%':<8} | {'Signals/mo':<12} | EV at 52¢")
    print("  " + "-"*60)
    for pair in PAIRS:
        symbol = pair.split("/")[0]
        best = max(
            ((s, r) for s, r in results[pair].items() if r["per_month"] >= 5),
            key=lambda x: x[1]["win%"],
            default=(None, None)
        )
        if best[0] is None:
            print(f"  {symbol:<12} | {'—':<12} | {'—':<8} | {'<5/mo':<12} |")
            continue
        s, r = best
        ev = r["win%"] / 100 * (1 / 0.52 - 1) - (1 - r["win%"] / 100)
        print(f"  {symbol:<12} | streak={s:<5}     | {r['win%']:.1f}%    | {r['per_month']:.1f}/mo       | {ev*100:+.1f}%")


if __name__ == "__main__":
    main()
