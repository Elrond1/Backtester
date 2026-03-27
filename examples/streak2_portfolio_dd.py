"""
ПОЛНЫЙ АНАЛИЗ — все комбинации стриков и условий входа.

Проверяем:
1. Без подтверждения (вход 00:00)
2. С 1-й 15m подтверждением (вход 00:15) — наша текущая стратегия
3. С 2-й 15m подтверждением (вход 00:30)
4. По длине стрика 2-7 (эксклюзивные)
5. По цвету стрика (зелёный vs красный)
6. "Чистый стрик" — стрик целиком в предыдущем часе (4 свечи = 1 час)
7. По сессии (Азия / Лондон / США)
"""

import sys
sys.path.insert(0, "/Users/arturbagian/Python_proj/Backtester")

import pandas as pd
import numpy as np
from backtester.data.manager import get_ohlcv

PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
SINCE = "2020-01-01"

SESSIONS = {
    "Азия":   (0,  8),   # UTC 00-08
    "Лондон": (8,  16),  # UTC 08-16
    "США":    (16, 24),  # UTC 16-24
}


def analyze_btc():
    """Deep analysis on BTC only — all variations."""
    print("Загружаем BTC...")
    df15 = get_ohlcv("BTC/USDT", "15m", since=SINCE)
    df1h = get_ohlcv("BTC/USDT", "1h", since=SINCE)
    df15["green"] = df15["close"] > df15["open"]

    months = (df15.index[-1] - df15.index[0]).days / 30.44

    # ─────────────────────────────────────────────
    # БЛОК 1: без подтверждения vs +15m vs +30m
    # ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  БЛОК 1: Эффект подтверждения (BTC, стрик=3 эксклюзивный)")
    print("="*70)
    print(f"  {'Метод':<30} {'Сделок':>7} {'Win%':>6} {'/мес':>6} {'EV@52¢':>8}")
    print(f"  {'-'*55}")

    for confirm_candles, label in [(0, "Без подтверж. (вход 00:00)"),
                                    (1, "1-я 15m подтвержд. (вход 00:15)"),
                                    (2, "2-я 15m подтвержд. (вход 00:30)")]:
        sigs, wins = 0, 0
        for i in range(4, len(df15) - confirm_candles - 1):
            ts = df15.index[i]
            if ts.minute != 0:
                continue
            if ts not in df1h.index:
                continue

            streak = df15["green"].iloc[i-3:i].values
            if not (all(streak) or not any(streak)):
                continue
            sc = bool(streak[0])
            if bool(df15["green"].iloc[i-4]) == sc:
                continue  # not exclusive

            if confirm_candles == 0:
                # No confirmation — predict reversal, check if 1h closes opposite
                h = df1h.loc[ts]
                win = bool(h["close"] < h["open"]) if sc else bool(h["close"] > h["open"])
                sigs += 1
                if win: wins += 1

            elif confirm_candles == 1:
                c1 = bool(df15["green"].iloc[i])
                if c1 == sc:
                    continue  # no confirmation
                h = df1h.loc[ts]
                hour_green = bool(h["close"] > h["open"])
                sigs += 1
                if hour_green == c1: wins += 1

            elif confirm_candles == 2:
                c1 = bool(df15["green"].iloc[i])
                if c1 == sc:
                    continue
                # need 2nd candle also same as c1
                if i + 1 >= len(df15):
                    continue
                c2 = bool(df15["green"].iloc[i+1])
                if c2 != c1:
                    continue  # 2nd candle reversed — no signal
                h = df1h.loc[ts]
                hour_green = bool(h["close"] > h["open"])
                sigs += 1
                if hour_green == c1: wins += 1

        wr = wins/sigs*100 if sigs else 0
        ev = wr/100*(1/0.52-1) - (1-wr/100)
        print(f"  {label:<30} {sigs:>7,} {wr:>5.1f}% {sigs/months:>5.1f} {ev*100:>+7.1f}%")

    # ─────────────────────────────────────────────
    # БЛОК 2: все стрики 2-7, с 1-й 15m подтверждением
    # ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  БЛОК 2: Стрики 2-7 эксклюзивные + 1-я 15m подтверждение (BTC)")
    print("="*70)
    print(f"  {'Стрик':<8} {'Сделок':>7} {'Win%':>6} {'/мес':>6} {'EV@52¢':>8} {'Зелёный':>9} {'Красный':>9}")
    print(f"  {'-'*60}")

    for streak_len in range(2, 8):
        sigs, wins = 0, 0
        g_sigs, g_wins = 0, 0
        r_sigs, r_wins = 0, 0

        for i in range(streak_len+1, len(df15)-1):
            ts = df15.index[i]
            if ts.minute != 0: continue
            if ts not in df1h.index: continue

            streak = df15["green"].iloc[i-streak_len:i].values
            if not (all(streak) or not any(streak)): continue
            sc = bool(streak[0])
            if bool(df15["green"].iloc[i-streak_len-1]) == sc: continue

            c1 = bool(df15["green"].iloc[i])
            if c1 == sc: continue

            h = df1h.loc[ts]
            hour_green = bool(h["close"] > h["open"])
            win = (hour_green == c1)

            sigs += 1
            if win: wins += 1
            if sc:  # green streak (reversal = bearish)
                g_sigs += 1
                if win: g_wins += 1
            else:   # red streak (reversal = bullish)
                r_sigs += 1
                if win: r_wins += 1

        wr = wins/sigs*100 if sigs else 0
        ev = wr/100*(1/0.52-1) - (1-wr/100)
        gwr = g_wins/g_sigs*100 if g_sigs else 0
        rwr = r_wins/r_sigs*100 if r_sigs else 0
        print(f"  streak={streak_len} {sigs:>7,} {wr:>5.1f}% {sigs/months:>5.1f} {ev*100:>+7.1f}%  {gwr:>7.1f}%  {rwr:>7.1f}%")

    # ─────────────────────────────────────────────
    # БЛОК 3: "Чистый стрик" — стрик целиком в предыдущем часе
    # (ровно 4 свечи = 1h, и все одного цвета)
    # ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  БЛОК 3: 'Чистый' стрик — ровно 4×15m = 1 полный час до открытия")
    print("="*70)
    print(f"  {'Условие':<35} {'Сделок':>7} {'Win%':>6} {'/мес':>6} {'EV@52¢':>8}")
    print(f"  {'-'*60}")

    # Clean streak: exactly 4 candles of prev hour all same color
    for require_full_hour in [True, False]:
        sigs, wins = 0, 0
        for i in range(5, len(df15)-1):
            ts = df15.index[i]
            if ts.minute != 0: continue
            if ts not in df1h.index: continue

            # Last 4 candles = exactly the previous hour (all 4 quarters)
            streak4 = df15["green"].iloc[i-4:i].values
            if not (all(streak4) or not any(streak4)): continue
            sc = bool(streak4[0])

            if require_full_hour:
                # Extra check: candle at i-5 must be from previous-previous hour (different minute)
                ts_m5 = df15.index[i-5]
                if ts_m5.minute != 45:  # should be :45 of prev-prev hour
                    continue

            # 1st candle of new hour confirms reversal
            c1 = bool(df15["green"].iloc[i])
            if c1 == sc: continue

            h = df1h.loc[ts]
            hour_green = bool(h["close"] > h["open"])
            win = (hour_green == c1)
            sigs += 1
            if win: wins += 1

        wr = wins/sigs*100 if sigs else 0
        ev = wr/100*(1/0.52-1) - (1-wr/100)
        label = "Ровно 4×15m предыдущего часа" if require_full_hour else "4 свечи любые (не эксклюзив.)"
        print(f"  {label:<35} {sigs:>7,} {wr:>5.1f}% {sigs/months:>5.1f} {ev*100:>+7.1f}%")

    # ─────────────────────────────────────────────
    # БЛОК 4: По сессиям (стрик=3, 1-я 15m подтверждение)
    # ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  БЛОК 4: По сессиям UTC (BTC, стрик=3 эксклюзивный + 1-я 15m)")
    print("="*70)
    print(f"  {'Сессия':<12} {'Часы UTC':>10} {'Сделок':>7} {'Win%':>6} {'/мес':>6} {'EV@52¢':>8}")
    print(f"  {'-'*52}")

    all_sigs = []
    for i in range(4, len(df15)-1):
        ts = df15.index[i]
        if ts.minute != 0: continue
        if ts not in df1h.index: continue
        streak = df15["green"].iloc[i-3:i].values
        if not (all(streak) or not any(streak)): continue
        sc = bool(streak[0])
        if bool(df15["green"].iloc[i-4]) == sc: continue
        c1 = bool(df15["green"].iloc[i])
        if c1 == sc: continue
        h = df1h.loc[ts]
        hour_green = bool(h["close"] > h["open"])
        win = (hour_green == c1)
        all_sigs.append({"ts": ts, "hour_utc": ts.hour, "win": win})

    df_sigs = pd.DataFrame(all_sigs)
    for sess, (h_start, h_end) in SESSIONS.items():
        sub = df_sigs[(df_sigs["hour_utc"] >= h_start) & (df_sigs["hour_utc"] < h_end)]
        wr = sub["win"].mean()*100 if len(sub) else 0
        ev = wr/100*(1/0.52-1) - (1-wr/100)
        print(f"  {sess:<12} {f'{h_start:02d}:00-{h_end:02d}:00':>10} {len(sub):>7,} {wr:>5.1f}% {len(sub)/months:>5.1f} {ev*100:>+7.1f}%")

    total = len(df_sigs)
    wr_total = df_sigs["win"].mean()*100
    ev_total = wr_total/100*(1/0.52-1) - (1-wr_total/100)
    print(f"  {'Все часы':<12} {'00:00-24:00':>10} {total:>7,} {wr_total:>5.1f}% {total/months:>5.1f} {ev_total*100:>+7.1f}%")

    # ─────────────────────────────────────────────
    # БЛОК 5: Сводная таблица — все пары, стрик=3, +1-я 15m
    # ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  БЛОК 5: Все пары, стрик=3 эксклюзивный + 1-я 15m подтверждение")
    print("="*70)
    print(f"  {'Пара':<8} {'Сделок':>7} {'Win%':>6} {'/мес':>6} {'EV@52¢':>8} {'Зелёный':>9} {'Красный':>9}")
    print(f"  {'-'*60}")

    for pair in PAIRS:
        symbol = pair.split("/")[0]
        d15 = get_ohlcv(pair, "15m", since=SINCE)
        d1h = get_ohlcv(pair, "1h", since=SINCE)
        d15["green"] = d15["close"] > d15["open"]
        m = (d15.index[-1] - d15.index[0]).days / 30.44
        sigs, wins, g_s, g_w, r_s, r_w = 0,0,0,0,0,0

        for i in range(4, len(d15)-1):
            ts = d15.index[i]
            if ts.minute != 0: continue
            if ts not in d1h.index: continue
            streak = d15["green"].iloc[i-3:i].values
            if not (all(streak) or not any(streak)): continue
            sc = bool(streak[0])
            if bool(d15["green"].iloc[i-4]) == sc: continue
            c1 = bool(d15["green"].iloc[i])
            if c1 == sc: continue
            h = d1h.loc[ts]
            hour_green = bool(h["close"] > h["open"])
            win = (hour_green == c1)
            sigs += 1
            if win: wins += 1
            if sc: g_s += 1; g_w += win
            else:  r_s += 1; r_w += win

        wr = wins/sigs*100 if sigs else 0
        ev = wr/100*(1/0.52-1) - (1-wr/100)
        gwr = g_w/g_s*100 if g_s else 0
        rwr = r_w/r_s*100 if r_s else 0
        print(f"  {symbol:<8} {sigs:>7,} {wr:>5.1f}% {sigs/m:>5.1f} {ev*100:>+7.1f}%  {gwr:>7.1f}%  {rwr:>7.1f}%")


if __name__ == "__main__":
    analyze_btc()
    print("\n  Готово.")
