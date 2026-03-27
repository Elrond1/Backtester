"""
Test: how often does a 5-candle streak (all green or all red)
get followed by an opposite candle?

BTC/USDT, Binance, 5m and 15m, from 2020-01-01
"""

import sys
sys.path.insert(0, "/Users/arturbagian/Python_proj/Backtester")

from backtester.data.manager import get_ohlcv


def analyze_streak(df, streak_len=5):
    close = df["close"].values
    open_ = df["open"].values

    # 1 = green (close > open), -1 = red (close < open), 0 = doji
    direction = []
    for i in range(len(close)):
        if close[i] > open_[i]:
            direction.append(1)
        elif close[i] < open_[i]:
            direction.append(-1)
        else:
            direction.append(0)

    green_after_red_streak = 0   # after N red -> next is green
    red_after_green_streak = 0   # after N green -> next is red
    total_red_streaks = 0
    total_green_streaks = 0

    for i in range(streak_len, len(direction) - 1):
        window = direction[i - streak_len:i]
        next_candle = direction[i]

        if all(d == -1 for d in window):
            total_red_streaks += 1
            if next_candle == 1:
                green_after_red_streak += 1

        if all(d == 1 for d in window):
            total_green_streaks += 1
            if next_candle == -1:
                red_after_green_streak += 1

    return {
        "total_red_streaks": total_red_streaks,
        "green_after_red": green_after_red_streak,
        "pct_green_after_red": green_after_red_streak / total_red_streaks * 100 if total_red_streaks else 0,
        "total_green_streaks": total_green_streaks,
        "red_after_green": red_after_green_streak,
        "pct_red_after_green": red_after_green_streak / total_green_streaks * 100 if total_green_streaks else 0,
    }


def run(timeframe):
    print(f"\n{'='*50}")
    print(f"  BTC/USDT  |  {timeframe}  |  2020-01-01 -> сейчас")
    print(f"{'='*50}")
    df = get_ohlcv("BTC/USDT", timeframe, since="2020-01-01")
    print(f"  Загружено свечей: {len(df):,}")

    result = analyze_streak(df, streak_len=5)

    print(f"\n  После 5 КРАСНЫХ подряд:")
    print(f"    Всего случаев : {result['total_red_streaks']:,}")
    print(f"    Следующая ЗЕЛЕНАЯ: {result['green_after_red']:,}  ({result['pct_green_after_red']:.1f}%)")
    print(f"    Следующая НЕ зеленая: {result['total_red_streaks'] - result['green_after_red']:,}  ({100 - result['pct_green_after_red']:.1f}%)")

    print(f"\n  После 5 ЗЕЛЕНЫХ подряд:")
    print(f"    Всего случаев : {result['total_green_streaks']:,}")
    print(f"    Следующая КРАСНАЯ: {result['red_after_green']:,}  ({result['pct_red_after_green']:.1f}%)")
    print(f"    Следующая НЕ красная: {result['total_green_streaks'] - result['red_after_green']:,}  ({100 - result['pct_red_after_green']:.1f}%)")


if __name__ == "__main__":
    run("5m")
    run("15m")
