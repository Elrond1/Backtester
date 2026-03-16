"""
Vectorized technical indicators built on pandas.
All functions accept pd.Series and return pd.Series.
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0–100)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD indicator.

    Returns
    -------
    (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns
    -------
    (upper, middle, lower)
    """
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.

    Returns
    -------
    (%K, %D)
    """
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (resets each day)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    date = df.index.normalize()
    cumtp = tp_vol.groupby(date).cumsum()
    cumvol = df["volume"].groupby(date).cumsum()
    return cumtp / cumvol


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator.

    Returns
    -------
    direction : pd.Series — 1 = bullish, -1 = bearish
    line      : pd.Series — Supertrend line value (lower band or upper band)
    """
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)

    basic_upper = hl2 + multiplier * atr_val
    basic_lower = hl2 - multiplier * atr_val

    n = len(df)
    close = df["close"].values
    bu    = basic_upper.values
    bl    = basic_lower.values

    final_upper = bu.copy()
    final_lower = bl.copy()
    direction   = np.ones(n)   # start bullish
    line        = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(bu[i]) or np.isnan(bl[i]):
            # ATR warmup period — carry direction, leave bands NaN
            direction[i] = direction[i - 1]
            continue

        # Upper band only moves down (tightens), or resets when price breaks above
        if np.isnan(final_upper[i - 1]):
            final_upper[i] = bu[i]
        elif bu[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = bu[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band only moves up (tightens), or resets when price breaks below
        if np.isnan(final_lower[i - 1]):
            final_lower[i] = bl[i]
        elif bl[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = bl[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: flip when price crosses the active band
        prev_dir = direction[i - 1]
        if prev_dir == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        elif prev_dir == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        else:
            direction[i] = prev_dir

        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return (
        pd.Series(direction, index=df.index),
        pd.Series(line,      index=df.index),
    )
