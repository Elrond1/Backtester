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


def halftrend(
    df: pd.DataFrame,
    amplitude: int   = 2,
    channel_dev: float = 2.0,
    atr_period: int  = 100,
) -> tuple[pd.Series, pd.Series]:
    """
    HalfTrend indicator (Alex Orekhov / everget).

    Returns
    -------
    direction : pd.Series — 1 = uptrend, -1 = downtrend
    line      : pd.Series — HalfTrend line value
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(df)

    atr_val = atr(df, atr_period).values
    dev_arr = channel_dev * atr_val / 2

    high_roll = df["high"].rolling(amplitude, min_periods=1).max().values
    low_roll  = df["low"].rolling(amplitude,  min_periods=1).min().values
    high_ma   = df["high"].rolling(amplitude, min_periods=1).mean().values
    low_ma    = df["low"].rolling(amplitude,  min_periods=1).mean().values

    trend       = np.zeros(n, dtype=int)   # 0 = up, 1 = down
    next_trend  = np.zeros(n, dtype=int)
    max_low     = low[0]
    min_high    = high[0]
    up          = np.full(n, np.nan)
    down        = np.full(n, np.nan)
    line_arr    = np.full(n, np.nan)
    direction   = np.ones(n)               # 1=up, -1=down for output

    for i in range(1, n):
        nt = next_trend[i - 1]
        tr = trend[i - 1]

        if nt == 1:
            max_low = max(low_roll[i], max_low)
            if high_ma[i] < max_low and close[i] < (low[i - 1] if i > 0 else low[i]):
                tr = 1
                nt = 0
                min_high = high_roll[i]
        else:
            min_high = min(high_roll[i], min_high)
            if low_ma[i] > min_high and close[i] > (high[i - 1] if i > 0 else high[i]):
                tr = 0
                nt = 1
                max_low = low_roll[i]

        trend[i]      = tr
        next_trend[i] = nt

        if tr == 0:  # uptrend
            prev_up = up[i - 1] if not np.isnan(up[i - 1]) else low_roll[i]
            up[i]   = max(low_roll[i], prev_up)
            line_arr[i]  = up[i]
            direction[i] = 1
        else:        # downtrend
            prev_dn  = down[i - 1] if not np.isnan(down[i - 1]) else high_roll[i]
            down[i]  = min(high_roll[i], prev_dn)
            line_arr[i]  = down[i]
            direction[i] = -1

    return (
        pd.Series(direction, index=df.index),
        pd.Series(line_arr,  index=df.index),
    )


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).

    Returns ADX values (0-100). High values (>25) indicate a strong trend.
    Low values (<20) indicate a ranging/sideways market.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    dm_plus  = (high - prev_high).clip(lower=0)
    dm_minus = (prev_low - low).clip(lower=0)

    # Where DM+ <= DM- set DM+ to 0, and vice versa
    mask = dm_plus >= dm_minus
    dm_plus  = dm_plus.where(mask, 0.0)
    dm_minus = dm_minus.where(~mask, 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr_s    = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val


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


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """
    Hull Moving Average.

    HMA(n) = WMA(2·WMA(n/2) − WMA(n), sqrt(n))
    Reduces lag while keeping smoothness.
    """
    half  = max(period // 2, 1)
    sqrt_ = max(int(np.sqrt(period)), 1)
    raw   = 2 * wma(series, half) - wma(series, period)
    return wma(raw, sqrt_)


def kama(
    series: pd.Series,
    period: int = 21,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """
    Kaufman's Adaptive Moving Average (KAMA).

    Adapts to market noise: fast in trends, slow in sideways markets.

    Parameters
    ----------
    period : Efficiency Ratio look-back (default 21)
    fast   : Fast EMA period (default 2)
    slow   : Slow EMA period (default 30)
    """
    close = series.values
    n = len(close)

    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    kama_arr = np.full(n, np.nan)
    kama_arr[period - 1] = close[period - 1]

    # Precompute absolute 1-bar changes for rolling volatility
    abs_diff = np.abs(np.diff(close, prepend=close[0]))

    for i in range(period, n):
        direction = abs(close[i] - close[i - period])
        volatility = abs_diff[i - period + 1 : i + 1].sum()
        er = direction / volatility if volatility != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_arr[i] = kama_arr[i - 1] + sc * (close[i] - kama_arr[i - 1])

    return pd.Series(kama_arr, index=series.index)


def squeeze_momentum(
    df: pd.DataFrame,
    length: int = 20,
    mult_bb: float = 2.0,
    mult_kc: float = 1.5,
    length_mom: int = 12,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Squeeze Momentum Indicator (LazyBear port).

    Detects when Bollinger Bands compress inside Keltner Channel ("squeeze"),
    and measures directional momentum via linear regression.

    Returns
    -------
    squeeze_on : pd.Series[bool]  — True while BB inside KC (black dots)
    histogram  : pd.Series[float] — momentum (linreg of delta)
    color      : pd.Series[str]   — 'lime'|'green'|'red'|'maroon'
                 lime   = positive & rising  (light green)
                 green  = positive & falling (dark green)
                 red    = negative & falling (dark red)
                 maroon = negative & rising  (light red)
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # Bollinger Bands
    bb_mid  = close.rolling(length, min_periods=length).mean()
    bb_std  = close.rolling(length, min_periods=length).std()
    bb_upper = bb_mid + mult_bb * bb_std
    bb_lower = bb_mid - mult_bb * bb_std

    # Keltner Channel (EMA center + ATR via EMA-smoothed TR, matching LazyBear)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    kc_mid   = close.ewm(span=length, adjust=False).mean()
    range_ma = tr.ewm(span=length, adjust=False).mean()
    kc_upper = kc_mid + mult_kc * range_ma
    kc_lower = kc_mid - mult_kc * range_ma

    # Squeeze: BB inside KC
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Momentum delta: close minus midpoint average
    highest_high = high.rolling(length, min_periods=length).max()
    lowest_low   = low.rolling(length, min_periods=length).min()
    mid_price    = (highest_high + lowest_low) / 2.0
    delta        = close - (mid_price + bb_mid) / 2.0

    # Linear regression over length_mom bars (value at current bar)
    def _linreg(x: np.ndarray) -> float:
        n_ = len(x)
        xs = np.arange(n_, dtype=float)
        a, b = np.polyfit(xs, x, 1)
        return a * (n_ - 1) + b

    histogram = delta.rolling(length_mom, min_periods=length_mom).apply(
        _linreg, raw=True
    )

    # Color labels
    hist_prev = histogram.shift(1)
    color = pd.Series("none", index=df.index)
    color[(histogram > 0) & (histogram >= hist_prev)] = "lime"
    color[(histogram > 0) & (histogram <  hist_prev)] = "green"
    color[(histogram < 0) & (histogram <= hist_prev)] = "red"
    color[(histogram < 0) & (histogram >  hist_prev)] = "maroon"

    return squeeze_on, histogram, color


def sr_weekly_window(
    df: pd.DataFrame,
    weeks_near: int = 2,
    weeks_far: int = 4,
) -> tuple[pd.Series, pd.Series]:
    """
    Support and resistance based on a 2–4 week historical window (daily timeframe).

    Looks at bars from `weeks_far` weeks ago back to `weeks_near` weeks ago:
      - resistance = highest high in that range
      - support    = lowest  low  in that range

    Designed for daily OHLCV data (1 bar = 1 calendar day, including weekends for
    crypto). For traditional markets pass weeks_near / weeks_far in trading weeks
    and set `trading_days=True` to use 5-day weeks instead of 7.

    Parameters
    ----------
    df         : daily OHLCV DataFrame with DatetimeIndex
    weeks_near : near edge of the lookback window in weeks (default 2)
    weeks_far  : far  edge of the lookback window in weeks (default 4)

    Returns
    -------
    support    : pd.Series — lowest  low  in the [weeks_far … weeks_near] window
    resistance : pd.Series — highest high in the [weeks_far … weeks_near] window
    """
    days_per_week = 7  # crypto trades 24/7; change to 5 for equities
    near_bars   = weeks_near * days_per_week          # 14
    window_bars = (weeks_far - weeks_near) * days_per_week  # 14

    # shift(near_bars) moves the series so bar[i] = original bar[i - near_bars]
    # rolling(window_bars).max/min then covers bars [i-near_bars .. i-near_bars-window_bars]
    # = [i-14 .. i-28] = 2-4 weeks ago — zero lookahead
    resistance = (
        df["high"]
        .shift(near_bars)
        .rolling(window=window_bars, min_periods=1)
        .max()
    )
    support = (
        df["low"]
        .shift(near_bars)
        .rolling(window=window_bars, min_periods=1)
        .min()
    )
    return support, resistance


def swing_highs_lows(
    df: pd.DataFrame,
    strength: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    Detect swing highs and swing lows (fractal pivots).

    A swing high at bar i: high[i] is the maximum over [i-strength .. i+strength].
    A swing low  at bar i: low[i]  is the minimum over [i-strength .. i+strength].

    Confirmation requires `strength` bars of lookahead, so the signal is placed
    `strength` bars after the actual pivot — no lookahead bias in backtesting.

    Returns
    -------
    swing_highs : pd.Series — NaN except at confirmed swing high bars (value = high)
    swing_lows  : pd.Series — NaN except at confirmed swing low  bars (value = low)
    """
    high = df["high"]
    low  = df["low"]
    w    = 2 * strength + 1

    roll_high_max = high.rolling(window=w, center=True, min_periods=w).max()
    roll_low_min  = low.rolling(window=w,  center=True, min_periods=w).min()

    is_swing_high = high == roll_high_max
    is_swing_low  = low  == roll_low_min

    # Shift forward by `strength` so confirmation lands at the current bar
    swing_highs = high.where(is_swing_high).shift(strength)
    swing_lows  = low.where(is_swing_low).shift(strength)

    return swing_highs, swing_lows


def support_resistance(
    df: pd.DataFrame,
    strength: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    Nearest confirmed support and resistance levels.

    Forward-fills each confirmed swing low (support) and swing high (resistance),
    so every bar carries the most recently confirmed S/R level with no lookahead.

    Returns
    -------
    support    : pd.Series — most recent swing low price (resistance below price)
    resistance : pd.Series — most recent swing high price (resistance above price)
    """
    swing_highs, swing_lows = swing_highs_lows(df, strength=strength)
    support    = swing_lows.ffill()
    resistance = swing_highs.ffill()
    return support, resistance


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow.

    CMF = Sum(MFV, period) / Sum(Volume, period)
    where MFV = ((Close - Low) - (High - Close)) / (High - Low) * Volume

    Range: -1 to +1. Positive = buying pressure, negative = selling pressure.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    vol   = df["volume"]

    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * vol

    return (
        mfv.rolling(window=period, min_periods=period).sum()
        / vol.rolling(window=period, min_periods=period).sum()
    )
