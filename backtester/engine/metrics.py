"""
Performance metrics for backtesting results.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio (uses downside deviation)."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.25 = -25%)."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def max_drawdown_duration(equity: pd.Series) -> pd.Timedelta:
    """Duration of the longest drawdown period."""
    roll_max = equity.cummax()
    in_dd = equity < roll_max

    if not in_dd.any():
        return pd.Timedelta(0)

    dd_start = None
    max_dur = pd.Timedelta(0)
    for t, is_dd in in_dd.items():
        if is_dd and dd_start is None:
            dd_start = t
        elif not is_dd and dd_start is not None:
            dur = t - dd_start
            if dur > max_dur:
                max_dur = dur
            dd_start = None
    return max_dur


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio = CAGR / |max_drawdown|."""
    cagr = total_cagr(equity, periods_per_year)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)


def total_return(equity: pd.Series) -> float:
    """Total return as a fraction."""
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def total_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    if len(equity) < 2:
        return 0.0
    n_periods = len(equity)
    years = n_periods / periods_per_year
    tr = total_return(equity)
    if tr <= -1:
        return -1.0
    return float((1 + tr) ** (1 / years) - 1)


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of trades that were profitable."""
    if trades.empty:
        return 0.0
    return float((trades["pnl_pct"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profits / gross losses (absolute)."""
    if trades.empty:
        return 0.0
    gains = trades.loc[trades["pnl_pct"] > 0, "pnl_pct"].sum()
    losses = trades.loc[trades["pnl_pct"] < 0, "pnl_pct"].abs().sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def _infer_periods_per_year(returns: pd.Series) -> int:
    if len(returns) < 2:
        return 252
    freq = pd.infer_freq(returns.index)
    if freq is None:
        # Estimate from median bar duration
        deltas = returns.index.to_series().diff().dropna()
        median_sec = deltas.median().total_seconds()
        return max(1, int(365.25 * 24 * 3600 / median_sec))
    freq_map = {
        "T": 525_960, "min": 525_960,   # 1-minute
        "5T": 105_192, "5min": 105_192,
        "15T": 35_064, "15min": 35_064,
        "30T": 17_532, "30min": 17_532,
        "H": 8766, "h": 8766,           # 1-hour
        "4H": 2191, "4h": 2191,
        "D": 365, "B": 252,             # daily
        "W": 52, "M": 12,
    }
    for key, val in freq_map.items():
        if freq.startswith(key):
            return val
    return 252


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """Compute all standard metrics and return as ordered dict."""
    ppy = _infer_periods_per_year(returns)

    return {
        "total_return_pct":     round(total_return(equity) * 100, 2),
        "cagr_pct":             round(total_cagr(equity, ppy) * 100, 2),
        "sharpe_ratio":         round(sharpe_ratio(returns, periods_per_year=ppy), 3),
        "sortino_ratio":        round(sortino_ratio(returns, periods_per_year=ppy), 3),
        "max_drawdown_pct":     round(max_drawdown(equity) * 100, 2),
        "calmar_ratio":         round(calmar_ratio(returns, equity, ppy), 3),
        "win_rate_pct":         round(win_rate(trades) * 100, 2),
        "profit_factor":        round(profit_factor(trades), 3),
        "total_trades":         len(trades),
        "final_capital":        round(equity.iloc[-1] if not equity.empty else initial_capital, 2),
    }
