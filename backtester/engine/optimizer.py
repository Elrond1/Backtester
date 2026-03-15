"""
Parameter optimizer for trading strategies.

grid_search   — exhaustive search over all parameter combinations
random_search — random sampling (faster for large parameter spaces)
"""

import itertools
import random
from typing import Type

import pandas as pd
from tqdm import tqdm

from backtester.engine.backtester import run_backtest
from backtester.engine.metrics import compute_metrics
from backtester.strategy.base import Strategy


def grid_search(
    strategy_class: Type[Strategy],
    param_grid: dict[str, list],
    df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    aux: dict | None = None,
    symbol: str = "",
    timeframe: str = "",
) -> pd.DataFrame:
    """
    Exhaustive grid search over all parameter combinations.

    Parameters
    ----------
    strategy_class : Strategy subclass (not an instance)
    param_grid     : {"fast": [10, 20, 30], "slow": [50, 100, 200]}
    df             : OHLCV DataFrame
    metric         : Column to sort results by (descending)
    ...

    Returns
    -------
    pd.DataFrame sorted by metric descending, with one row per combination.

    Example
    -------
    results = grid_search(
        MACrossover,
        {"fast": [10, 20, 50], "slow": [50, 100, 200]},
        df,
        metric="sharpe_ratio",
    )
    print(results.head(10))
    """
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    rows = []
    for combo in tqdm(combinations, desc="Grid search"):
        params = dict(zip(keys, combo))
        try:
            strategy = strategy_class(**params)
            result = run_backtest(
                df, strategy,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                symbol=symbol,
                timeframe=timeframe,
                aux=aux,
            )
            row = {**params, **result.metrics}
            rows.append(row)
        except Exception as e:
            rows.append({**params, "error": str(e)})

    results = pd.DataFrame(rows)
    if metric in results.columns:
        results = results.sort_values(metric, ascending=False).reset_index(drop=True)
    return results


def random_search(
    strategy_class: Type[Strategy],
    param_distributions: dict[str, list | range],
    n_iter: int = 50,
    df: pd.DataFrame = None,
    metric: str = "sharpe_ratio",
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    aux: dict | None = None,
    symbol: str = "",
    timeframe: str = "",
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Random parameter search — faster than grid search for large spaces.

    Parameters
    ----------
    param_distributions : {"fast": range(5, 100), "slow": [50, 100, 150, 200]}
    n_iter              : Number of random combinations to try
    seed                : Random seed for reproducibility

    Returns
    -------
    pd.DataFrame sorted by metric descending.
    """
    if seed is not None:
        random.seed(seed)

    rows = []
    for _ in tqdm(range(n_iter), desc="Random search"):
        params = {k: random.choice(list(v)) for k, v in param_distributions.items()}
        try:
            strategy = strategy_class(**params)
            result = run_backtest(
                df, strategy,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                symbol=symbol,
                timeframe=timeframe,
                aux=aux,
            )
            row = {**params, **result.metrics}
            rows.append(row)
        except Exception as e:
            rows.append({**params, "error": str(e)})

    results = pd.DataFrame(rows)
    if metric in results.columns:
        results = results.sort_values(metric, ascending=False).reset_index(drop=True)
    return results
