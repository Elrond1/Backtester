"""
High-level data access functions.
Automatically picks data.binance.vision for Binance and ccxt for others.
Caches everything in DuckDB — incremental updates on repeated calls.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from backtester.data.cache import DataCache
from backtester.data.downloader import BinanceVisionDownloader, CCXTDownloader

_cache = DataCache()
_vision = BinanceVisionDownloader()


def _parse_dt(dt: Union[str, datetime]) -> datetime:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_ohlcv(
    symbol: str,
    timeframe: str,
    since: Union[str, datetime] = "2023-01-01",
    until: Union[str, datetime, None] = None,
    exchange: str = "binance",
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download and cache OHLCV data.

    For Binance uses data.binance.vision (bulk, no rate limits).
    For other exchanges uses ccxt.

    Parameters
    ----------
    symbol    : "BTC/USDT"
    timeframe : "1m", "5m", "1h", "4h", "1d", etc.
    since     : start date (str ISO or datetime)
    until     : end date (str ISO or datetime), defaults to now
    exchange  : exchange id, default "binance"
    db_path   : custom path to DuckDB file (optional)

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns:
    open, high, low, close, volume [, quote_volume, trades]
    """
    cache = _cache if db_path is None else DataCache(db_path)

    start = _parse_dt(since)
    end = _parse_dt(until) if until else datetime.now(tz=timezone.utc)

    cached_min, cached_max = cache.get_ohlcv_range(exchange, symbol, timeframe)

    # Determine what ranges need downloading
    to_download: list[tuple[datetime, datetime]] = []

    if cached_min is None:
        to_download.append((start, end))
    else:
        if start < cached_min:
            to_download.append((start, cached_min))
        if end > cached_max:
            to_download.append((cached_max, end))

    for dl_start, dl_end in to_download:
        if exchange.lower() == "binance":
            df = _vision.fetch_klines(symbol, timeframe, dl_start, dl_end)
        else:
            downloader = CCXTDownloader(exchange)
            df = downloader.fetch_klines(symbol, timeframe, dl_start, dl_end)
        cache.save_ohlcv(exchange, symbol, timeframe, df)

    return cache.load_ohlcv(exchange, symbol, timeframe, start, end)


def get_aggtrades(
    symbol: str,
    since: Union[str, datetime] = "2024-01-01",
    until: Union[str, datetime, None] = None,
    exchange: str = "binance",
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download and cache Binance aggTrades (tick-level data).

    Returns pd.DataFrame with DatetimeIndex (UTC) and columns:
    price, qty, is_buyer_maker.

    Note: Only supported for Binance (data.binance.vision).
    aggTrades files are ~1 GB/month for BTC/USDT — use for short periods.
    """
    if exchange.lower() != "binance":
        raise ValueError("aggTrades download is only supported for Binance.")

    cache = _cache if db_path is None else DataCache(db_path)

    start = _parse_dt(since)
    end = _parse_dt(until) if until else datetime.now(tz=timezone.utc)

    cached_min, cached_max = cache.get_aggtrades_range(exchange, symbol)

    to_download: list[tuple[datetime, datetime]] = []

    if cached_min is None:
        to_download.append((start, end))
    else:
        if start < cached_min:
            to_download.append((start, cached_min))
        if end > cached_max:
            to_download.append((cached_max, end))

    for dl_start, dl_end in to_download:
        df = _vision.fetch_aggtrades(symbol, dl_start, dl_end)
        cache.save_aggtrades(exchange, symbol, df)

    return cache.load_aggtrades(exchange, symbol, start, end)
