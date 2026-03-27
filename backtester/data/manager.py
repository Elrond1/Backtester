"""
High-level data access functions.
Automatically picks data.binance.vision for Binance and ccxt for others.
Caches everything in DuckDB — incremental updates on repeated calls.

1-second bars are stored as Parquet files (one per month) in
~/.backtester/ticks/<SYMBOL>/ to avoid DuckDB memory issues with 190M+ rows.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from backtester.data.cache import DataCache
from backtester.data.downloader import BinanceVisionDownloader, CCXTDownloader

_cache: "DataCache | None" = None
_vision = BinanceVisionDownloader()


def _get_cache() -> "DataCache":
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache

# Directory for 1s Parquet files
_TICKS_DIR = Path.home() / ".backtester" / "ticks"


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
    cache = _get_cache() if db_path is None else DataCache(db_path)

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
            if timeframe == "1s":
                # Download and cache month-by-month to avoid loading
                # 190M+ rows into memory at once.
                _fetch_1s_incremental(cache, symbol, dl_start, dl_end)
            else:
                df = _vision.fetch_klines(symbol, timeframe, dl_start, dl_end)
                cache.save_ohlcv(exchange, symbol, timeframe, df)
        else:
            downloader = CCXTDownloader(exchange)
            df = downloader.fetch_klines(symbol, timeframe, dl_start, dl_end)
            cache.save_ohlcv(exchange, symbol, timeframe, df)

    return cache.load_ohlcv(exchange, symbol, timeframe, start, end)


def _1s_parquet_path(symbol: str, year: int, month: int) -> Path:
    """Path to the Parquet file for one month of 1s bars."""
    safe = symbol.replace("/", "").upper()
    return _TICKS_DIR / safe / f"{year:04d}-{month:02d}.parquet"


def _d1_parquet_path(symbol: str) -> Path:
    """Path to the Parquet file for D1 bars."""
    safe = symbol.replace("/", "").upper()
    return _TICKS_DIR / safe / "d1.parquet"


def get_d1_bars(
    symbol: str,
    since: Union[str, datetime] = "2019-01-01",
    until: Union[str, datetime, None] = None,
    exchange: str = "binance",
) -> pd.DataFrame:
    """
    Download and cache D1 bars as Parquet (no DuckDB lock needed).
    Multiple processes can read simultaneously.
    """
    start = _parse_dt(since)
    end   = _parse_dt(until) if until else datetime.now(tz=timezone.utc)

    path  = _d1_parquet_path(symbol)
    needs_download = True

    if path.exists():
        cached = pd.read_parquet(path)
        if not cached.empty:
            cached_min = cached.index.min()
            cached_max = cached.index.max()
            if pd.Timestamp(start) >= cached_min and pd.Timestamp(end) <= cached_max + timedelta(days=2):
                needs_download = False

    if needs_download:
        df = _vision.fetch_klines(symbol, "1d", start, end)
        if not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
        cached = df

    if cached.empty:
        return cached
    ts_start = pd.Timestamp(start).tz_localize("UTC") if start.tzinfo is None else pd.Timestamp(start)
    ts_end   = pd.Timestamp(end).tz_localize("UTC")   if end.tzinfo is None   else pd.Timestamp(end)
    return cached.loc[ts_start:ts_end]


def _fetch_1s_incremental(
    _cache_unused,
    symbol: str,
    start: datetime,
    end: datetime,
) -> None:
    """Download 1s klines one month at a time, save as Parquet files.

    Parquet avoids DuckDB memory issues with 190M+ row datasets.
    Files: ~/.backtester/ticks/<SYMBOL>/YYYY-MM.parquet (~20 MB each)
    """
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        next_month = (current + timedelta(days=32)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        # Use last second of month so fetch_klines doesn't spill into next month
        chunk_end = min(next_month - timedelta(seconds=1), end)

        path = _1s_parquet_path(symbol, current.year, current.month)
        if path.exists():
            current = next_month
            continue

        df = _vision.fetch_klines(symbol, "1s", current, chunk_end)
        if not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            print(f"      Saved {path.name}  ({len(df):,} rows)")

        current = next_month


def load_1s_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    """Load one month of 1s bars from Parquet cache. Returns empty DF if missing."""
    path = _1s_parquet_path(symbol, year, month)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_liquidations(
    symbol: str,
    since: Union[str, datetime] = "2023-01-01",
    until: Union[str, datetime, None] = None,
    timeframe: str = "1h",
    exchange: str = "binance",
    api_key: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download and cache hourly liquidation aggregates from Coinalyze.

    Requires FREE Coinalyze API key:
      1. Register at https://coinalyze.net
      2. Profile → API Key (free)
      3. export COINALYZE_API_KEY=your_key

    Parameters
    ----------
    symbol    : "BTC/USDT"
    timeframe : "1h", "4h", "12h", "1d"
    exchange  : "binance" (default), "bybit", "okx"

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns:
      liq_long, liq_short, liq_total (USD values)

    Note
    ----
    Free tier retains ~2000 most recent intraday points per symbol.
    Daily aggregates are preserved indefinitely.
    """
    from backtester.data.coinalyze import CoinalyzeDownloader

    cache = _get_cache() if db_path is None else DataCache(db_path)

    start = _parse_dt(since)
    end = _parse_dt(until) if until else datetime.now(tz=timezone.utc)

    cached_min, cached_max = cache.get_liq_range(exchange, symbol, timeframe)

    needs_download = (
        cached_min is None
        or start < cached_min
        or end > cached_max
    )

    if needs_download:
        dl = CoinalyzeDownloader(api_key=api_key)
        df = dl.fetch_liquidations(symbol, start, end, timeframe, exchange)
        cache.save_liq(exchange, symbol, timeframe, df)

    return cache.load_liq(exchange, symbol, timeframe, start, end)


def get_seconds(
    symbol: str,
    since: Union[str, datetime] = "2020-01-01",
    until: Union[str, datetime, None] = None,
    exchange: str = "binance",
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download and cache 1-second OHLCV bars from data.binance.vision.

    1-second bars give MT4 "Every Tick" precision for TP/SL simulation
    at a fraction of the storage cost of raw aggTrades (~15-30 MB/month
    vs ~2 GB/month for aggTrades).

    Use with run_tick_backtest() for maximum accuracy.

    Parameters
    ----------
    symbol   : "BTC/USDT"
    since    : start date — Binance 1s klines available from ~2020
    until    : end date (defaults to now)
    exchange : only "binance" is supported

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns:
    open, high, low, close, volume, quote_volume, trades
    """
    return get_ohlcv(symbol, "1s", since=since, until=until,
                     exchange=exchange, db_path=db_path)


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

    cache = _get_cache() if db_path is None else DataCache(db_path)

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
