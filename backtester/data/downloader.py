"""
Data downloaders.

A) BinanceVisionDownloader — bulk historical data from data.binance.vision
   No API keys, no rate limits. Covers spot klines and aggTrades since 2017.

B) CCXTDownloader — any exchange via ccxt (recent data, non-Binance exchanges).
"""

import io
import zipfile
from datetime import datetime, timezone, timedelta
from typing import Optional

import ccxt
import pandas as pd
import requests
from tqdm import tqdm

_VISION_BASE = "https://data.binance.vision/data/spot"

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

_AGGTRADE_COLS = [
    "agg_trade_id", "price", "qty", "first_trade_id",
    "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match",
]


def _vision_url(data_type: str, symbol: str, interval: str,
                year: int, month: int, day: Optional[int] = None) -> str:
    sym = symbol.replace("/", "").upper()
    if day is not None:
        period = f"{year:04d}-{month:02d}-{day:02d}"
        freq = "daily"
    else:
        period = f"{year:04d}-{month:02d}"
        freq = "monthly"
    return f"{_VISION_BASE}/{freq}/{data_type}/{sym}/{interval}/{sym}-{interval}-{period}.zip"


def _download_zip(url: str) -> Optional[bytes]:
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def _zip_to_df(content: bytes, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return pd.read_csv(f, header=None, names=columns)


class BinanceVisionDownloader:
    """Downloads klines and aggTrades from data.binance.vision."""

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Download OHLCV klines for symbol between start and end (UTC).
        Returns DataFrame with DatetimeIndex and columns:
        open, high, low, close, volume, quote_volume, trades.
        """
        frames = []
        current = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
        end_month = datetime(end.year, end.month, 1, tzinfo=timezone.utc)

        months = []
        while current <= end_month:
            months.append((current.year, current.month))
            current += timedelta(days=32)
            current = current.replace(day=1)

        sym_display = symbol.replace("/", "").upper()
        for year, month in tqdm(months, desc=f"Downloading {sym_display} {interval}"):
            url = _vision_url("klines", symbol, interval, year, month)
            content = _download_zip(url)
            if content is not None:
                df = _zip_to_df(content, _KLINE_COLS)
                frames.append(df)
            else:
                # Monthly file not available — fall back to daily files
                month_start = datetime(year, month, 1, tzinfo=timezone.utc)
                next_month = (month_start + timedelta(days=32)).replace(day=1)
                day = month_start
                while day < next_month:
                    day_url = _vision_url("klines", symbol, interval, day.year, day.month, day.day)
                    day_content = _download_zip(day_url)
                    if day_content is not None:
                        frames.append(_zip_to_df(day_content, _KLINE_COLS))
                    day += timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)
        return self._process_klines(raw, start, end)

    def fetch_aggtrades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Download aggregated trades. Returns DataFrame with columns:
        timestamp (UTC index), price, qty, is_buyer_maker.
        """
        frames = []
        sym_display = symbol.replace("/", "").upper()

        current = start
        while current <= end:
            url = _vision_url("aggTrades", symbol, "aggTrades",
                              current.year, current.month, current.day)
            content = _download_zip(url)
            if content is not None:
                df = _zip_to_df(content, _AGGTRADE_COLS)
                frames.append(df)
            current += timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
        raw = raw.set_index("timestamp").sort_index()
        raw = raw.loc[start:end]

        for col in ("price", "qty"):
            raw[col] = pd.to_numeric(raw[col])

        return raw[["price", "qty", "is_buyer_maker"]]

    @staticmethod
    def _process_klines(raw: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        # Binance changed timestamp format: pre-2026 = milliseconds (13 digits),
        # 2026+ = microseconds (16 digits). Convert per-row to handle mixed batches.
        ts = raw["open_time"].astype(float)
        ms_mask = ts <= 1e15  # ms values ≈ 1.7e12, us values ≈ 1.7e15
        seconds = ts.copy()
        seconds[ms_mask]  = ts[ms_mask]  / 1_000       # ms → seconds
        seconds[~ms_mask] = ts[~ms_mask] / 1_000_000   # us → seconds
        raw["open_time"] = pd.to_datetime(seconds, unit="s", utc=True)
        raw = raw.set_index("open_time").sort_index()

        for col in ("open", "high", "low", "close", "volume", "quote_volume"):
            raw[col] = pd.to_numeric(raw[col])
        raw["trades"] = pd.to_numeric(raw["trades"], downcast="integer")

        # Normalize tz to pandas UTC to avoid "same UTC offset" errors
        start_ts = pd.Timestamp(start).tz_convert("UTC")
        end_ts = pd.Timestamp(end).tz_convert("UTC")
        raw = raw.loc[start_ts:end_ts]
        return raw[["open", "high", "low", "close", "volume", "quote_volume", "trades"]]


class CCXTDownloader:
    """Downloads OHLCV via ccxt (any exchange, recent or non-Binance data)."""

    def __init__(self, exchange_id: str = "binance"):
        cls = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = cls({"enableRateLimit": True})

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_rows = []

        pbar = tqdm(desc=f"CCXT {symbol} {timeframe}")
        while True:
            rows = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=1000
            )
            if not rows:
                break
            all_rows.extend(rows)
            last_ts = rows[-1][0]
            pbar.update(len(rows))
            if last_ts >= end_ms or len(rows) < 1000:
                break
            since_ms = last_ts + 1
        pbar.close()

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_rows, columns=["open_time", "open", "high", "low", "close", "volume"]
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time").sort_index()
        df = df.loc[start:end]
        return df
