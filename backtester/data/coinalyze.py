"""
Coinalyze API — aggregated liquidation history.

Free tier: unlimited API key, 40 req/min, ~2000 most recent data points per symbol.
Register: https://coinalyze.net → Profile → API Key

Symbol format: {BASE}{QUOTE}_PERP.{EXCHANGE}
  Exchange codes: A=Binance, D=Bybit, G=OKX, Q=Bitget, I=BitMEX

Examples:
  BTCUSDT_PERP.A   — BTC/USDT perpetual on Binance
  ETHUSDT_PERP.A   — ETH/USDT perpetual on Binance
  BTCUSDT_PERP.D   — BTC/USDT perpetual on Bybit

Set key via: export COINALYZE_API_KEY=your_key
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Load .env from project root if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
except ImportError:
    pass

_BASE = "https://api.coinalyze.net/v1"

_INTERVAL_MAP = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1hour",
    "2h":  "2hour",
    "4h":  "4hour",
    "6h":  "6hour",
    "12h": "12hour",
    "1d":  "1day",
}

# Exchange suffix in Coinalyze symbol format
_EXCHANGE_SUFFIX = {
    "binance":  "A",
    "bybit":    "D",
    "okx":      "G",
    "bitget":   "Q",
    "bitmex":   "I",
}


def symbol_to_coinalyze(symbol: str, exchange: str = "binance") -> str:
    """Convert 'BTC/USDT' + 'binance' → 'BTCUSDT_PERP.A'"""
    base = symbol.replace("/", "").replace("-", "").upper()
    suffix = _EXCHANGE_SUFFIX.get(exchange.lower(), "A")
    return f"{base}_PERP.{suffix}"


class CoinalyzeDownloader:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("COINALYZE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Coinalyze API key required.\n"
                "Register free at https://coinalyze.net → Profile → API Key\n"
                "Then: export COINALYZE_API_KEY=your_key"
            )
        self._session = requests.Session()
        self._session.headers.update({"api_key": self.api_key})

    def fetch_liquidations(
        self,
        symbol: str,
        since: datetime,
        until: datetime,
        timeframe: str = "1h",
        exchange: str = "binance",
    ) -> pd.DataFrame:
        """
        Fetch aggregated liquidation history.

        Parameters
        ----------
        symbol    : "BTC/USDT"
        since     : start datetime (UTC)
        until     : end datetime (UTC)
        timeframe : "1h", "4h", "1d", etc.
        exchange  : "binance", "bybit", "okx", etc.

        Returns
        -------
        pd.DataFrame with DatetimeIndex (UTC) and columns:
          liq_long   — liquidated longs in USD
          liq_short  — liquidated shorts in USD
          liq_total  — sum
        """
        sym = symbol_to_coinalyze(symbol, exchange)
        interval = _INTERVAL_MAP.get(timeframe, timeframe)

        from_ts = int(since.timestamp())
        to_ts = int(until.timestamp())

        # Coinalyze limits to ~2000 points per request — paginate if needed
        all_rows = []
        chunk_size = 2000 * self._interval_seconds(interval)
        current = from_ts

        while current < to_ts:
            chunk_end = min(current + chunk_size, to_ts)
            r = self._session.get(
                f"{_BASE}/liquidation-history",
                params={
                    "symbols": sym,
                    "interval": interval,
                    "from": current,
                    "to": chunk_end,
                },
                timeout=30,
            )

            if r.status_code == 429:
                time.sleep(2)
                continue

            r.raise_for_status()
            data = r.json()

            if not data or not isinstance(data, list) or not data[0].get("history"):
                break

            rows = data[0]["history"]
            all_rows.extend(rows)
            current = chunk_end + 1

        if not all_rows:
            return pd.DataFrame(
                columns=["liq_long", "liq_short", "liq_total"],
                index=pd.DatetimeIndex([], tz="UTC", name="ts"),
            )

        df = pd.DataFrame(all_rows)
        # Coinalyze response fields: t=timestamp(seconds), l=long_liq, s=short_liq
        df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.set_index("ts").sort_index()
        df = df.rename(columns={"l": "liq_long", "s": "liq_short"})
        df["liq_total"] = df["liq_long"] + df["liq_short"]

        # NOTE: Coinalyze returns values in millions of USD.
        # liq_long=50 means $50M of long liquidations in that period.
        return df[["liq_long", "liq_short", "liq_total"]]

    @staticmethod
    def _interval_seconds(interval: str) -> int:
        units = {"min": 60, "hour": 3600, "day": 86400}
        for unit, secs in units.items():
            if unit in interval:
                n = int(interval.replace(unit, "") or "1")
                return n * secs
        return 3600
