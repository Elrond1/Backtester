"""
DuckDB-based local cache for OHLCV and aggTrades data.

Single file at ~/.backtester/data.duckdb — portable, fast, SQL-queryable.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

_DEFAULT_DB = Path.home() / ".backtester" / "data.duckdb"


def _table_name(exchange: str, symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "").replace("-", "").lower()
    return f"{exchange.lower()}__{safe}__{timeframe.lower()}"


def _aggtrades_table(exchange: str, symbol: str) -> str:
    safe = symbol.replace("/", "").replace("-", "").lower()
    return f"{exchange.lower()}__{safe}__aggtrades"


class DataCache:
    """Manages OHLCV and aggTrades storage in DuckDB."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))

    # ------------------------------------------------------------------ OHLCV

    def _ensure_ohlcv_table(self, table: str):
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                ts          TIMESTAMPTZ PRIMARY KEY,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      DOUBLE,
                quote_volume DOUBLE,
                trades      BIGINT
            )
        """)

    def get_ohlcv_range(
        self, exchange: str, symbol: str, timeframe: str
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Returns (min_ts, max_ts) of cached data, or (None, None) if empty."""
        table = _table_name(exchange, symbol, timeframe)
        tables = self._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table]
        ).fetchone()
        if not tables:
            return None, None
        row = self._conn.execute(
            f"SELECT MIN(ts), MAX(ts) FROM {table}"
        ).fetchone()
        if row is None or row[0] is None:
            return None, None
        return row[0], row[1]

    def load_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        table = _table_name(exchange, symbol, timeframe)
        tables = self._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table]
        ).fetchone()
        if not tables:
            return pd.DataFrame()
        df = self._conn.execute(
            f"SELECT * FROM {table} WHERE ts >= ? AND ts <= ? ORDER BY ts",
            [start, end]
        ).df()
        if df.empty:
            return df
        df = df.set_index("ts")
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "open_time"
        return df

    def save_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ):
        if df.empty:
            return
        table = _table_name(exchange, symbol, timeframe)
        self._ensure_ohlcv_table(table)

        tmp = df.copy().reset_index()
        tmp.rename(columns={"open_time": "ts"}, inplace=True)

        for col in ("quote_volume", "trades"):
            if col not in tmp.columns:
                tmp[col] = None

        self._conn.register("_tmp_df", tmp)
        self._conn.execute(f"""
            INSERT OR IGNORE INTO {table}
            SELECT ts, open, high, low, close, volume, quote_volume, trades
            FROM _tmp_df
        """)

    # -------------------------------------------------------------- AggTrades

    def _ensure_aggtrades_table(self, table: str):
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                ts              TIMESTAMPTZ PRIMARY KEY,
                price           DOUBLE,
                qty             DOUBLE,
                is_buyer_maker  BOOLEAN
            )
        """)

    def get_aggtrades_range(
        self, exchange: str, symbol: str
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        table = _aggtrades_table(exchange, symbol)
        tables = self._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table]
        ).fetchone()
        if not tables:
            return None, None
        row = self._conn.execute(
            f"SELECT MIN(ts), MAX(ts) FROM {table}"
        ).fetchone()
        if row is None or row[0] is None:
            return None, None
        return row[0], row[1]

    def load_aggtrades(
        self,
        exchange: str,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        table = _aggtrades_table(exchange, symbol)
        tables = self._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table]
        ).fetchone()
        if not tables:
            return pd.DataFrame()
        df = self._conn.execute(
            f"SELECT * FROM {table} WHERE ts >= ? AND ts <= ? ORDER BY ts",
            [start, end]
        ).df()
        if df.empty:
            return df
        df = df.set_index("ts")
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        return df

    def save_aggtrades(
        self,
        exchange: str,
        symbol: str,
        df: pd.DataFrame,
    ):
        if df.empty:
            return
        table = _aggtrades_table(exchange, symbol)
        self._ensure_aggtrades_table(table)

        tmp = df.copy().reset_index()
        tmp.rename(columns={"timestamp": "ts"}, inplace=True)

        self._conn.register("_tmp_at", tmp)
        self._conn.execute(f"""
            INSERT OR IGNORE INTO {table}
            SELECT ts, price, qty, is_buyer_maker FROM _tmp_at
        """)

    # ---------------------------------------------------------------- Utility

    def list_datasets(self) -> pd.DataFrame:
        """Returns a DataFrame listing all cached datasets with row counts and date ranges."""
        tables = self._conn.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
        rows = []
        for (t,) in tables:
            row = self._conn.execute(
                f"SELECT COUNT(*), MIN(ts), MAX(ts) FROM {t}"
            ).fetchone()
            rows.append({"table": t, "rows": row[0], "from": row[1], "to": row[2]})
        return pd.DataFrame(rows)

    def close(self):
        self._conn.close()
