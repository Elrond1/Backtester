"""
Download OHLCV + 1S bars for multiple coins sequentially.
Processes year-by-year to avoid DuckDB memory crashes.
Usage: python dl_all_coins.py
"""
from datetime import datetime, timezone
from backtester.data.downloader import BinanceVisionDownloader
from backtester.data.cache import DataCache
from backtester.data.manager import _fetch_1s_incremental

COINS = [
    ("ETH/USDT", 2017),
    ("SOL/USDT", 2020),
    ("BNB/USDT", 2017),
    ("XRP/USDT", 2018),
    ("DOGE/USDT", 2019),
]

TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m", "5m"]
TICK_START_YEAR = 2020
END_YEAR = 2026

dl = BinanceVisionDownloader()

# --- OHLCV year by year ---
print("=" * 50)
print("STEP 1: OHLCV for all coins (year by year)")
print("=" * 50)

for symbol, start_year in COINS:
    print(f"\n--- {symbol} ---")
    for tf in TIMEFRAMES:
        for year in range(start_year, END_YEAR + 1):
            cache = DataCache()
            # Check if already cached
            cached_min, cached_max = cache.get_ohlcv_range("binance", symbol, tf)
            year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
            year_end   = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)
            if (cached_min is not None
                    and cached_min <= year_start
                    and cached_max >= year_end):
                cache.close()
                print(f"  {tf} {year}: cached, skipping")
                continue
            df = dl.fetch_klines(symbol, tf, year_start, year_end)
            if not df.empty:
                cache.save_ohlcv("binance", symbol, tf, df)
                cache._conn.execute("CHECKPOINT")
                print(f"  {tf} {year}: {len(df)} rows saved")
            else:
                print(f"  {tf} {year}: no data")
            cache.close()

# --- 1S bars month by month ---
print("\n" + "=" * 50)
print("STEP 2: 1S bars for all coins")
print("=" * 50)

for symbol, _ in COINS:
    print(f"\n--- {symbol} 1S bars ---")
    since = datetime(TICK_START_YEAR, 1, 1, tzinfo=timezone.utc)
    end   = datetime(END_YEAR, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    _fetch_1s_incremental(None, symbol, since, end)
    print(f"  done")

print("\nAll coins downloaded successfully.")
