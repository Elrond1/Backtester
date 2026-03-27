"""
Download all OHLCV timeframes + 1S bars for a given symbol.
Usage: python dl_coin.py ETH/USDT 2017
"""
import sys
from datetime import datetime, timezone
from backtester.data.downloader import BinanceVisionDownloader
from backtester.data.cache import DataCache
from backtester.data.manager import _fetch_1s_incremental, _1s_parquet_path

symbol = sys.argv[1]          # e.g. "ETH/USDT"
start_year = int(sys.argv[2]) # e.g. 2017
end_year = 2026

TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m", "5m"]
TICK_START_YEAR = 2020  # 1S bars available on Binance from 2020

dl = BinanceVisionDownloader()
cache = DataCache()

# --- OHLCV ---
for tf in TIMEFRAMES:
    for year in range(start_year, end_year + 1):
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end   = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)
        df = dl.fetch_klines(symbol, tf, start, end)
        if not df.empty:
            cache.save_ohlcv("binance", symbol, tf, df)
            cache._conn.execute("CHECKPOINT")
            print(f"{tf} {year}: {len(df)} rows saved")
        else:
            print(f"{tf} {year}: no data")

cache.close()

# --- 1S bars ---
print("\nStarting 1S bars download...")
from datetime import timedelta

for year in range(TICK_START_YEAR, end_year + 1):
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    _fetch_1s_incremental(None, symbol, start, end)
    print(f"1S {year}: done")

print("All done.")
