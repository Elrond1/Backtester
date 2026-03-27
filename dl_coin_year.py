"""
Download one year of one TF for given symbol. Same pattern as dl_year.py.
Usage: python dl_coin_year.py ETH/USDT 5m 2021
"""
import sys
import time
from datetime import datetime, timezone
from backtester.data.downloader import BinanceVisionDownloader
from backtester.data.cache import DataCache

symbol = sys.argv[1]   # e.g. "ETH/USDT"
tf     = sys.argv[2]   # e.g. "5m"
year   = int(sys.argv[3])

start = datetime(year, 1, 1, tzinfo=timezone.utc)
end   = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)

# Retry connecting to DuckDB — previous process may still be releasing lock
for attempt in range(10):
    try:
        cache = DataCache()
        break
    except Exception as e:
        if attempt < 9:
            time.sleep(3)
        else:
            print(f"{symbol} {tf} {year}: could not open DB after retries: {e}")
            sys.exit(1)

# Skip if already cached
cached_min, cached_max = cache.get_ohlcv_range("binance", symbol, tf)
if cached_min is not None:
    import pandas as pd
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    if cached_min <= ts_start and cached_max >= ts_end:
        cache.close()
        print(f"{symbol} {tf} {year}: cached, skip")
        sys.exit(0)

cache.close()

# Download first, then open DB to save (minimize lock time)
dl = BinanceVisionDownloader()
df = dl.fetch_klines(symbol, tf, start, end)

if df.empty:
    print(f"{symbol} {tf} {year}: no data")
    sys.exit(0)

# Retry saving
for attempt in range(10):
    try:
        cache = DataCache()
        cache.save_ohlcv("binance", symbol, tf, df)
        cache._conn.execute("CHECKPOINT")
        cache.close()
        print(f"{symbol} {tf} {year}: {len(df)} rows saved")
        sys.exit(0)
    except Exception as e:
        if attempt < 9:
            time.sleep(3)
        else:
            print(f"{symbol} {tf} {year}: save failed: {e}")
            sys.exit(1)
