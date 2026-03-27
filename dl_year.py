import sys
from datetime import datetime, timezone
from backtester.data.downloader import BinanceVisionDownloader
from backtester.data.cache import DataCache

tf, year = sys.argv[1], int(sys.argv[2])
start = datetime(year, 1, 1, tzinfo=timezone.utc)
end   = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)

dl = BinanceVisionDownloader()
cache = DataCache()
df = dl.fetch_klines('BTC/USDT', tf, start, end)
cache.save_ohlcv('binance', 'BTC/USDT', tf, df)
cache._conn.execute("CHECKPOINT")
cache.close()
print(f"{tf} {year}: {len(df)} rows saved")
