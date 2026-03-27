#!/bin/bash
# Download OHLCV + 1S bars for all coins, year by year (same pattern as BTC)
# Note: DuckDB exits with SIGSEGV on cleanup (known bug) but data is CHECKPOINT'd safely
cd "$(dirname "$0")"

run() {
    python dl_coin_year.py "$@" || true
}

echo "===== STEP 1: OHLCV ====="

# ETH from 2017
for tf in 1d 4h 1h 30m 15m 5m; do
  for year in 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
    run ETH/USDT $tf $year
  done
done

# SOL from 2020
for tf in 1d 4h 1h 30m 15m 5m; do
  for year in 2020 2021 2022 2023 2024 2025 2026; do
    run SOL/USDT $tf $year
  done
done

# BNB from 2017
for tf in 1d 4h 1h 30m 15m 5m; do
  for year in 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
    run BNB/USDT $tf $year
  done
done

# XRP from 2018
for tf in 1d 4h 1h 30m 15m 5m; do
  for year in 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
    run XRP/USDT $tf $year
  done
done

# DOGE from 2019
for tf in 1d 4h 1h 30m 15m 5m; do
  for year in 2019 2020 2021 2022 2023 2024 2025 2026; do
    run DOGE/USDT $tf $year
  done
done

echo ""
echo "===== STEP 2: 1S bars ====="

python - <<'EOF' || true
from datetime import datetime, timezone
from backtester.data.manager import _fetch_1s_incremental

COINS = ["ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

for symbol in COINS:
    print(f"\n--- {symbol} 1S ---")
    _fetch_1s_incremental(None, symbol, START, END)
    print(f"  done")

print("\nAll done.")
EOF

echo "===== FINISHED ====="
