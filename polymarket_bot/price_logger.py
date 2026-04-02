"""
Логгер реальных цен Polymarket в момент HH:15 UTC.

Запускает раз в час в HH:15 и записывает цены YES/NO токенов
для BTC/ETH/SOL в CSV файл price_log.csv.

Запуск на сервере (параллельно с ботом):
    python -m polymarket_bot.price_logger

Через неделю в price_log.csv будет ~168 строк с реальными ценами.
"""

import asyncio
import csv
import os
import logging
from datetime import datetime, timezone, timedelta

import aiohttp

from polymarket_bot.markets import MarketFinder, build_slug
from polymarket_bot import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

CSV_FILE = "price_log.csv"
COINS    = ["bitcoin", "ethereum", "solana"]


async def get_token_price(token_id: str) -> float:
    """Получает лучшую цену ask для токена через CLOB."""
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
        asks = data.get("asks", [])
        if asks:
            return float(asks[0]["price"])
    except Exception as e:
        log.warning(f"Ошибка получения цены {token_id[:12]}: {e}")
    return 0.0


async def log_prices():
    """Один цикл: находит рынки и логирует цены."""
    now = datetime.now(timezone.utc)
    hour_start = now.replace(minute=0, second=0, microsecond=0)

    finder = MarketFinder()
    rows = []

    for coin in COINS:
        market = await finder.find(coin, hour_start)
        if market is None:
            log.warning(f"[{coin.upper()}] Рынок не найден, пропускаем")
            rows.append({
                "ts": now.strftime("%Y-%m-%d %H:%M UTC"),
                "coin": coin,
                "yes_price": "",
                "no_price": "",
                "cheaper_side": "",
                "market_found": "NO",
            })
            continue

        yes_price = await get_token_price(market.yes_token_id)
        no_price  = await get_token_price(market.no_token_id)

        cheaper = "YES" if (yes_price > 0 and no_price > 0 and yes_price <= no_price) else "NO"
        below_limit = yes_price < config.MAX_ENTRY_PRICE or no_price < config.MAX_ENTRY_PRICE

        rows.append({
            "ts": now.strftime("%Y-%m-%d %H:%M UTC"),
            "coin": coin,
            "yes_price": f"{yes_price:.2f}" if yes_price else "",
            "no_price":  f"{no_price:.2f}"  if no_price  else "",
            "cheaper_side": cheaper,
            "market_found": "YES",
        })

        flag = "✅ входим" if below_limit else "⚠️  пропускаем (дорого)"
        log.info(
            f"[{coin.upper():8s}] YES={yes_price:.2f}¢  NO={no_price:.2f}¢  "
            f"дешевле={cheaper}  {flag}"
        )

    # Записываем в CSV
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "coin", "yes_price", "no_price", "cheaper_side", "market_found"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    log.info(f"Записано {len(rows)} строк в {CSV_FILE}")


async def main():
    log.info("Price logger запущен. Замеры каждый час в HH:15 UTC.")
    log.info(f"Данные пишутся в: {os.path.abspath(CSV_FILE)}")

    while True:
        now = datetime.now(timezone.utc)

        # Считаем сколько ждать до следующего HH:15
        next_15 = now.replace(minute=15, second=0, microsecond=0)
        if now.minute >= 15:
            next_15 += timedelta(hours=1)

        wait_sec = (next_15 - now).total_seconds()
        log.info(f"Следующий замер: {next_15.strftime('%H:%M UTC')} (через {wait_sec/60:.1f} мин)")

        await asyncio.sleep(wait_sec)

        log.info(f"=== Замер цен {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")
        await log_prices()


if __name__ == "__main__":
    asyncio.run(main())
