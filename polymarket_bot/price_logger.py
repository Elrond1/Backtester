"""
Логгер реальных цен Polymarket в моменты HH:05 и HH:15 UTC.

HH:05 — через 5 минут после начала часа (для сравнения с HH:15)
HH:15 — основной замер (момент сигнала в боте)

Цель: понять насколько цена дешевле при входе в HH:05 vs HH:15.

Запуск на сервере (параллельно с ботом):
    python -m polymarket_bot.price_logger

Через неделю в price_log.csv будет ~672 строк (4 замера в час).
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

COIN_TO_SYMBOL = {
    "bitcoin":  "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana":   "SOLUSDT",
}


async def get_candle_meta(coin: str, hour_start: datetime) -> dict:
    """
    Тянет последние 15m свечи с Binance и вычисляет:
    - streak_len: длина стрика перед началом часа
    - streak_dir: up / down
    - first_15m_dir: up / down / none (если свеча ещё не закрылась)
    - signal_type: reversal / continuation / none
    """
    symbol = COIN_TO_SYMBOL.get(coin)
    if not symbol:
        return {}

    url = "https://api.binance.com/api/v3/klines"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, params={
                "symbol": symbol,
                "interval": "15m",
                "limit": 20,
            }, timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
    except Exception as e:
        log.warning(f"[{coin.upper()}] Ошибка получения свечей: {e}")
        return {}

    # Парсим свечи
    candles = []
    for k in data:
        open_time = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
        candles.append({
            "open_time": open_time,
            "open":  float(k[1]),
            "close": float(k[4]),
            "closed": k[6] < datetime.now(timezone.utc).timestamp() * 1000,
        })

    # Ищем первую 15m свечу текущего часа (открылась в HH:00)
    first_15m = None
    pre_candles = []
    for c in candles:
        if c["open_time"] == hour_start:
            first_15m = c
        elif c["open_time"] < hour_start:
            pre_candles.append(c)

    if not pre_candles:
        return {}

    # Считаем стрик перед часом (идём с конца pre_candles назад)
    pre_candles.sort(key=lambda x: x["open_time"])
    streak_dir = pre_candles[-1]["close"] > pre_candles[-1]["open"]  # True=up
    streak_len = 0
    for c in reversed(pre_candles):
        c_up = c["close"] > c["open"]
        if c_up == streak_dir:
            streak_len += 1
        else:
            break

    # Первая 15m свеча
    first_15m_dir = None
    signal_type = "none"
    if first_15m and first_15m["closed"]:
        first_15m_up = first_15m["close"] > first_15m["open"]
        first_15m_dir = "up" if first_15m_up else "down"
        if streak_len >= 3:
            if first_15m_up != streak_dir:
                signal_type = "reversal"
            else:
                signal_type = "continuation"

    return {
        "streak_len":    streak_len,
        "streak_dir":    "up" if streak_dir else "down",
        "first_15m_dir": first_15m_dir or "",
        "signal_type":   signal_type,
    }


async def get_token_price(token_id: str) -> float:
    """Получает лучшую цену ask для токена через CLOB."""
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
        asks = data.get("asks", [])
        if asks:
            prices = [float(a["price"]) for a in asks if "price" in a]
            if prices:
                return min(prices)
    except Exception as e:
        log.warning(f"Ошибка получения цены {token_id[:12]}: {e}")
    return 0.0


async def log_prices(trigger_minute: int):
    """Один цикл: находит рынки и логирует цены."""
    now = datetime.now(timezone.utc)
    hour_start = now.replace(minute=0, second=0, microsecond=0)

    finder = MarketFinder()
    rows = []

    for coin in COINS:
        # Метаданные свечей (стрик + первая 15m) — для всех замеров
        meta = await get_candle_meta(coin, hour_start)

        market = await finder.find(coin, hour_start)
        if market is None:
            log.warning(f"[{coin.upper()}] Рынок не найден, пропускаем")
            rows.append({
                "ts": now.strftime("%Y-%m-%d %H:%M UTC"),
                "minute": trigger_minute,
                "coin": coin,
                "yes_price": "",
                "no_price": "",
                "cheaper_side": "",
                "market_found": "NO",
                "streak_len":    meta.get("streak_len", ""),
                "streak_dir":    meta.get("streak_dir", ""),
                "first_15m_dir": meta.get("first_15m_dir", ""),
                "signal_type":   meta.get("signal_type", ""),
            })
            continue

        yes_price = await get_token_price(market.yes_token_id)
        no_price  = await get_token_price(market.no_token_id)

        cheaper = "YES" if (yes_price > 0 and no_price > 0 and yes_price <= no_price) else "NO"
        below_limit = yes_price < config.MAX_ENTRY_PRICE or no_price < config.MAX_ENTRY_PRICE

        rows.append({
            "ts": now.strftime("%Y-%m-%d %H:%M UTC"),
            "minute": trigger_minute,
            "coin": coin,
            "yes_price": f"{yes_price:.2f}" if yes_price else "",
            "no_price":  f"{no_price:.2f}"  if no_price  else "",
            "cheaper_side": cheaper,
            "market_found": "YES",
            "streak_len":    meta.get("streak_len", ""),
            "streak_dir":    meta.get("streak_dir", ""),
            "first_15m_dir": meta.get("first_15m_dir", ""),
            "signal_type":   meta.get("signal_type", ""),
        })

        signal_info = f"  стрик={meta.get('streak_len','')}×{meta.get('streak_dir','')}  1м={meta.get('first_15m_dir','')}  тип={meta.get('signal_type','')}"
        flag = "✅ входим" if below_limit else "⚠️  пропускаем (дорого)"
        log.info(
            f"[{coin.upper():8s}] YES={yes_price:.2f}¢  NO={no_price:.2f}¢  "
            f"дешевле={cheaper}  {flag}{signal_info}"
        )

    # Записываем в CSV
    fieldnames = ["ts", "minute", "coin", "yes_price", "no_price", "cheaper_side",
                  "market_found", "streak_len", "streak_dir", "first_15m_dir", "signal_type"]
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    log.info(f"Записано {len(rows)} строк в {CSV_FILE}")


async def main():
    log.info("Price logger запущен. Замеры в HH:05, HH:15, HH:20, HH:30 UTC.")
    log.info(f"Данные пишутся в: {os.path.abspath(CSV_FILE)}")

    while True:
        now = datetime.now(timezone.utc)

        # Следующие триггеры: HH:05, HH:15, HH:20, HH:30
        # HH:05 — ранний вход вместо HH:15 (сравнение цен)
        # HH:15 — основной сигнал B/C/D/D_C
        # HH:20 — ранний вход вместо HH:30 (сравнение цен)
        # HH:30 — сигнал DC (двойное подтверждение)
        candidates = []
        for minute in [5, 15, 20, 30]:
            t = now.replace(minute=minute, second=0, microsecond=0)
            if t <= now:
                t += timedelta(hours=1)
            candidates.append((t, minute))

        # Берём ближайший
        next_trigger, trigger_minute = min(candidates, key=lambda x: x[0])

        wait_sec = (next_trigger - now).total_seconds()
        log.info(f"Следующий замер: {next_trigger.strftime('%H:%M UTC')} (HH:{trigger_minute:02d}, через {wait_sec/60:.1f} мин)")

        await asyncio.sleep(wait_sec)

        log.info(f"=== Замер цен {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} (HH:{trigger_minute:02d}) ===")
        await log_prices(trigger_minute)


if __name__ == "__main__":
    asyncio.run(main())
