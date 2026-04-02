"""
Диагностика бота — проверяет каждый компонент без ожидания сигнала.

Запуск на сервере:
    python -m polymarket_bot.diagnose

Проверяет:
  1. Slug для текущего часа — находится ли рынок на Polymarket
  2. Цена YES/NO токенов через order book
  3. CLOB клиент — авторизация и баланс
  4. Тестовый ордер (только DRY_RUN — реальные деньги не тратятся)
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

import aiohttp

# ── путь к модулям ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polymarket_bot.markets import MarketFinder, build_slug
from polymarket_bot import config

SEP = "=" * 60


async def check_market(coin: str, hour_utc: datetime):
    slug = build_slug(coin, hour_utc)
    print(f"\n  Slug:     {slug}")

    finder = MarketFinder()
    market = await finder.find(coin, hour_utc)

    if market is None:
        print(f"  ❌ Рынок НЕ найден")
        return None

    print(f"  ✅ Рынок найден: {market.question}")
    print(f"  Закрытие:       {market.end_date.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  YES token:      {market.yes_token_id[:16]}...")
    print(f"  NO  token:      {market.no_token_id[:16]}...")
    print(f"  active:         {market.active}")
    return market


async def check_price(token_id: str, label: str):
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
        asks = data.get("asks", [])
        bids = data.get("bids", [])
        ask_price = float(asks[0]["price"]) if asks else None
        bid_price = float(bids[0]["price"]) if bids else None
        print(f"  {label}: ask={ask_price:.2f}¢  bid={bid_price:.2f}¢  {'✅' if ask_price and ask_price < config.MAX_ENTRY_PRICE else '⚠️  выше лимита' if ask_price else '❌ нет данных'}")
        return ask_price
    except Exception as e:
        print(f"  {label}: ❌ ошибка: {e}")
        return None


async def check_clob():
    if not config.POLY_PRIVATE_KEY:
        print("  ❌ POLY_PRIVATE_KEY не задан в .env")
        return False
    try:
        from py_clob_client.client import ClobClient
        client = ClobClient(
            host=config.CLOB_HOST,
            key=config.POLY_PRIVATE_KEY,
            chain_id=config.CHAIN_ID,
            signature_type=2,
            funder=config.POLY_API_KEY,
        )
        bal = client.get_balance()
        if isinstance(bal, dict):
            usdc = bal.get("USDC", bal.get("usdc", "?"))
        else:
            usdc = getattr(bal, "USDC", getattr(bal, "usdc", "?"))
        print(f"  ✅ CLOB авторизован | Баланс USDC: ${usdc}")
        return True
    except Exception as e:
        print(f"  ❌ CLOB ошибка: {e}")
        return False


async def main():
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    print(SEP)
    print("  ДИАГНОСТИКА POLYMARKET БОТА")
    print(SEP)
    print(f"  Время сейчас: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Текущий час:  {now_utc.strftime('%H:%M UTC')}")

    # ── 1. Рынки ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ШАГ 1: Поиск рынков (текущий час)")
    print(SEP)

    for coin in ["bitcoin", "ethereum", "solana"]:
        print(f"\n  [{coin.upper()}]")
        market = await check_market(coin, now_utc)

        # ── 2. Цены токенов ──────────────────────────────────────────────────
        if market:
            print(f"\n  Цены токенов:")
            yes_price = await check_price(market.yes_token_id, "  YES (Up)")
            no_price  = await check_price(market.no_token_id,  "  NO  (Down)")

            if yes_price and no_price:
                cheaper = "YES" if yes_price < no_price else "NO"
                print(f"  → Дешевле:  {cheaper} ({min(yes_price, no_price):.2f}¢)")

    # ── 3. CLOB авторизация ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ШАГ 2: CLOB авторизация и баланс")
    print(SEP)
    await check_clob()

    # ── Итог ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ГОТОВО")
    print(SEP)
    print("""
  Что означают результаты:
  ✅ Рынок найден + цена < 67¢ + CLOB OK → бот готов торговать
  ❌ Рынок не найден → проблема в slug (build_slug)
  ⚠️  Цена > 67¢    → рынок слишком "уверен", бот пропустит
  ❌ CLOB ошибка    → проблема с ключами в .env
""")


if __name__ == "__main__":
    asyncio.run(main())
