"""
Поиск Polymarket рынков для 1h крипто контрактов.

Polymarket создаёт рынки вида:
  "Will Bitcoin (BTC) be higher at 15:00 UTC than at 14:00 UTC?"

Gamma API используется для поиска и кэширования market IDs.
Кэш обновляется каждый час.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

from . import config

log = logging.getLogger(__name__)


@dataclass
class PolyMarket:
    condition_id:  str
    question:      str
    yes_token_id:  str   # CLOB token для ставки YES (цена растёт)
    no_token_id:   str   # CLOB token для ставки NO (цена падает)
    end_date:      datetime
    active:        bool


class MarketFinder:
    """
    Ищет активные Polymarket рынки для заданных крипто пар и временного слота.
    Результаты кэшируются по (symbol, hour) ключу.
    """

    def __init__(self):
        self._cache: dict[tuple, PolyMarket] = {}

    async def find(
        self,
        symbol_keyword: str,    # "bitcoin", "dogecoin", "bnb"
        target_hour: datetime,  # HH:00 UTC — начало часа для контракта
    ) -> Optional[PolyMarket]:
        """
        Возвращает рынок который резолвится в target_hour + 1h (т.е. HH+1:00).
        """
        resolve_time = target_hour + timedelta(hours=1)
        cache_key = (symbol_keyword, resolve_time.strftime("%Y%m%d%H"))

        if cache_key in self._cache:
            m = self._cache[cache_key]
            if m.active:
                return m

        market = await self._search(symbol_keyword, resolve_time)
        if market:
            self._cache[cache_key] = market
            log.info(
                f"Found market for {symbol_keyword} @ {resolve_time.strftime('%H:%M UTC')}: "
                f"{market.question[:70]}"
            )
        else:
            log.warning(
                f"No market found for {symbol_keyword} @ {resolve_time.strftime('%H:%M UTC')}"
            )

        return market

    async def _search(
        self,
        keyword: str,
        resolve_time: datetime,
    ) -> Optional[PolyMarket]:
        """
        Ищет в Gamma API активные рынки по ключевому слову + времени резолюции.
        """
        resolve_iso = resolve_time.strftime("%Y-%m-%dT%H:%M:%S")   # напр. 2025-03-26T15:00:00

        params = {
            "active":       "true",
            "closed":       "false",
            "limit":        "50",
            "search":       keyword,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{config.GAMMA_HOST}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        log.error(f"Gamma API error: {resp.status}")
                        return None
                    data = await resp.json()
        except Exception as e:
            log.error(f"Gamma API request failed: {e}")
            return None

        markets = data if isinstance(data, list) else data.get("data", [])

        # Фильтруем по времени резолюции (с допуском ±5 минут)
        best: Optional[dict] = None
        best_delta = timedelta(days=999)

        for m in markets:
            # Пропускаем неактивные / без токенов
            if not m.get("active"):
                continue
            tokens = m.get("tokens", [])
            if len(tokens) < 2:
                continue

            end_str = m.get("endDate") or m.get("end_date_iso", "")
            if not end_str:
                continue

            try:
                # ISO string может быть с Z или +00:00
                end_str = end_str.replace("Z", "+00:00")
                end_dt = datetime.fromisoformat(end_str).astimezone(timezone.utc)
            except ValueError:
                continue

            delta = abs(end_dt - resolve_time)
            if delta < timedelta(minutes=5) and delta < best_delta:
                best = m
                best_delta = delta

        if not best:
            return None

        tokens = best["tokens"]
        yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
        no_token  = next((t for t in tokens if t.get("outcome", "").upper() == "NO"),  None)

        if not yes_token or not no_token:
            log.warning(f"Market has unexpected token structure: {tokens}")
            return None

        return PolyMarket(
            condition_id=best.get("conditionId", best.get("condition_id", "")),
            question=best.get("question", ""),
            yes_token_id=yes_token["token_id"],
            no_token_id=no_token["token_id"],
            end_date=resolve_time,
            active=True,
        )
