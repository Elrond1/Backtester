"""
Поиск Polymarket рынков для 1h крипто контрактов.

Slug строится по текущей дате/часу в ET:
  bitcoin-up-or-down-march-31-2026-3pm-et
  ethereum-up-or-down-march-31-2026-3pm-et
  solana-up-or-down-march-31-2026-3pm-et

Кэш обновляется каждый час.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

from . import config

log = logging.getLogger(__name__)

# EDT = UTC-4 (14 Mar – 7 Nov), EST = UTC-5 (остальное время)
def _et_offset(dt: datetime) -> timedelta:
    year = dt.year
    # DST начинается второе воскресенье марта, заканчивается первое воскресенье ноября
    # Упрощённо: март 14 – ноябрь 7
    dst_start = datetime(year, 3, 14, 2, tzinfo=timezone.utc)
    dst_end   = datetime(year, 11, 7,  2, tzinfo=timezone.utc)
    if dst_start <= dt < dst_end:
        return timedelta(hours=-4)   # EDT
    return timedelta(hours=-5)       # EST


def build_slug(coin: str, hour_utc: datetime) -> str:
    """
    Строит slug для Polymarket hourly рынка.

    coin:     "bitcoin" | "ethereum" | "solana"
    hour_utc: HH:00 UTC — начало часа (signal.hour_start)

    Пример: bitcoin-up-or-down-march-31-2026-3pm-et
    """
    et_time = hour_utc + _et_offset(hour_utc)
    hour24  = et_time.hour
    ampm    = "am" if hour24 < 12 else "pm"
    hour12  = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    month = et_time.strftime("%B").lower()   # march, april ...
    day   = et_time.day
    year  = et_time.year
    return f"{coin}-up-or-down-{month}-{day}-{year}-{hour12}{ampm}-et"


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
    Ищет активные Polymarket рынки по автоматически сгенерированному slug.
    Результаты кэшируются по (slug) ключу.
    """

    def __init__(self):
        self._cache: dict[str, PolyMarket] = {}

    async def find(
        self,
        coin_keyword: str,      # "bitcoin" | "ethereum" | "solana"
        target_hour: datetime,  # HH:00 UTC — начало часа
    ) -> Optional[PolyMarket]:
        """
        Возвращает рынок для данного часа по slug.
        """
        slug = build_slug(coin_keyword, target_hour)

        if slug in self._cache:
            m = self._cache[slug]
            if m.active:
                return m

        market = await self._search(slug)
        if market:
            self._cache[slug] = market
            log.info(
                f"[{coin_keyword.upper()}] Рынок найден: {market.question} "
                f"(slug={slug})"
            )
        else:
            log.error(
                f"[{coin_keyword.upper()}] Market not found – SKIP (slug: {slug})"
            )

        return market

    async def _search(self, slug: str) -> Optional[PolyMarket]:
        """
        Запрашивает Gamma API по slug события.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{config.GAMMA_HOST}/events",
                    params={"slug": slug},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        log.error(f"Gamma API error: {resp.status} for slug={slug}")
                        return None
                    data = await resp.json()
        except Exception as e:
            log.error(f"Gamma API request failed: {e}")
            return None

        events = data if isinstance(data, list) else data.get("data", [])
        if not events:
            return None

        event   = events[0]
        markets = event.get("markets", [])
        if not markets:
            return None

        # Берём первый активный рынок из события
        best = next((m for m in markets if m.get("active")), None)
        if not best:
            return None

        # Парсим токены из clobTokenIds + outcomes
        token_ids = best.get("clobTokenIds")
        if isinstance(token_ids, str):
            import json
            token_ids = json.loads(token_ids)

        outcomes = best.get("outcomes", [])
        if isinstance(outcomes, str):
            import json
            outcomes = json.loads(outcomes)

        if not token_ids or len(token_ids) < 2:
            log.warning(f"Нет clobTokenIds для slug={slug}: {token_ids}")
            return None

        # outcomes: ["Up", "Down"] → Up=YES, Down=NO
        # Если outcomes нет — первый токен YES, второй NO
        if outcomes and len(outcomes) >= 2:
            up_idx   = next((i for i, o in enumerate(outcomes) if o.lower() == "up"), 0)
            down_idx = next((i for i, o in enumerate(outcomes) if o.lower() == "down"), 1)
        else:
            up_idx, down_idx = 0, 1

        yes_token_id = token_ids[up_idx]
        no_token_id  = token_ids[down_idx]

        # Парсим endDate
        end_str = best.get("endDate", "")
        try:
            end_str = end_str.replace("Z", "+00:00")
            end_dt  = datetime.fromisoformat(end_str).astimezone(timezone.utc)
        except (ValueError, AttributeError):
            end_dt = datetime.now(timezone.utc) + timedelta(hours=1)

        return PolyMarket(
            condition_id=best.get("conditionId", ""),
            question=best.get("question", ""),
            yes_token_id=str(yes_token_id),
            no_token_id=str(no_token_id),
            end_date=end_dt,
            active=True,
        )
