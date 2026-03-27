"""
Размещение ордеров на Polymarket через CLOB API.

Использует официальный py-clob-client.
При DRY_RUN=true — только логирует, не ставит реальные ордера.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from . import config
from .markets import PolyMarket
from .signals import Signal

log = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success:    bool
    order_id:   str = ""
    filled:     float = 0.0   # $
    avg_price:  float = 0.0   # ¢
    error:      str = ""


class Executor:
    def __init__(self):
        self._client = None
        if not config.DRY_RUN:
            self._init_client()

    def _init_client(self):
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.constants import POLYGON

            if config.POLY_API_KEY:
                # L2 auth (рекомендуется для программной торговли)
                self._client = ClobClient(
                    host=config.CLOB_HOST,
                    key=config.POLY_PRIVATE_KEY,
                    chain_id=config.CHAIN_ID,
                    signature_type=2,    # L2
                    funder=config.POLY_API_KEY,
                )
            else:
                # L1 auth (только приватный ключ)
                self._client = ClobClient(
                    host=config.CLOB_HOST,
                    key=config.POLY_PRIVATE_KEY,
                    chain_id=config.CHAIN_ID,
                )

            log.info("CLOB client initialized")
        except ImportError:
            raise RuntimeError(
                "py-clob-client не установлен. "
                "Запусти: pip install py-clob-client"
            )

    async def place(
        self,
        signal: Signal,
        market: PolyMarket,
        bet_usd: float,
    ) -> OrderResult:
        """
        Размещает Market Order на Polymarket.

        direction == "YES" → покупаем YES токен (ставим на рост)
        direction == "NO"  → покупаем NO токен  (ставим на падение)
        """
        token_id = (
            market.yes_token_id
            if signal.direction == "YES"
            else market.no_token_id
        )

        log.info(
            f"[{signal.symbol.upper()}] ORDER "
            f"{signal.sig_type} {signal.direction} "
            f"${bet_usd:.0f} | market: {market.question[:60]} | "
            f"token: {token_id[:12]}..."
        )

        if config.DRY_RUN:
            log.info(
                f"[DRY RUN] Would place ${bet_usd:.0f} {signal.direction} "
                f"for {signal.symbol.upper()} {signal.sig_type}"
            )
            return OrderResult(success=True, order_id="DRY_RUN", avg_price=0.52)

        try:
            return await self._place_real(token_id, bet_usd)
        except Exception as e:
            log.error(f"Order placement failed: {e}")
            return OrderResult(success=False, error=str(e))

    async def _place_real(self, token_id: str, size_usd: float) -> OrderResult:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        import asyncio

        def _sync():
            # py-clob-client — синхронный; запускаем в executor
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=size_usd,    # $ сумма (не количество токенов)
            )
            signed = self._client.create_market_order(order_args)
            resp = self._client.post_order(signed, OrderType.FOK)  # Fill-or-Kill
            return resp

        loop = __import__("asyncio").get_event_loop()
        resp = await loop.run_in_executor(None, _sync)

        if resp and resp.get("success"):
            return OrderResult(
                success=True,
                order_id=resp.get("orderID", ""),
                filled=float(resp.get("size", 0)),
                avg_price=float(resp.get("price", 0)),
            )
        else:
            return OrderResult(
                success=False,
                error=str(resp),
            )

    async def get_book_price(self, token_id: str) -> float:
        """
        Получает лучшую цену ask (цену входа) для токена.
        Возвращает 0 при ошибке.
        """
        if config.DRY_RUN or self._client is None:
            return 0.52   # заглушка для dry run

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            book = await loop.run_in_executor(
                None, self._client.get_order_book, token_id
            )
            asks = book.get("asks", [])
            if asks:
                return float(asks[0]["price"])
        except Exception as e:
            log.warning(f"Failed to get book price: {e}")

        return 0.0
