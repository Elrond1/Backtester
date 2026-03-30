"""
Главный оркестратор бота.

Поток:
  1. Binance WS → 15m свечи по BTC/DOGE/BNB
  2. При закрытии ПЕРВОЙ 15m свечи нового часа (HH:15):
     - Проверяем сигнал B/C/D/D_C
     - Ищем рынок на Polymarket
     - Проверяем цену входа (≤ MAX_ENTRY_PRICE)
     - Размещаем ордер
     - Обновляем состояние
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

from . import config
from .candles import BinanceFeed, Candle
from .executor import Executor
from .journal import log_open, log_result, print_summary
from .markets import MarketFinder
from .signals import Signal, check_signal, check_dc_signal
from .state import BotState
from .telegram_notify import send, fmt_open, fmt_result, fmt_error

log = logging.getLogger(__name__)


class Bot:
    def __init__(self):
        self._symbols = list(config.PAIRS.keys())
        self._state   = BotState(config.STATE_FILE, self._symbols)
        self._finder  = MarketFinder()
        self._exec    = Executor()
        self._feed    = BinanceFeed(
            symbols=self._symbols,
            on_candle_close=self._on_candle,
        )

    # ── candle handler ────────────────────────────────────────────────────────

    async def _on_candle(self, candle: Candle):
        """Вызывается Binance feed при каждой закрытой 15m свече."""
        buf = self._feed.get_buffer(candle.symbol)

        # ── S2 сигнал (HH:15) ─────────────────────────────────────────────────
        signal = check_signal(
            candle=candle,
            buf=buf,
            prev_b_win=self._state.get_prev_b_win(candle.symbol),
            prev_d_win=self._state.get_prev_d_win(candle.symbol),
        )
        if signal is not None:
            # Запоминаем для DC: S2 сработал в этом часу
            self._state.set_dc_pending(
                pair=candle.symbol,
                hour_iso=signal.hour_start.isoformat(),
                direction=signal.direction,
            )
            await self._handle_signal(signal)
            return

        # ── DC сигнал (HH:30) ─────────────────────────────────────────────────
        dc_signal = check_dc_signal(
            candle=candle,
            buf=buf,
            pending_direction=self._state.get_dc_pending_direction(candle.symbol),
        )
        if dc_signal is not None:
            self._state.clear_dc_pending(candle.symbol)
            await self._handle_dc_signal(dc_signal)

    # ── signal handler ────────────────────────────────────────────────────────

    async def _handle_signal(self, signal: Signal):
        pair_cfg = config.PAIRS[signal.symbol]
        bet_usd  = pair_cfg["bets"][signal.sig_type]
        keyword  = pair_cfg["poly_slug"]

        log.info(
            f"[{signal.symbol.upper()}] {'='*50}\n"
            f"  Signal:    {signal.sig_type} ({signal.direction})\n"
            f"  Bet:       ${bet_usd}\n"
            f"  Hour:      {signal.hour_start.strftime('%H:%M UTC')}\n"
            f"  {'='*50}"
        )

        # Ищем рынок на Polymarket
        market = await self._finder.find(keyword, signal.hour_start)
        if market is None:
            log.error(
                f"[{signal.symbol.upper()}] Market not found — SKIP "
                f"(проверь slug: {keyword})"
            )
            return

        # Проверяем цену входа
        token_id = market.yes_token_id if signal.direction == "YES" else market.no_token_id
        price = await self._exec.get_book_price(token_id)

        if price > config.MAX_ENTRY_PRICE:
            log.warning(
                f"[{signal.symbol.upper()}] Price too high: {price:.2f}¢ > "
                f"{config.MAX_ENTRY_PRICE:.2f}¢ — SKIP"
            )
            return

        if price > 0:
            log.info(f"[{signal.symbol.upper()}] Entry price: {price:.2f}¢ ✓")

        # Размещаем ордер
        result = await self._exec.place(signal, market, bet_usd)

        if result.success:
            log.info(
                f"[{signal.symbol.upper()}] ORDER PLACED ✓ "
                f"id={result.order_id} "
                f"filled=${result.filled:.0f} @ {result.avg_price:.2f}¢"
            )
            asyncio.create_task(send(
                config.TG_TOKEN, config.TG_CHAT_ID,
                fmt_open(signal.symbol, signal.sig_type, signal.direction, bet_usd, result.avg_price, config.DRY_RUN),
            ))
            # Записываем в журнал
            log_open(
                pair=signal.symbol,
                signal=signal.sig_type,
                streak_len=signal.streak_len,
                direction=signal.direction,
                bet_usd=bet_usd,
                entry_price=result.avg_price,
                order_id=result.order_id,
                condition_id=market.condition_id,
                dry_run=config.DRY_RUN,
            )
            asyncio.create_task(
                self._schedule_resolution_check(signal, market, result.order_id, result.avg_price, bet_usd)
            )
        else:
            log.error(
                f"[{signal.symbol.upper()}] ORDER FAILED: {result.error}"
            )

    # ── DC signal handler (пирамидинг) ────────────────────────────────────────

    async def _handle_dc_signal(self, signal: Signal):
        level   = self._state.get_dc_pyramid_level(signal.symbol)
        base    = config.DC_BASE_BETS[signal.symbol]
        bet_usd = base * config.DC_PYRAMID_MULTIPLIERS[level]
        keyword = config.PAIRS[signal.symbol]["poly_slug"]

        log.info(
            f"[{signal.symbol.upper()}] {'='*50}\n"
            f"  Signal:    DC (двойное подтверждение)\n"
            f"  Направление: {signal.direction}\n"
            f"  Пирамида:  уровень {level} → ставка ${bet_usd}\n"
            f"  Hour:      {signal.hour_start.strftime('%H:%M UTC')}\n"
            f"  {'='*50}"
        )

        market = await self._finder.find(keyword, signal.hour_start)
        if market is None:
            log.error(f"[{signal.symbol.upper()}] DC: Market not found — SKIP")
            return

        token_id = market.yes_token_id if signal.direction == "YES" else market.no_token_id
        price = await self._exec.get_book_price(token_id)

        if price > config.MAX_ENTRY_PRICE:
            log.warning(
                f"[{signal.symbol.upper()}] DC: Price too high: {price:.2f}¢ — SKIP"
            )
            return

        result = await self._exec.place(signal, market, bet_usd)

        if result.success:
            log.info(
                f"[{signal.symbol.upper()}] DC ORDER PLACED ✓ "
                f"id={result.order_id} "
                f"filled=${result.filled:.0f} @ {result.avg_price:.2f}¢ "
                f"(уровень пирамиды {level})"
            )
            asyncio.create_task(send(
                config.TG_TOKEN, config.TG_CHAT_ID,
                fmt_open(signal.symbol, f"DC[{level}]", signal.direction, bet_usd, result.avg_price, config.DRY_RUN),
            ))
            log_open(
                pair=signal.symbol,
                signal="DC",
                streak_len=signal.streak_len,
                direction=signal.direction,
                bet_usd=bet_usd,
                entry_price=result.avg_price,
                order_id=result.order_id,
                condition_id=market.condition_id,
                dry_run=config.DRY_RUN,
                dc_level=level,
            )
            asyncio.create_task(
                self._schedule_dc_resolution(signal, market, level, result.order_id, result.avg_price)
            )
        else:
            log.error(f"[{signal.symbol.upper()}] DC ORDER FAILED: {result.error}")

    # ── resolution check ──────────────────────────────────────────────────────

    async def _schedule_resolution_check(
        self,
        signal: Signal,
        market,
        order_id: str,
        entry_price: float,
        bet_usd: float,
    ):
        """
        Ждём до конца часа и проверяем результат.
        Win = направление верное (YES выиграл если цена выросла).
        Обновляем prev_b_win / prev_d_win.
        """
        now = datetime.now(tz=timezone.utc)
        resolve_at = market.end_date
        wait_sec = (resolve_at - now).total_seconds() + 30  # +30s после резолюции

        if wait_sec > 0:
            log.info(
                f"[{signal.symbol.upper()}] Waiting {wait_sec:.0f}s for resolution "
                f"at {resolve_at.strftime('%H:%M UTC')}..."
            )
            await asyncio.sleep(wait_sec)

        # Получаем результат через Gamma API
        win = await self._check_resolution(market)
        if win is None:
            log.warning(
                f"[{signal.symbol.upper()}] Could not determine resolution "
                f"for {signal.sig_type}"
            )
            return

        outcome = "WIN ✓" if win else "LOSS ✗"
        log.info(f"[{signal.symbol.upper()}] {signal.sig_type} → {outcome}")

        pnl = bet_usd * (1 / entry_price - 1) if win else -bet_usd
        asyncio.create_task(send(
            config.TG_TOKEN, config.TG_CHAT_ID,
            fmt_result(signal.symbol, signal.sig_type, win, pnl, config.DRY_RUN),
        ))
        log_result(order_id, win, entry_price, bet_usd)

        if signal.sig_type in ("B", "C"):
            self._state.set_prev_b_win(signal.symbol, win)
        else:  # D, D_C
            self._state.set_prev_d_win(signal.symbol, win)

    async def _schedule_dc_resolution(
        self,
        signal: Signal,
        market,
        level: int,
        order_id: str,
        entry_price: float,
    ):
        """Ждём результата DC сделки и обновляем уровень пирамиды."""
        now = datetime.now(tz=timezone.utc)
        wait_sec = (market.end_date - now).total_seconds() + 30
        if wait_sec > 0:
            await asyncio.sleep(wait_sec)

        win = await self._check_resolution(market)
        if win is None:
            log.warning(f"[{signal.symbol.upper()}] DC: Не удалось определить результат")
            return

        outcome = "WIN ✓" if win else "LOSS ✗"
        base    = config.DC_BASE_BETS[signal.symbol]
        bet_usd = base * config.DC_PYRAMID_MULTIPLIERS[level]
        pnl     = bet_usd * (1 / entry_price - 1) if win else -bet_usd

        asyncio.create_task(send(
            config.TG_TOKEN, config.TG_CHAT_ID,
            fmt_result(signal.symbol, f"DC[{level}]", win, pnl, config.DRY_RUN),
        ))
        log_result(order_id, win, entry_price, bet_usd)

        self._state.update_dc_pyramid(signal.symbol, win)
        new_level = self._state.get_dc_pyramid_level(signal.symbol)

        log.info(
            f"[{signal.symbol.upper()}] DC → {outcome}  "
            f"PnL: {pnl:+.2f}$  "
            f"Следующий уровень пирамиды: {new_level}"
        )

    async def _check_resolution(self, market) -> bool | None:
        """
        Проверяет разрешился ли рынок и в какую сторону.
        Returns True (YES выиграл) / False (NO выиграл) / None (ошибка).
        """
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{config.GAMMA_HOST}/markets/{market.condition_id}",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()

            resolved = data.get("resolved", False)
            if not resolved:
                # Рынок ещё не разрешён — подождём ещё
                await asyncio.sleep(60)
                return await self._check_resolution(market)

            winning_outcome = data.get("winningOutcome", "")
            return winning_outcome.upper() == "YES"

        except Exception as e:
            log.error(f"Resolution check failed: {e}")
            return None

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        log.info(
            f"Bot starting | pairs: {list(config.PAIRS.keys())} | "
            f"DRY_RUN={config.DRY_RUN}"
        )
        if config.DRY_RUN:
            log.info("*** DRY RUN MODE — ордера не размещаются ***")

        print_summary()
        await self._feed.run()
