"""
Главный оркестратор бота.

Поток:
  1. Binance WS → 15m свечи по BTC/ETH/SOL
  2. При закрытии ПЕРВОЙ 15m свечи нового часа (HH:15):
     - Проверяем сигнал B/C/D/D_C
     - Определяем множитель ставки: x2 если сигнал совпадает с направлением вчерашнего дня
     - Ищем рынок на Polymarket
     - Проверяем цену входа (≤ MAX_ENTRY_PRICE)
     - Размещаем ордер
     - Обновляем состояние
  3. При закрытии ВТОРОЙ 15m свечи (HH:30):
     - Проверяем DC сигнал (двойное подтверждение)
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

import aiohttp

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
        # Направление предыдущего дня: {symbol: True=вверх, False=вниз, None=неизвестно}
        self._prev_day_up: dict[str, bool | None] = {s: None for s in self._symbols}
        # Открытие текущего дня: {symbol: float}
        self._day_open: dict[str, float | None] = {s: None for s in self._symbols}

    # ── инициализация дневного тренда ─────────────────────────────────────────

    async def _init_daily_trend(self):
        """Загружаем вчерашнюю дневную свечу для каждой пары при старте."""
        for symbol in self._symbols:
            try:
                pair = symbol.upper().replace("USDT", "") + "USDT"
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.binance.com/api/v3/klines",
                        params={"symbol": pair, "interval": "1d", "limit": 2},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        data = await resp.json()

                if len(data) >= 2:
                    prev = data[-2]   # вчерашняя закрытая свеча
                    prev_open  = float(prev[1])
                    prev_close = float(prev[4])
                    self._prev_day_up[symbol] = prev_close > prev_open

                    today = data[-1]  # сегодняшняя (ещё не закрыта)
                    self._day_open[symbol] = float(today[1])

                    direction = "▲" if self._prev_day_up[symbol] else "▼"
                    log.info(f"[{symbol.upper()}] Вчерашний день: {direction}  "
                             f"(open={prev_open:.2f} close={prev_close:.2f})")
            except Exception as e:
                log.warning(f"[{symbol.upper()}] Не удалось загрузить дневной тренд: {e}")

    def _get_trend_multiplier(self, symbol: str, direction: str) -> int:
        """
        Возвращает множитель ставки на основе тренда предыдущего дня.
        x2 если сигнал совпадает с направлением вчерашнего дня.
        x1 если против тренда или тренд неизвестен.
        """
        prev_up = self._prev_day_up.get(symbol)
        if prev_up is None:
            return 1
        signal_up = (direction == "YES")
        return config.TREND_MULTIPLIER if (signal_up == prev_up) else 1

    # ── candle handler ────────────────────────────────────────────────────────

    async def _on_candle(self, candle: Candle):
        """Вызывается Binance feed при каждой закрытой 15m свече."""
        buf = self._feed.get_buffer(candle.symbol)

        # Обновляем дневной тренд при начале нового дня (00:00 UTC)
        if candle.open_dt.hour == 0 and candle.open_dt.minute == 0:
            prev_open = self._day_open.get(candle.symbol)
            if prev_open is not None:
                self._prev_day_up[candle.symbol] = candle.open > prev_open
                direction = "▲" if self._prev_day_up[candle.symbol] else "▼"
                log.info(f"[{candle.symbol.upper()}] Новый день. Вчера: {direction}")
            self._day_open[candle.symbol] = candle.open

        # При начале нового часа (HH:00) сбрасываем pending от предыдущего часа
        # чтобы незакрытый DC прошлого часа не сработал ложно в следующем часу
        if candle.open_dt.minute == 0:
            self._state.clear_dc_pending(candle.symbol)

        # ── S2 сигнал (HH:15) ─────────────────────────────────────────────────
        signal = check_signal(
            candle=candle,
            buf=buf,
            prev_b_win=self._state.get_prev_b_win(candle.symbol),
            prev_d_win=self._state.get_prev_d_win(candle.symbol),
        )
        if signal is not None:
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
        pair_cfg   = config.PAIRS[signal.symbol]
        multiplier = self._get_trend_multiplier(signal.symbol, signal.direction)
        bet_usd    = pair_cfg["bets"][signal.sig_type] * multiplier
        keyword    = pair_cfg["poly_slug"]

        trend_info = f" [ТРЕНД x{multiplier}]" if multiplier > 1 else ""
        log.info(
            f"[{signal.symbol.upper()}] {'='*50}\n"
            f"  Signal:    {signal.sig_type} ({signal.direction}){trend_info}\n"
            f"  Bet:       ${bet_usd}\n"
            f"  Hour:      {signal.hour_start.strftime('%H:%M UTC')}\n"
            f"  {'='*50}"
        )

        # Ищем рынок ТЕКУЩЕГО часа — slug строится по hour_start (HH:00 UTC)
        # Пример: сигнал в 16:15 UTC, hour_start=16:00 UTC → "12pm-et" (закрытие 17:00 UTC)
        market = await self._finder.find(keyword, signal.hour_start)
        if market is None:
            log.error(
                f"[{signal.symbol.upper()}] Market not found — SKIP "
                f"(проверь slug: {keyword})"
            )
            return

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

        result = await self._exec.place(signal, market, bet_usd)

        if result.success:
            log.info(
                f"[{signal.symbol.upper()}] ORDER PLACED ✓ "
                f"id={result.order_id} "
                f"filled=${result.filled:.0f} @ {result.avg_price:.2f}¢"
            )
            asyncio.create_task(send(
                config.TG_TOKEN, config.TG_CHAT_ID,
                fmt_open(signal.symbol, signal.sig_type, signal.direction,
                         bet_usd, result.avg_price, config.DRY_RUN),
            ))
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
                self._schedule_resolution_check(
                    signal, market, result.order_id, result.avg_price, bet_usd
                )
            )
        else:
            log.error(f"[{signal.symbol.upper()}] ORDER FAILED: {result.error}")

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
                fmt_open(signal.symbol, f"DC[{level}]", signal.direction,
                         bet_usd, result.avg_price, config.DRY_RUN),
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
                self._schedule_dc_resolution(
                    signal, market, level, result.order_id, result.avg_price
                )
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
        now      = datetime.now(tz=timezone.utc)
        wait_sec = (market.end_date - now).total_seconds() + 30

        if wait_sec > 0:
            log.info(
                f"[{signal.symbol.upper()}] Waiting {wait_sec:.0f}s for resolution "
                f"at {market.end_date.strftime('%H:%M UTC')}..."
            )
            await asyncio.sleep(wait_sec)

        win = await self._check_resolution(market)
        if win is None:
            log.warning(f"[{signal.symbol.upper()}] Could not determine resolution")
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
        else:
            self._state.set_prev_d_win(signal.symbol, win)

    async def _schedule_dc_resolution(
        self,
        signal: Signal,
        market,
        level: int,
        order_id: str,
        entry_price: float,
    ):
        now      = datetime.now(tz=timezone.utc)
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
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{config.GAMMA_HOST}/markets/{market.condition_id}",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()

            if not data.get("resolved", False):
                await asyncio.sleep(60)
                return await self._check_resolution(market)

            outcome = data.get("winningOutcome", "").lower()
            return outcome in ("yes", "up")

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
        await self._init_daily_trend()
        await self._feed.run()
