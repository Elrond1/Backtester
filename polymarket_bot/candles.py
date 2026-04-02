"""
Real-time 15m candle feed через Binance WebSocket.

Поддерживает скользящий буфер последних N закрытых свечей по каждой паре.
Сигнализирует через asyncio.Queue когда свеча закрывается.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional

import aiohttp
import websockets

log = logging.getLogger(__name__)

BINANCE_WS = "wss://stream.binance.com:9443/stream"
BUFFER_SIZE = 10   # сколько закрытых свечей хранить (нужно минимум 7 для стрика-5 + 2 запаса)


@dataclass
class Candle:
    symbol: str       # "btcusdt"
    open_time: int    # unix ms
    open:  float
    high:  float
    low:   float
    close: float
    is_closed: bool

    @property
    def green(self) -> bool:
        return self.close > self.open

    @property
    def open_dt(self):
        from datetime import datetime, timezone
        return datetime.fromtimestamp(self.open_time / 1000, tz=timezone.utc)


class CandleBuffer:
    """Скользящий буфер закрытых свечей."""

    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._buf: Deque[Candle] = deque(maxlen=maxlen)

    def push(self, candle: Candle):
        """Добавить закрытую свечу."""
        assert candle.is_closed
        self._buf.append(candle)

    def closed(self) -> list[Candle]:
        """Последние N закрытых свечей (от старых к новым)."""
        return list(self._buf)

    def __len__(self):
        return len(self._buf)


class BinanceFeed:
    """
    Подписывается на Binance multi-stream WS для нескольких пар, 15m свечи.
    Коллбек вызывается при каждой закрытой свече.
    """

    def __init__(
        self,
        symbols: list[str],          # ["btcusdt", "dogeusdt", "bnbusdt"]
        on_candle_close: Callable,   # async fn(candle: Candle)
        interval: str = "15m",
    ):
        self._symbols = [s.lower() for s in symbols]
        self._interval = interval
        self._on_close = on_candle_close
        self._buffers: Dict[str, CandleBuffer] = {
            s: CandleBuffer() for s in self._symbols
        }
        self._running = False

    def get_buffer(self, symbol: str) -> CandleBuffer:
        return self._buffers[symbol.lower()]

    async def preload(self):
        """Загружает последние 10 закрытых свечей через Binance REST API."""
        url = "https://api.binance.com/api/v3/klines"
        for symbol in self._symbols:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(url, params={
                        "symbol": symbol.upper(),
                        "interval": self._interval,
                        "limit": BUFFER_SIZE + 1,  # +1 т.к. последняя может быть незакрытой
                    }) as r:
                        data = await r.json()
                candles_loaded = 0
                for k in data[:-1]:  # последнюю пропускаем — она ещё открыта
                    candle = Candle(
                        symbol=symbol,
                        open_time=k[0],
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        is_closed=True,
                    )
                    self._buffers[symbol].push(candle)
                    candles_loaded += 1
                log.info(f"[{symbol.upper()}] Preloaded {candles_loaded} candles")
            except Exception as e:
                log.warning(f"[{symbol.upper()}] Preload failed: {e}")

    def _stream_url(self) -> str:
        streams = "/".join(
            f"{s}@kline_{self._interval}" for s in self._symbols
        )
        return f"{BINANCE_WS}?streams={streams}"

    @staticmethod
    def _parse(msg: dict) -> Optional[Candle]:
        data = msg.get("data", {})
        k = data.get("k", {})
        if not k:
            return None
        return Candle(
            symbol=data["s"].lower(),
            open_time=k["t"],
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            is_closed=bool(k["x"]),
        )

    async def run(self):
        self._running = True
        url = self._stream_url()
        log.info(f"Connecting to Binance WS: {url[:80]}...")

        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("Binance WS connected")
                    async for raw in ws:
                        msg = json.loads(raw)
                        candle = self._parse(msg)
                        if candle is None:
                            continue

                        if candle.is_closed:
                            self._buffers[candle.symbol].push(candle)
                            log.info(
                                f"[{candle.symbol.upper()}] closed "
                                f"{candle.open_dt.strftime('%H:%M')} "
                                f"{'▲' if candle.green else '▼'} "
                                f"o={candle.open} c={candle.close}"
                            )
                            await self._on_close(candle)

            except websockets.ConnectionClosed as e:
                log.warning(f"WS connection closed ({e}), reconnecting in 3s...")
                await asyncio.sleep(3)
            except Exception as e:
                log.error(f"WS error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def stop(self):
        self._running = False
