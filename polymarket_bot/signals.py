"""
Детектор сигналов B / C / D / D_C.

Логика идентична бэктесту (strategy2_realistic.py), но работает в реальном времени.

Временная шкала:
  HH:00 — начало часа
  HH:15 — первая 15m свеча часа закрывается → ЗДЕСЬ мы проверяем сигнал
  HH:59 — Polymarket контракт резолвится

Условия сигнала (проверяем при закрытии свечи в HH:15):
  buf[-7]  — свеча до стрика (проверка "изолированности" стрика-5)
  buf[-6..-2] — 5 свечей стрика (или buf[-4..-2] — 3 свечи для D)
  buf[-1]  — текущая свеча (первая 15m нового часа, закрытая)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .candles import Candle, CandleBuffer

log = logging.getLogger(__name__)


@dataclass
class Signal:
    symbol:    str        # "btcusdt"
    sig_type:  str        # "B" | "C" | "D" | "D_C"
    direction: str        # "YES" (лонг) | "NO" (шорт)
    timestamp: datetime   # время сигнала (HH:15)
    hour_start: datetime  # начало часа (HH:00) — время resolution контракта


def _is_hour_start_candle(c: Candle) -> bool:
    """Свеча открылась ровно в начале часа (00, 15, 30, 45 min → только :00)."""
    return c.open_dt.minute == 0


def check_signal(
    candle: Candle,
    buf: CandleBuffer,
    prev_b_win: Optional[bool],
    prev_d_win: Optional[bool],
) -> Optional[Signal]:
    """
    Вызывается когда закрылась первая 15m свеча часа (время: HH:15).
    buf должен содержать >= 8 закрытых свечей.

    Returns Signal или None если сигнала нет.
    """
    if not _is_hour_start_candle(candle):
        return None   # эта свеча не первая в часе — пропускаем

    closed = buf.closed()
    # closed[-1] = только что закрытая свеча (первая 15m часа, HH:00–HH:15)
    # closed[-2] = HH-1:45 (предыдущая, 15m перед началом часа)
    # ...и так далее в прошлое

    if len(closed) < 8:
        log.debug(f"[{candle.symbol}] buffer too short ({len(closed)} < 8), skip")
        return None

    c0 = closed[-1]   # первая 15m нового часа — сигнальная свеча
    assert c0.open_time == candle.open_time

    colors = [c.green for c in closed]  # True=green, False=red
    # colors[-1] = c0
    # colors[-2] = HH-1:45
    # colors[-3] = HH-1:30
    # ...

    c0_green = colors[-1]

    # ── B / C: 5-стрик (closed[-6..-2]) ──────────────────────────────────────
    streak5 = colors[-6:-1]   # 5 свечей перед c0
    if len(streak5) == 5 and (all(streak5) or not any(streak5)):
        s5_color = streak5[0]
        pre_streak = colors[-7] if len(closed) >= 7 else None

        # изолированный стрик: свеча ДО стрика имела другой цвет
        isolated = (pre_streak is not None) and (pre_streak != s5_color)
        # разворот: c0 идёт против стрика
        reversal = (c0_green != s5_color)

        if isolated and reversal:
            sig_type = "C" if prev_b_win is True else "B"
            direction = "YES" if c0_green else "NO"
            log.info(
                f"[{candle.symbol.upper()}] SIGNAL {sig_type} "
                f"{'▲' if c0_green else '▼'} "
                f"streak5={['G' if x else 'R' for x in streak5]}"
            )
            return Signal(
                symbol=candle.symbol,
                sig_type=sig_type,
                direction=direction,
                timestamp=c0.open_dt.replace(minute=15),   # HH:15
                hour_start=c0.open_dt,                      # HH:00
            )

    # ── D / D_C: 3-стрик (closed[-4..-2]) ────────────────────────────────────
    # Только если 5-стрика НЕТ (чтобы не дублировать сигналы)
    streak3 = colors[-4:-1]   # 3 свечи перед c0
    if len(streak3) == 3 and (all(streak3) or not any(streak3)):
        s3_color = streak3[0]
        reversal = (c0_green != s3_color)

        if reversal:
            sig_type = "D_C" if prev_d_win is True else "D"
            direction = "YES" if c0_green else "NO"
            log.info(
                f"[{candle.symbol.upper()}] SIGNAL {sig_type} "
                f"{'▲' if c0_green else '▼'} "
                f"streak3={['G' if x else 'R' for x in streak3]}"
            )
            return Signal(
                symbol=candle.symbol,
                sig_type=sig_type,
                direction=direction,
                timestamp=c0.open_dt.replace(minute=15),
                hour_start=c0.open_dt,
            )

    return None
