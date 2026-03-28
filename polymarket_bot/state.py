"""
Персистентное состояние бота (prev_b_win / prev_d_win по каждой паре).
Сохраняется в JSON файл — выживает при перезапуске.
"""

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


def _empty(pairs: list[str]) -> dict:
    return {
        pair: {
            "prev_b_win": None,
            "prev_d_win": None,
            "dc_pyramid_level": 1,        # текущий уровень пирамиды: 1, 2 или 3
            "dc_pending_hour": None,      # ISO-строка HH:00 если S2 уже сработал в этом часу
            "dc_pending_direction": None, # "YES" или "NO" — направление S2
        }
        for pair in pairs
    }


class BotState:
    def __init__(self, filepath: str, pairs: list[str]):
        self._path = filepath
        self._pairs = pairs
        self._data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
                log.info(f"State loaded from {self._path}")
            except Exception as e:
                log.warning(f"State load failed ({e}), starting fresh")
                self._data = _empty(self._pairs)
        else:
            self._data = _empty(self._pairs)

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _ensure(self, pair: str):
        if pair not in self._data:
            self._data[pair] = {
                "prev_b_win": None,
                "prev_d_win": None,
                "dc_pyramid_level": 1,
                "dc_pending_hour": None,
                "dc_pending_direction": None,
            }
        # добавляем новые поля если их нет в старом state файле
        for key, default in [
            ("dc_pyramid_level", 1),
            ("dc_pending_hour", None),
            ("dc_pending_direction", None),
        ]:
            if key not in self._data[pair]:
                self._data[pair][key] = default

    # ── B/C sequence ─────────────────────────────────────────────────────────

    def get_prev_b_win(self, pair: str) -> Optional[bool]:
        self._ensure(pair)
        return self._data[pair]["prev_b_win"]

    def set_prev_b_win(self, pair: str, win: bool):
        self._ensure(pair)
        self._data[pair]["prev_b_win"] = win
        self._save()

    # ── D/D_C sequence ────────────────────────────────────────────────────────

    def get_prev_d_win(self, pair: str) -> Optional[bool]:
        self._ensure(pair)
        return self._data[pair]["prev_d_win"]

    def set_prev_d_win(self, pair: str, win: bool):
        self._ensure(pair)
        self._data[pair]["prev_d_win"] = win
        self._save()

    # ── DC пирамида ───────────────────────────────────────────────────────────

    def get_dc_pyramid_level(self, pair: str) -> int:
        self._ensure(pair)
        return self._data[pair]["dc_pyramid_level"]

    def update_dc_pyramid(self, pair: str, win: bool):
        """Обновляет уровень пирамиды после результата DC сделки."""
        self._ensure(pair)
        level = self._data[pair]["dc_pyramid_level"]
        if win and level < 3:
            self._data[pair]["dc_pyramid_level"] = level + 1  # повышаем уровень
        else:
            self._data[pair]["dc_pyramid_level"] = 1          # сброс после проигрыша или уровня 3
        self._save()

    def get_dc_pending_direction(self, pair: str) -> Optional[str]:
        """Возвращает направление S2 сигнала этого часа (если был)."""
        self._ensure(pair)
        return self._data[pair]["dc_pending_direction"]

    def set_dc_pending(self, pair: str, hour_iso: str, direction: str):
        """Запоминаем что S2 сработал в этом часу."""
        self._ensure(pair)
        self._data[pair]["dc_pending_hour"] = hour_iso
        self._data[pair]["dc_pending_direction"] = direction
        self._save()

    def clear_dc_pending(self, pair: str):
        """Сбрасываем pending после того как DC проверен (или час закончился)."""
        self._ensure(pair)
        self._data[pair]["dc_pending_hour"] = None
        self._data[pair]["dc_pending_direction"] = None
        self._save()
