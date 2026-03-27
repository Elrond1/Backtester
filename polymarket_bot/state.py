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
        pair: {"prev_b_win": None, "prev_d_win": None}
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
            self._data[pair] = {"prev_b_win": None, "prev_d_win": None}

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
