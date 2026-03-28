"""
Polymarket Bot — конфигурация.

Перед запуском:
1. Создай .env файл с POLY_PRIVATE_KEY=0x...
2. Настрой BETS под свой капитал
3. Проверь лимиты ликвидности на Polymarket (BTC: $500/$750, DOGE/BNB: $75/$100)

Текущий режим: $100 стартовый капитал (пропорционально к основным ставкам 1:100)
Минимальный ордер Polymarket: $1
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Polymarket ────────────────────────────────────────────────────────────────
POLY_PRIVATE_KEY   = os.getenv("POLY_PRIVATE_KEY", "")   # 0x...
POLY_API_KEY       = os.getenv("POLY_API_KEY", "")        # опционально (L2 auth)
POLY_API_SECRET    = os.getenv("POLY_API_SECRET", "")
POLY_API_PASSPHRASE = os.getenv("POLY_API_PASSPHRASE", "")

CLOB_HOST   = "https://clob.polymarket.com"
GAMMA_HOST  = "https://gamma-api.polymarket.com"
CHAIN_ID    = 137   # Polygon mainnet

# ── Пары и ставки ─────────────────────────────────────────────────────────────
# Ключ: Binance symbol (lowercase)  →  конфиг пары
PAIRS = {
    "btcusdt": {
        "label":      "BTC",
        "poly_slug":  "bitcoin",   # ключевое слово для поиска рынка на Polymarket
        "bets": {
            # $100 старт: BTC = 2% / 3% капитала (мин. ордер $1)
            "B":   2,   # $ на сделку
            "C":   3,
            "D":   2,
            "D_C": 3,
            # $10k старт: раскомментировать и закомментировать строки выше
            # "B":   200,
            # "C":   300,
            # "D":   200,
            # "D_C": 300,
        },
    },
    "dogeusdt": {
        "label":     "DOGE",
        "poly_slug": "dogecoin",
        "bets": {
            # $100 старт: DOGE/BNB = 1% капитала (мин. ордер $1)
            "B":   1,
            "C":   1,
            "D":   1,
            "D_C": 1,
            # $10k старт:
            # "B":   75,
            # "C":   100,
            # "D":   75,
            # "D_C": 100,
        },
    },
    "bnbusdt": {
        "label":     "BNB",
        "poly_slug": "bnb",
        "bets": {
            # $100 старт
            "B":   1,
            "C":   1,
            "D":   1,
            "D_C": 1,
            # $10k старт:
            # "B":   75,
            # "C":   100,
            # "D":   75,
            # "D_C": 100,
        },
    },
}

# ── Параметры стратегии ───────────────────────────────────────────────────────
TIMEFRAME_MINUTES = 15     # 15m свечи
MAX_SLIPPAGE      = 0.03   # максимальный slippage от целевой цены (3 цента)
MAX_ENTRY_PRICE   = 0.67   # не входить если цена выше 67¢ (нет edge)

# ── Файл состояния ────────────────────────────────────────────────────────────
STATE_FILE = "bot_state.json"   # хранит prev_b_win / prev_d_win на рестарт

# ── Логирование ───────────────────────────────────────────────────────────────
LOG_FILE      = "bot.log"
LOG_LEVEL     = "INFO"

# ── Dry run (не ставит реальные ордера) ──────────────────────────────────────
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
