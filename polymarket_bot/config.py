"""
Polymarket Bot — конфигурация.

Перед запуском:
1. Создай .env файл с POLY_PRIVATE_KEY=0x...
2. Настрой BETS под свой капитал
3. Проверь лимиты ликвидности на Polymarket (BTC: $500/$750, ETH/SOL: $75/$100)

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
        "poly_slug":  "Bitcoin Up or Down",   # ключевое слово для поиска рынка на Polymarket
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
    "ethusdt": {
        "label":     "ETH",
        "poly_slug": "Ethereum Up or Down",
        "bets": {
            # $100 старт: ETH/SOL = 1% капитала (мин. ордер $1)
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
    "solusdt": {
        "label":     "SOL",
        "poly_slug": "Solana Up or Down",
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

# ── Множитель при совпадении всех 3 пар (WR 71.9% vs 67.5%) ─────────────────
# Если все 3 пары дают сигнал в одном направлении → ставка × множитель
TRIPLE_CONFIRM_MULTIPLIER = 2   # удвоить ставку
TRIPLE_CONFIRM_WAIT_SEC   = 5   # секунд ждать остальные пары после первого сигнала

# ── DC пирамида (двойное подтверждение, WR 85.1%) ────────────────────────────
# Базовая ставка уровня 1. Уровни: $1 → $2 → $4 (для $100 капитала)
# Уровни: $100 → $200 → $400 (для $10k капитала)
DC_BASE_BETS = {
    "btcusdt": 1,   # $1 базовая ($100 старт) / $100 ($10k старт)
    "ethusdt": 1,
    "solusdt": 1,
}
# Множители пирамиды: уровень 1 → ×1, уровень 2 → ×2, уровень 3 → ×4
DC_PYRAMID_MULTIPLIERS = {1: 1, 2: 2, 3: 4}

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

# ── Telegram уведомления ──────────────────────────────────────────────────────
TG_TOKEN   = os.getenv("TG_TOKEN", "")              # токен от BotFather
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "-1001093356102")  # канал "Спекулянт"
