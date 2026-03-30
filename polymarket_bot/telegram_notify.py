"""
Telegram уведомления для бота.

Отправляет сообщения при:
- Открытии сделки
- Получении результата (WIN/LOSS)
- Ошибках бота

Настройка: добавь в .env файл:
  TG_TOKEN=<токен от BotFather>
  TG_CHAT_ID=306374789
"""

import logging
import aiohttp

log = logging.getLogger(__name__)


async def send(token: str, chat_id: str, text: str) -> None:
    """Отправляет сообщение в Telegram."""
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=aiohttp.ClientTimeout(total=5),
            )
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def fmt_open(pair: str, signal: str, direction: str, bet_usd: float, entry_price: float, dry_run: bool) -> str:
    arrow = "▲" if direction == "YES" else "▼"
    mode  = " [DRY RUN]" if dry_run else ""
    return (
        f"🟡 <b>Сделка открыта{mode}</b>\n"
        f"Пара:    {pair.upper()}\n"
        f"Сигнал:  {signal} {arrow}\n"
        f"Ставка:  ${bet_usd}\n"
        f"Цена:    {entry_price:.2f}¢\n"
        f"Потенциал: +${bet_usd * (1 / entry_price - 1):.2f}"
    )


def fmt_result(pair: str, signal: str, win: bool, pnl: float, dry_run: bool) -> str:
    icon  = "✅" if win else "❌"
    label = "WIN" if win else "LOSS"
    mode  = " [DRY RUN]" if dry_run else ""
    return (
        f"{icon} <b>{label}{mode}</b>\n"
        f"Пара:   {pair.upper()}\n"
        f"Сигнал: {signal}\n"
        f"P&L:    {pnl:+.2f}$"
    )


def fmt_error(pair: str, message: str) -> str:
    return f"⚠️ <b>Ошибка [{pair.upper()}]</b>\n{message}"
