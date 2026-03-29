"""
Журнал сделок — записывает каждую сделку в CSV файл.

После каждой недели можно открыть trades.csv и проанализировать:
- Реальная средняя цена входа
- WR по парам и сигналам
- P&L в деньгах
- Сравнение с бэктестом
"""

import csv
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

JOURNAL_FILE = "trades.csv"

COLUMNS = [
    "datetime",        # время открытия сделки (UTC)
    "pair",            # btcusdt / dogeusdt / bnbusdt
    "signal",          # B / C / D / D_C / DC
    "direction",       # YES / NO
    "bet_usd",         # размер ставки $
    "entry_price",     # реальная цена входа (¢)
    "potential_win",   # потенциальный выигрыш $
    "order_id",        # ID ордера на Polymarket
    "result",          # WIN / LOSS / UNKNOWN
    "pnl",             # фактический P&L $
    "dry_run",         # True / False
]


def _ensure_header():
    """Создаём файл с заголовком если не существует."""
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
        log.info(f"Журнал сделок создан: {JOURNAL_FILE}")


def log_open(
    pair: str,
    signal: str,
    direction: str,
    bet_usd: float,
    entry_price: float,
    order_id: str,
    dry_run: bool,
) -> dict:
    """
    Записываем открытие сделки. Возвращает словарь для последующего обновления результата.
    """
    _ensure_header()

    row = {
        "datetime":      datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "pair":          pair,
        "signal":        signal,
        "direction":     direction,
        "bet_usd":       bet_usd,
        "entry_price":   round(entry_price, 4),
        "potential_win": round(bet_usd * (1 / entry_price - 1), 2) if entry_price > 0 else 0,
        "order_id":      order_id,
        "result":        "OPEN",
        "pnl":           "",
        "dry_run":       dry_run,
    }

    with open(JOURNAL_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)

    log.info(f"[ЖУРНАЛ] Открыта: {pair} {signal} {direction} ${bet_usd} @ {entry_price:.2f}¢")
    return row


def log_result(
    order_id: str,
    win: bool,
    entry_price: float,
    bet_usd: float,
):
    """
    Обновляем результат сделки по order_id.
    Читаем все строки, находим нужную, обновляем result и pnl.
    """
    if not os.path.exists(JOURNAL_FILE):
        return

    rows = []
    updated = False

    with open(JOURNAL_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["order_id"] == order_id and row["result"] == "OPEN":
                row["result"] = "WIN" if win else "LOSS"
                pnl = bet_usd * (1 / entry_price - 1) if win else -bet_usd
                row["pnl"] = round(pnl, 2)
                updated = True
            rows.append(row)

    if updated:
        with open(JOURNAL_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        result_str = "WIN" if win else "LOSS"
        pnl = bet_usd * (1 / entry_price - 1) if win else -bet_usd
        log.info(f"[ЖУРНАЛ] Результат: {order_id} → {result_str} {pnl:+.2f}$")


def print_summary():
    """Выводит краткую статистику по журналу в лог."""
    if not os.path.exists(JOURNAL_FILE):
        log.info("Журнал пустой — сделок ещё не было.")
        return

    rows = []
    with open(JOURNAL_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["result"] in ("WIN", "LOSS")]

    if not rows:
        log.info("Нет завершённых сделок.")
        return

    total = len(rows)
    wins = sum(1 for r in rows if r["result"] == "WIN")
    wr = wins / total * 100
    pnl = sum(float(r["pnl"]) for r in rows if r["pnl"])
    prices = [float(r["entry_price"]) for r in rows if r["entry_price"]]
    avg_price = sum(prices) / len(prices) if prices else 0

    log.info(
        f"[ЖУРНАЛ] Итого: {total} сделок | "
        f"WR: {wr:.1f}% | "
        f"P&L: {pnl:+.2f}$ | "
        f"Ср. цена входа: {avg_price:.2f}¢"
    )
