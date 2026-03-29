"""
Журнал сделок — записывает каждую сделку в CSV файл.

После каждой недели можно открыть trades.csv и проанализировать:
- Реальная средняя цена входа
- WR по парам, сигналам, длине стрика, уровням пирамиды
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
    "open_datetime",   # время открытия сделки (UTC)
    "close_datetime",  # время резолюции (UTC)
    "pair",            # btcusdt / dogeusdt / bnbusdt
    "signal",          # B / C / D / D_C / DC
    "streak_len",      # длина стрика (5 / 3 / 2)
    "dc_level",        # уровень пирамиды для DC (1/2/3), пусто для S2
    "direction",       # YES / NO
    "bet_usd",         # размер ставки $
    "entry_price",     # реальная цена входа (¢)
    "potential_win",   # потенциальный выигрыш $
    "order_id",        # ID ордера на Polymarket
    "condition_id",    # ID рынка на Polymarket (для верификации)
    "result",          # WIN / LOSS / OPEN
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
    streak_len: int,
    direction: str,
    bet_usd: float,
    entry_price: float,
    order_id: str,
    condition_id: str,
    dry_run: bool,
    dc_level: int = 0,
) -> None:
    """Записываем открытие сделки в CSV."""
    _ensure_header()

    row = {
        "open_datetime":  datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "close_datetime": "",
        "pair":           pair,
        "signal":         signal,
        "streak_len":     streak_len,
        "dc_level":       dc_level if dc_level > 0 else "",
        "direction":      direction,
        "bet_usd":        bet_usd,
        "entry_price":    round(entry_price, 4),
        "potential_win":  round(bet_usd * (1 / entry_price - 1), 2) if entry_price > 0 else 0,
        "order_id":       order_id,
        "condition_id":   condition_id,
        "result":         "OPEN",
        "pnl":            "",
        "dry_run":        dry_run,
    }

    with open(JOURNAL_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)

    log.info(
        f"[ЖУРНАЛ] Открыта: {pair} {signal}(streak={streak_len}) "
        f"{direction} ${bet_usd} @ {entry_price:.2f}¢"
    )


def log_result(
    order_id: str,
    win: bool,
    entry_price: float,
    bet_usd: float,
) -> None:
    """Обновляем результат сделки по order_id."""
    if not os.path.exists(JOURNAL_FILE):
        return

    rows = []
    updated = False
    close_dt = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with open(JOURNAL_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["order_id"] == order_id and row["result"] == "OPEN":
                row["result"]         = "WIN" if win else "LOSS"
                row["close_datetime"] = close_dt
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


def print_summary() -> None:
    """Выводит статистику по журналу в лог."""
    if not os.path.exists(JOURNAL_FILE):
        log.info("[ЖУРНАЛ] Файл пустой — сделок ещё не было.")
        return

    with open(JOURNAL_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["result"] in ("WIN", "LOSS")]

    if not rows:
        log.info("[ЖУРНАЛ] Нет завершённых сделок.")
        return

    total = len(rows)
    wins  = sum(1 for r in rows if r["result"] == "WIN")
    wr    = wins / total * 100
    pnl   = sum(float(r["pnl"]) for r in rows if r["pnl"])
    prices = [float(r["entry_price"]) for r in rows if r["entry_price"]]
    avg_price = sum(prices) / len(prices) if prices else 0

    log.info(
        f"[ЖУРНАЛ] ── Статистика ──────────────────────────\n"
        f"  Всего сделок:  {total}\n"
        f"  Win Rate:      {wr:.1f}%\n"
        f"  P&L итого:     {pnl:+.2f}$\n"
        f"  Ср. цена входа:{avg_price:.2f}¢\n"
        f"  ────────────────────────────────────────────────"
    )

    # Разбивка по сигналам
    for sig in ("B", "C", "D", "D_C", "DC"):
        sig_rows = [r for r in rows if r["signal"] == sig]
        if not sig_rows:
            continue
        s_wins = sum(1 for r in sig_rows if r["result"] == "WIN")
        s_wr   = s_wins / len(sig_rows) * 100
        s_pnl  = sum(float(r["pnl"]) for r in sig_rows if r["pnl"])
        log.info(
            f"[ЖУРНАЛ]   {sig:4s}: {len(sig_rows):3d} сделок  "
            f"WR {s_wr:.1f}%  P&L {s_pnl:+.2f}$"
        )
