#!/usr/bin/env python3
"""
Точка входа Polymarket Bot.

Использование:
  python -m polymarket_bot.main              # dry run (из .env DRY_RUN=true)
  DRY_RUN=false python -m polymarket_bot.main  # реальные ордера
"""

import asyncio
import logging
import os
import signal
import sys


def setup_logging():
    from . import config

    fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if config.LOG_FILE:
        handlers.append(logging.FileHandler(config.LOG_FILE))

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


async def main():
    setup_logging()
    log = logging.getLogger("main")

    from . import config
    if not config.POLY_PRIVATE_KEY and not config.DRY_RUN:
        log.error(
            "POLY_PRIVATE_KEY не задан. "
            "Создай .env файл или запусти с DRY_RUN=true"
        )
        sys.exit(1)

    from .bot import Bot
    bot = Bot()

    # Корректное завершение по Ctrl+C / SIGTERM
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(bot)))

    try:
        await bot.run()
    except KeyboardInterrupt:
        log.info("Stopped by user")


async def _shutdown(bot):
    logging.getLogger("main").info("Shutting down...")
    bot._feed.stop()
    await asyncio.sleep(1)
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    asyncio.run(main())
