from __future__ import annotations

import logging
import sys


def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Неизвестный уровень логирования: {level_name}")

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    return logging.getLogger("docmanage")
