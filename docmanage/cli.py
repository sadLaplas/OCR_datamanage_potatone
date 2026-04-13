from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from .config import AppConfig, ConfigError, load_config, prepare_directories
from .logger import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        logger = setup_logging(config.log_level)
        directory_statuses = prepare_directories(config)
    except ConfigError as error:
        print(f"Ошибка конфигурации: {error}", file=sys.stderr)
        return 1

    logger.info("Конфигурация загружена.")
    print(render_report(config, directory_statuses))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docmanage",
        description="Показывает базовое состояние проекта и конфигурации.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Путь к YAML-конфигурации.",
    )
    return parser


def render_report(config: AppConfig, directory_statuses: dict[str, str]) -> str:
    lines = [
        f"Проект: {config.project_name}",
        f"Режим: {config.run_mode}",
        f"Конфиг: {config.config_path}",
        f"Текущая директория: {Path.cwd()}",
        f"Python: {platform.python_version()}",
        "Директории:",
    ]

    for name, path in config.directories.items():
        status = directory_statuses[name]
        lines.append(f"- {name}: {path} ({status})")

    lines.append("Статус: конфигурация и окружение доступны.")
    return "\n".join(lines)
