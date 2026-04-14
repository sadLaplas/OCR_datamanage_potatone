from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from .config import AppConfig, ConfigError, load_config, prepare_directories
from .documents import (
    DocumentRegistrationError,
    RegisteredDocument,
    register_documents,
)
from .logger import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        logger = setup_logging(config.log_level)
        directory_statuses = prepare_directories(config)
        if args.command == "register":
            registered_documents, manifest_path = register_documents(config, args.paths)
        else:
            registered_documents = []
            manifest_path = config.manifest_path
    except ConfigError as error:
        print(f"Ошибка конфигурации: {error}", file=sys.stderr)
        return 1
    except DocumentRegistrationError as error:
        print(f"Ошибка регистрации: {error}", file=sys.stderr)
        return 1

    if args.command == "register":
        logger.info("Документы зарегистрированы.")
        print(render_registration_report(registered_documents, manifest_path))
    else:
        logger.info("Конфигурация загружена.")
        print(render_report(config, directory_statuses))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docmanage",
        description="Показывает состояние проекта и регистрирует документы.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Путь к YAML-конфигурации.",
    )
    subparsers = parser.add_subparsers(dest="command")
    register_parser = subparsers.add_parser(
        "register",
        help="Регистрирует документы в manifest.",
    )
    register_parser.add_argument(
        "paths",
        nargs="*",
        help="Пути к файлам для регистрации.",
    )
    return parser


def render_report(config: AppConfig, directory_statuses: dict[str, str]) -> str:
    lines = [
        f"Проект: {config.project_name}",
        f"Режим: {config.run_mode}",
        f"Конфиг: {config.config_path}",
        f"Текущая директория: {Path.cwd()}",
        f"Python: {platform.python_version()}",
        f"Manifest: {config.manifest_path}",
        "Директории:",
    ]

    for name, path in config.directories.items():
        status = directory_statuses[name]
        lines.append(f"- {name}: {path} ({status})")

    lines.append("Статус: конфигурация и окружение доступны.")
    return "\n".join(lines)


def render_registration_report(
    registered_documents: list[RegisteredDocument], manifest_path: Path
) -> str:
    lines = [
        f"Manifest: {manifest_path}",
        f"Зарегистрировано документов: {len(registered_documents)}",
    ]

    for document in registered_documents:
        lines.append(
            f"- {document.document_id} | {document.document_kind} | {document.original_name}"
        )

    return "\n".join(lines)
