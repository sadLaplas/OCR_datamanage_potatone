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
from .image_ingestion import (
    ImageIngestionError,
    ImageIngestionResult,
    ingest_image,
)
from .logger import setup_logging
from .pdf_ingestion import PdfIngestionError, PdfIngestionResult, ingest_pdf


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        logger = setup_logging(config.log_level)
        directory_statuses = prepare_directories(config)
        if args.command == "register":
            registered_documents, manifest_path = register_documents(config, args.paths)
            pdf_result = None
            image_result = None
        elif args.command == "ingest-pdf":
            pdf_result = ingest_pdf(config, args.source)
            registered_documents = []
            manifest_path = config.manifest_path
            image_result = None
        elif args.command == "ingest-image":
            image_result = ingest_image(config, args.source)
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
        else:
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
    except ConfigError as error:
        print(f"Ошибка конфигурации: {error}", file=sys.stderr)
        return 1
    except DocumentRegistrationError as error:
        print(f"Ошибка регистрации: {error}", file=sys.stderr)
        return 1
    except PdfIngestionError as error:
        print(f"Ошибка PDF ingestion: {error}", file=sys.stderr)
        return 1
    except ImageIngestionError as error:
        print(f"Ошибка image ingestion: {error}", file=sys.stderr)
        return 1

    if args.command == "register":
        logger.info("Документы зарегистрированы.")
        print(render_registration_report(registered_documents, manifest_path))
    elif args.command == "ingest-pdf":
        logger.info("PDF обработан.")
        print(render_pdf_ingestion_report(pdf_result))
    elif args.command == "ingest-image":
        logger.info("Изображение обработано.")
        print(render_image_ingestion_report(image_result))
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
    ingest_pdf_parser = subparsers.add_parser(
        "ingest-pdf",
        help="Читает PDF и сохраняет постраничный manifest.",
    )
    ingest_pdf_parser.add_argument(
        "source",
        help="Путь к PDF или document_id зарегистрированного PDF.",
    )
    ingest_image_parser = subparsers.add_parser(
        "ingest-image",
        help="Читает изображение и сохраняет page manifest.",
    )
    ingest_image_parser.add_argument(
        "source",
        help="Путь к PNG/JPG/JPEG или document_id зарегистрированного изображения.",
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


def render_pdf_ingestion_report(result: PdfIngestionResult | None) -> str:
    if result is None:
        return "PDF не обработан."

    lines = [
        f"Документ: {result.document_id}",
        f"Файл: {result.original_name}",
        f"Страниц: {result.page_count}",
        f"Страниц с текстовым слоем: {result.text_layer_page_count}",
        f"Page manifest: {result.page_manifest_path}",
    ]
    return "\n".join(lines)


def render_image_ingestion_report(result: ImageIngestionResult | None) -> str:
    if result is None:
        return "Изображение не обработано."

    page = result.pages[0]
    lines = [
        f"Документ: {result.document_id}",
        f"Файл: {result.original_name}",
        f"Размер: {page.width}x{page.height}",
        f"Режим: {page.image_mode}",
        f"Page manifest: {result.page_manifest_path}",
        f"Нормализованная копия: {result.normalized_image_path}",
    ]
    return "\n".join(lines)
