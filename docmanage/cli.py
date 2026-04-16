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
from .image_preprocessing import (
    ImagePreprocessError,
    ImagePreprocessResult,
    preprocess_image,
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
            preprocess_result = None
        elif args.command == "ingest-pdf":
            pdf_result = ingest_pdf(config, args.source)
            registered_documents = []
            manifest_path = config.manifest_path
            image_result = None
            preprocess_result = None
        elif args.command == "ingest-image":
            image_result = ingest_image(config, args.source)
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            preprocess_result = None
        elif args.command == "preprocess-image":
            preprocess_result = preprocess_image(config, args.source)
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
        else:
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
            preprocess_result = None
    except ConfigError as error:
        print(f"Проблема с конфигом: {error}", file=sys.stderr)
        return 1
    except DocumentRegistrationError as error:
        print(f"Не получилось добавить файл: {error}", file=sys.stderr)
        return 1
    except PdfIngestionError as error:
        print(f"Не получилось прочитать PDF: {error}", file=sys.stderr)
        return 1
    except ImageIngestionError as error:
        print(f"Не получилось обработать картинку: {error}", file=sys.stderr)
        return 1
    except ImagePreprocessError as error:
        print(f"Не получилось подготовить картинку: {error}", file=sys.stderr)
        return 1

    if args.command == "register":
        logger.info("Файлы добавлены.")
        print(render_registration_report(registered_documents, manifest_path))
    elif args.command == "ingest-pdf":
        logger.info("PDF готов.")
        print(render_pdf_ingestion_report(pdf_result))
    elif args.command == "ingest-image":
        logger.info("Картинка готова.")
        print(render_image_ingestion_report(image_result))
    elif args.command == "preprocess-image":
        logger.info("Обработка завершена.")
        print(render_preprocess_report(preprocess_result))
    else:
        logger.info("Все ок.")
        print(render_report(config, directory_statuses))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docmanage",
        description="Показывает состояние проекта и работает с файлами.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Путь к YAML-конфигурации.",
    )
    subparsers = parser.add_subparsers(dest="command")
    register_parser = subparsers.add_parser(
        "register",
        help="Добавляет файлы в manifest.",
    )
    register_parser.add_argument(
        "paths",
        nargs="*",
        help="Пути к файлам для регистрации.",
    )
    ingest_pdf_parser = subparsers.add_parser(
        "ingest-pdf",
        help="Читает PDF и сохраняет страницы.",
    )
    ingest_pdf_parser.add_argument(
        "source",
        help="Путь к PDF или document_id зарегистрированного PDF.",
    )
    ingest_image_parser = subparsers.add_parser(
        "ingest-image",
        help="Читает картинку и сохраняет страницу.",
    )
    ingest_image_parser.add_argument(
        "source",
        help="Путь к PNG/JPG/JPEG или document_id зарегистрированного изображения.",
    )
    preprocess_image_parser = subparsers.add_parser(
        "preprocess-image",
        help="Готовит картинку и сохраняет результат.",
    )
    preprocess_image_parser.add_argument(
        "source",
        help="Путь к картинке или document_id зарегистрированного изображения.",
    )
    return parser


def render_report(config: AppConfig, directory_statuses: dict[str, str]) -> str:
    lines = [
        f"Проект: {config.project_name}",
        f"Режим: {config.run_mode}",
        f"Конфиг: {config.config_path}",
        f"Текущая папка: {Path.cwd()}",
        f"Python: {platform.python_version()}",
        f"Файл manifest: {config.manifest_path}",
        "Папки:",
    ]

    for name, path in config.directories.items():
        status = directory_statuses[name]
        lines.append(f"- {name}: {path} ({status})")

    lines.append("Все ок.")
    return "\n".join(lines)


def render_registration_report(
    registered_documents: list[RegisteredDocument], manifest_path: Path
) -> str:
    lines = [
        f"Файл manifest: {manifest_path}",
        f"Добавил файлов: {len(registered_documents)}",
    ]

    for document in registered_documents:
        lines.append(
            f"- {document.document_id} | {document.document_kind} | {document.original_name}"
        )

    return "\n".join(lines)


def render_pdf_ingestion_report(result: PdfIngestionResult | None) -> str:
    if result is None:
        return "PDF пока не готов."

    lines = [
        f"Документ: {result.document_id}",
        f"Файл: {result.original_name}",
        f"Страниц: {result.page_count}",
        f"Страниц с текстом: {result.text_layer_page_count}",
        f"Сохранил страницы: {result.page_manifest_path}",
    ]
    return "\n".join(lines)


def render_image_ingestion_report(result: ImageIngestionResult | None) -> str:
    if result is None:
        return "Картинка пока не готова."

    page = result.pages[0]
    lines = [
        f"Документ: {result.document_id}",
        f"Файл: {result.original_name}",
        f"Размер: {page.width}x{page.height}",
        f"Режим: {page.image_mode}",
        f"Сохранил manifest: {result.page_manifest_path}",
        f"Сохранил копию: {result.normalized_image_path}",
    ]
    return "\n".join(lines)


def render_preprocess_report(result: ImagePreprocessResult | None) -> str:
    if result is None:
        return "Картинка пока не готова."

    metadata = result.metadata
    steps = ", ".join(metadata.steps) if metadata.steps else "без шагов"
    lines = [
        f"Документ: {result.document_id}",
        f"Файл: {result.original_name}",
        f"Шаги: {steps}",
        f"Размер: {metadata.width}x{metadata.height}",
    ]

    if metadata.deskew_applied:
        lines.append(f"Повернул на: {metadata.deskew_angle} град.")
    else:
        lines.append("Поворот не понадобился.")

    lines.append(f"Сохранил результат: {result.preprocessed_image_path}")
    lines.append(f"Сохранил метаданные: {result.metadata_path}")
    return "\n".join(lines)
