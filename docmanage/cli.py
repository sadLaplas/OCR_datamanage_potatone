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
from .ocr_dataset_generation import (
    OcrDatasetError,
    OcrDatasetResult,
    generate_ocr_dataset,
)
from .ocr_model import OcrModelCheckResult, OcrModelError, run_ocr_model_check
from .ocr_inference import (
    OcrInferenceError,
    OcrInferenceResult,
    run_ocr_line_inference,
)
from .ocr_line_check import (
    LineCheckResult,
    OcrLineCheckError,
    run_line_folder_check,
)
from .ocr_training import (
    OcrTrainingConfig,
    OcrTrainingError,
    OcrTrainingResult,
    run_ocr_training,
)
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
            dataset_result = None
            model_check_result = None
            training_result = None
            inference_result = None
            line_check_result = None
        elif args.command == "ingest-pdf":
            pdf_result = ingest_pdf(config, args.source)
            registered_documents = []
            manifest_path = config.manifest_path
            image_result = None
            preprocess_result = None
            dataset_result = None
            model_check_result = None
            training_result = None
            inference_result = None
            line_check_result = None
        elif args.command == "ingest-image":
            image_result = ingest_image(config, args.source)
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            preprocess_result = None
            dataset_result = None
            model_check_result = None
            training_result = None
            inference_result = None
            line_check_result = None
        elif args.command == "preprocess-image":
            preprocess_result = preprocess_image(config, args.source)
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
            dataset_result = None
            model_check_result = None
            training_result = None
            inference_result = None
            line_check_result = None
        elif args.command == "generate-ocr-data":
            dataset_result = generate_ocr_dataset(
                config,
                output_dir=args.output,
                count=args.count,
                val_ratio=args.val_ratio,
                seed=args.seed,
                demo=args.demo,
                mode=args.mode,
            )
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
            preprocess_result = None
            model_check_result = None
            training_result = None
            inference_result = None
            line_check_result = None
        elif args.command == "check-ocr-model":
            model_check_result = run_ocr_model_check(
                config,
                dataset_path=args.dataset,
                split=args.split,
                batch_size=args.batch_size,
            )
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
            preprocess_result = None
            dataset_result = None
            training_result = None
            inference_result = None
            line_check_result = None
        elif args.command == "train-ocr":
            training_result = run_ocr_training(
                config,
                dataset_path=args.dataset,
                output_dir=args.output,
                training_config=OcrTrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    demo=args.demo,
                    seed=args.seed,
                    example_limit=args.examples,
                    show_progress=not args.no_progress,
                ),
            )
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
            preprocess_result = None
            dataset_result = None
            model_check_result = None
            inference_result = None
            line_check_result = None
        elif args.command == "ocr-infer-line":
            inference_result = run_ocr_line_inference(
                image_path=args.image,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                device_name=args.device,
            )
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
            preprocess_result = None
            dataset_result = None
            model_check_result = None
            training_result = None
            line_check_result = None
        elif args.command == "check-ocr-lines":
            line_check_result = run_line_folder_check(
                images_dir=args.images,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                ground_truth_path=args.ground_truth,
                device_name=args.device,
            )
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
            preprocess_result = None
            dataset_result = None
            model_check_result = None
            training_result = None
            inference_result = None
        else:
            registered_documents = []
            manifest_path = config.manifest_path
            pdf_result = None
            image_result = None
            preprocess_result = None
            dataset_result = None
            model_check_result = None
            training_result = None
            inference_result = None
            line_check_result = None
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
    except OcrDatasetError as error:
        print(f"Не получилось собрать датасет: {error}", file=sys.stderr)
        return 1
    except OcrModelError as error:
        print(f"Не получилось проверить модель: {error}", file=sys.stderr)
        return 1
    except OcrTrainingError as error:
        print(f"Не получилось запустить обучение: {error}", file=sys.stderr)
        return 1
    except OcrInferenceError as error:
        print(f"Не получилось распознать строку: {error}", file=sys.stderr)
        return 1
    except OcrLineCheckError as error:
        print(f"Не получилось проверить строки: {error}", file=sys.stderr)
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
    elif args.command == "generate-ocr-data":
        logger.info("Готово.")
        print(render_dataset_report(dataset_result))
    elif args.command == "check-ocr-model":
        logger.info("Проверка готова.")
        print(render_model_check_report(model_check_result))
    elif args.command == "train-ocr":
        logger.info("Обучение завершено.")
        print(render_training_report(training_result))
    elif args.command == "ocr-infer-line":
        logger.info("Распознавание завершено.")
        print(render_inference_report(inference_result))
    elif args.command == "check-ocr-lines":
        logger.info("Проверка строк завершена.")
        print(render_line_check_report(line_check_result))
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
    generate_dataset_parser = subparsers.add_parser(
        "generate-ocr-data",
        help="Делает маленький OCR датасет из строк.",
    )
    generate_dataset_parser.add_argument(
        "--output",
        default=None,
        help="Куда сохранить датасет.",
    )
    generate_dataset_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Сколько примеров",
    )
    generate_dataset_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Доля для val.",
    )
    generate_dataset_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Сид для генерации.",
    )
    generate_dataset_parser.add_argument(
        "--demo",
        action="store_true",
        help="Сделать маленький demo",
    )
    generate_dataset_parser.add_argument(
        "--mode",
        choices=("clean", "realistic"),
        default="clean",
        help="Режим генерации строк.",
    )
    check_model_parser = subparsers.add_parser(
        "check-ocr-model",
        help="Проверить OCR-модель на одном батче.",
    )
    check_model_parser.add_argument(
        "--dataset",
        default=None,
        help="Путь к OCR датасету.",
    )
    check_model_parser.add_argument(
        "--split",
        default="train",
        choices=("train", "val"),
        help="Какой split взять.",
    )
    check_model_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Размер батча для проверки.",
    )
    train_parser = subparsers.add_parser(
        "train-ocr",
        help="Обучить OCR-модель.",
    )
    train_parser.add_argument(
        "--dataset",
        default=None,
        help="Путь к OCR датасету.",
    )
    train_parser.add_argument(
        "--output",
        default=None,
        help="Куда сохранить чекпоинт и историю.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Сколько эпох запускать.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Размер батча.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    train_parser.add_argument(
        "--device",
        default="cpu",
        help="Устройство: cpu, cuda, mps или auto.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для обучения.",
    )
    train_parser.add_argument(
        "--demo",
        action="store_true",
        help="Сделать короткий demo-run.",
    )
    train_parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Сколько примеров сохранить после проверки.",
    )
    train_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Не показывать progress bar.",
    )
    infer_parser = subparsers.add_parser(
        "ocr-infer-line",
        help="Распознать одно изображение строки.",
    )
    infer_parser.add_argument(
        "--image",
        required=True,
        help="Путь к изображению строки.",
    )
    infer_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Путь к checkpoint.",
    )
    infer_parser.add_argument(
        "--output",
        default=None,
        help="Куда сохранить JSON с результатом.",
    )
    infer_parser.add_argument(
        "--device",
        default="cpu",
        help="Устройство: cpu, cuda, mps или auto.",
    )
    check_lines_parser = subparsers.add_parser(
        "check-ocr-lines",
        help="Проверить папку изображений строк.",
    )
    check_lines_parser.add_argument(
        "--images",
        required=True,
        help="Папка с изображениями строк.",
    )
    check_lines_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Путь к checkpoint.",
    )
    check_lines_parser.add_argument(
        "--ground-truth",
        default=None,
        help="JSONL с правильными строками.",
    )
    check_lines_parser.add_argument(
        "--output",
        required=True,
        help="Куда сохранить отчет JSON.",
    )
    check_lines_parser.add_argument(
        "--device",
        default="cpu",
        help="Устройство: cpu, cuda, mps или auto.",
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


def render_dataset_report(result: OcrDatasetResult | None) -> str:
    if result is None:
        return "Датасет пока не готов."

    lines = [
        f"Папка: {result.output_dir}",
        f"Всего примеров: {result.total_samples}",
        f"Train: {result.train_samples}",
        f"Val: {result.val_samples}",
        f"Аннотации train: {result.train_annotations_path}",
        f"Аннотации val: {result.val_annotations_path}",
        f"Метаданные: {result.metadata_path}",
        f"Режим генерации: {result.generation_mode}",
    ]

    if result.fonts_used:
        lines.append(f"Шрифты: {', '.join(result.fonts_used)}")
    return "\n".join(lines)


def render_model_check_report(result: OcrModelCheckResult | None) -> str:
    if result is None:
        return "Модель пока не проверена."

    return "\n".join(
        [
            "Модель создалась.",
            f"Папка с датасетом: {result.dataset_path}",
            f"Беру split: {result.split}",
            f"Входной батч: {result.batch_shape}",
            f"Выход модели: {result.logits_shape}",
            f"Длины выхода: {result.output_lengths}",
            f"Число классов: {result.num_classes}",
            "Прогон батча прошел нормально.",
            "Проверка прошла.",
        ]
    )


def render_training_report(result: OcrTrainingResult | None) -> str:
    if result is None:
        return "Обучение пока не запустилось."

    lines = [
        "Начинаю обучение.",
        f"Папка с датасетом: {result.dataset_path}",
        f"Папка с результатом: {result.output_dir}",
        f"Устройство: {result.device}",
    ]

    for epoch in result.history:
        line = (
            f"Эпоха {epoch.epoch} из {result.epochs_ran}"
            f" | Train loss: {epoch.train_loss:.4f}"
            f" | Val loss: {epoch.val_loss:.4f}"
            f" | Ошибка по символам: {epoch.val_cer:.4f}"
            f" | Точное совпадение: {epoch.exact_match:.1%}"
        )
        if epoch.was_best:
            line += " | сохранил лучший чекпоинт"
        lines.append(line)

    last_epoch = result.history[-1]
    lines.append(f"Лучший val loss: {result.best_val_loss:.4f}")
    lines.append(f"Сохранил лучший чекпоинт: {result.best_checkpoint_path}")
    lines.append(f"Сохранил матрицу ошибок: {last_epoch.error_matrix_path}")
    lines.append(f"Сохранил примеры: {last_epoch.examples_path}")
    lines.append(f"Сохранил историю: {result.history_path}")
    lines.append("Обучение завершено.")
    return "\n".join(lines)


def render_inference_report(result: OcrInferenceResult | None) -> str:
    if result is None:
        return "Распознавание пока не запустилось."

    lines = [
        "Загрузил модель.",
        f"Checkpoint: {result.checkpoint_path}",
        f"Открыл изображение: {result.image_path}",
        f"Устройство: {result.device}",
        f"Распознанный текст: {result.prediction}",
    ]
    if result.output_path is not None:
        lines.append(f"Результат сохранен: {result.output_path}")
    return "\n".join(lines)


def render_line_check_report(result: LineCheckResult | None) -> str:
    if result is None:
        return "Проверка строк пока не запустилась."

    lines = [
        "Проверяю папку со строками.",
        f"Папка: {result.images_dir}",
        f"Нашел изображений: {result.image_count}",
    ]

    if result.has_ground_truth:
        lines.append("Разметка найдена, считаю ошибки.")
        lines.append(f"Средняя ошибка по символам: {result.average_cer:.4f}")
        lines.append(f"Точных строк: {result.exact_match_rate:.1%}")
    else:
        lines.append("Разметки нет, сохраняю только предсказания.")

    lines.append(f"Отчет сохранен: {result.report_path}")
    return "\n".join(lines)
