import json
import subprocess
import sys
from pathlib import Path

import torch

from docmanage.config import load_config, prepare_directories
from docmanage.ocr_dataset import build_charset_from_dataset
from docmanage.ocr_dataset_generation import generate_ocr_dataset
from docmanage.ocr_model import OcrCrnnModel, OcrModelConfig
from tests.image_helpers import create_document_like_image, create_image
from tests.pdf_helpers import create_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_starts_with_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "docmanage", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Все ок." in result.stdout
    assert "Проект: docmanage-test" in result.stdout
    assert "Файл manifest:" in result.stdout


def test_cli_registers_documents(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    first_path = tmp_path / "scan.jpg"
    second_path = tmp_path / "table.xlsx"
    first_path.write_bytes(b"image-data")
    second_path.write_bytes(b"sheet-data")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "register",
            str(first_path),
            str(second_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Файлы добавлены." in result.stdout
    assert "Добавил файлов: 2" in result.stdout
    assert "image" in result.stdout
    assert "spreadsheet" in result.stdout


def test_cli_ingests_pdf(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    pdf_path = create_pdf(tmp_path / "sample.pdf", ["First", None])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "ingest-pdf",
            str(pdf_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PDF готов." in result.stdout
    assert "Страниц: 2" in result.stdout
    assert "Страниц с текстом: 1" in result.stdout
    assert "Сохранил страницы:" in result.stdout


def test_cli_ingests_image(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    image_path = create_image(tmp_path / "scan.png", size=(220, 330), mode="L", color=200)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "ingest-image",
            str(image_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Картинка готова." in result.stdout
    assert "Размер: 220x330" in result.stdout
    assert "Сохранил manifest:" in result.stdout
    assert "Сохранил копию:" in result.stdout


def test_cli_preprocesses_image(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    image_path = create_document_like_image(tmp_path / "scan.png")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "preprocess-image",
            str(image_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Обработка завершена." in result.stdout
    assert "Шаги:" in result.stdout
    assert "Сохранил результат:" in result.stdout
    assert "Сохранил метаданные:" in result.stdout


def test_cli_extracts_lines_from_page_image(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    image_path = create_document_like_image(tmp_path / "page.png")
    output_dir = tmp_path / "lines"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "extract-lines",
            str(image_path),
            "--output",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Строки выделены." in result.stdout
    assert "Открыл страницу" in result.stdout
    assert "Ищу строки" in result.stdout
    assert "Нашел 8 строк" in result.stdout
    assert "Сохранил строки сюда:" in result.stdout
    assert "Сохранил manifest:" in result.stdout
    assert (output_dir / "line_manifest.json").exists()
    assert (output_dir / "lines_preview.png").exists()


def test_cli_runs_page_ocr_from_line_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path, checkpoint_path = prepare_page_ocr_cli_files(tmp_path, config_path)
    output_dir = tmp_path / "page_text"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "ocr-page-lines",
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_dir),
            "--no-progress",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Страница распознана." in result.stdout
    assert "Загружаю manifest" in result.stdout
    assert "Нашел 2 строк" in result.stdout
    assert "Загружаю модель" in result.stdout
    assert "Распознаю строки" in result.stdout
    assert "Сохранил результат:" in result.stdout
    assert "Сохранил текст:" in result.stdout
    assert (output_dir / "page_text.json").exists()
    assert (output_dir / "page_text.txt").exists()


def test_cli_generates_ocr_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "ocr_dataset"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "generate-ocr-data",
            "--demo",
            "--output",
            str(output_dir),
            "--seed",
            "5",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Готово." in result.stdout
    assert "Всего примеров: 12" in result.stdout
    assert "Train:" in result.stdout
    assert "Val:" in result.stdout
    assert "Аннотации train:" in result.stdout


def test_cli_checks_ocr_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "ocr_dataset"

    generate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "generate-ocr-data",
            "--demo",
            "--output",
            str(output_dir),
            "--seed",
            "7",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert generate_result.returncode == 0

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "check-ocr-model",
            "--dataset",
            str(output_dir),
            "--batch-size",
            "3",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Проверка готова." in result.stdout
    assert "Модель создалась." in result.stdout
    assert "Входной батч:" in result.stdout
    assert "Выход модели:" in result.stdout
    assert "Прогон батча прошел нормально." in result.stdout
    assert "Проверка прошла." in result.stdout


def test_cli_trains_ocr_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "ocr_dataset"
    output_dir = tmp_path / "ocr_training"

    generate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "generate-ocr-data",
            "--demo",
            "--output",
            str(dataset_dir),
            "--seed",
            "8",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert generate_result.returncode == 0

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "train-ocr",
            "--dataset",
            str(dataset_dir),
            "--output",
            str(output_dir),
            "--demo",
            "--batch-size",
            "4",
            "--seed",
            "8",
            "--no-progress",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Обучение завершено." in result.stdout
    assert "Начинаю обучение." in result.stdout
    assert "Эпоха 1 из 1" in result.stdout
    assert "Train loss:" in result.stdout
    assert "Val loss:" in result.stdout
    assert "Ошибка по символам:" in result.stdout
    assert "Точное совпадение:" in result.stdout
    assert "Сохранил лучший чекпоинт:" in result.stdout
    assert "Сохранил матрицу ошибок:" in result.stdout
    assert "Сохранил примеры:" in result.stdout
    assert "Сохранил историю:" in result.stdout


def test_cli_runs_line_ocr_inference(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "ocr_dataset"
    training_dir = tmp_path / "ocr_training"
    result_path = tmp_path / "line_result.json"

    generate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "generate-ocr-data",
            "--demo",
            "--output",
            str(dataset_dir),
            "--seed",
            "10",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert generate_result.returncode == 0

    train_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "train-ocr",
            "--dataset",
            str(dataset_dir),
            "--output",
            str(training_dir),
            "--demo",
            "--batch-size",
            "4",
            "--no-progress",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert train_result.returncode == 0

    image_path = next((dataset_dir / "images" / "train").glob("*.png"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "ocr-infer-line",
            "--image",
            str(image_path),
            "--checkpoint",
            str(training_dir / "best_model.pt"),
            "--output",
            str(result_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Распознавание завершено." in result.stdout
    assert "Загрузил модель." in result.stdout
    assert "Открыл изображение:" in result.stdout
    assert "Распознанный текст:" in result.stdout
    assert "Результат сохранен:" in result.stdout
    assert result_path.exists()


def test_cli_checks_line_folder(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "ocr_dataset"
    training_dir = tmp_path / "ocr_training"
    report_path = tmp_path / "line_report.json"

    generate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "generate-ocr-data",
            "--demo",
            "--mode",
            "realistic",
            "--output",
            str(dataset_dir),
            "--seed",
            "15",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert generate_result.returncode == 0

    train_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "train-ocr",
            "--dataset",
            str(dataset_dir),
            "--output",
            str(training_dir),
            "--demo",
            "--batch-size",
            "4",
            "--no-progress",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert train_result.returncode == 0

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "docmanage",
            "--config",
            str(config_path),
            "check-ocr-lines",
            "--images",
            str(dataset_dir / "images" / "val"),
            "--checkpoint",
            str(training_dir / "best_model.pt"),
            "--output",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Проверка строк завершена." in result.stdout
    assert "Проверяю папку со строками." in result.stdout
    assert "Нашел изображений:" in result.stdout
    assert "Разметки нет, сохраняю только предсказания." in result.stdout
    assert "Отчет сохранен:" in result.stdout
    assert report_path.exists()


def prepare_page_ocr_cli_files(
    tmp_path: Path,
    config_path: Path,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    prepare_directories(config)
    dataset_dir = tmp_path / "ocr_dataset_for_page"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=21)
    image_paths = sorted((dataset_dir / "images" / "train").glob("*.png"))[:2]

    manifest_path = tmp_path / "line_manifest.json"
    manifest_payload = {
        "source_image": str(tmp_path / "page.png"),
        "line_count": 2,
        "lines": [
            {
                "line_id": "line_0001",
                "image_path": str(image_paths[0]),
                "bbox": [0, 10, 120, 30],
            },
            {
                "line_id": "line_0002",
                "image_path": str(image_paths[1]),
                "bbox": [0, 40, 120, 60],
            },
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    charset = build_charset_from_dataset(dataset_dir)
    model_config = OcrModelConfig()
    model = OcrCrnnModel(model_config, num_classes=charset.size)
    checkpoint_path = tmp_path / "page_ocr_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "charset": list(charset.characters),
            "blank_index": charset.blank_index,
            "model_config": {
                "input_channels": model_config.input_channels,
                "conv_channels": list(model_config.conv_channels),
                "lstm_hidden_size": model_config.lstm_hidden_size,
                "lstm_num_layers": model_config.lstm_num_layers,
                "dropout": model_config.dropout,
                "image_height": model_config.image_height,
            },
            "best_val_loss": 0.0,
        },
        checkpoint_path,
    )

    return manifest_path, checkpoint_path
