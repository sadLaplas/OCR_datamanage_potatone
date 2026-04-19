import subprocess
import sys
from pathlib import Path

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
    assert "Похоже, все работает." in result.stdout
