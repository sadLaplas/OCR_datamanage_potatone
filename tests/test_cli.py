import subprocess
import sys
from pathlib import Path

from tests.image_helpers import create_image
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
    assert "Конфигурация загружена." in result.stdout
    assert "Проект: docmanage-test" in result.stdout
    assert "Статус: конфигурация и окружение доступны." in result.stdout


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
    assert "Документы зарегистрированы." in result.stdout
    assert "Зарегистрировано документов: 2" in result.stdout
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
    assert "PDF обработан." in result.stdout
    assert "Страниц: 2" in result.stdout
    assert "Страниц с текстовым слоем: 1" in result.stdout
    assert "page_manifest.json" in result.stdout


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
    assert "Изображение обработано." in result.stdout
    assert "Размер: 220x330" in result.stdout
    assert "Page manifest:" in result.stdout
    assert "normalized_page.png" in result.stdout
