from pathlib import Path

import pytest

from docmanage.config import ConfigError, load_config, prepare_directories


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_reads_base_file() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    assert config.project_name == "docmanage"
    assert config.run_mode == "development"
    assert config.data_dir == (PROJECT_ROOT / "data").resolve()
    assert config.artifacts_dir == (PROJECT_ROOT / "artifacts").resolve()
    assert config.temp_dir == (PROJECT_ROOT / "tmp").resolve()
    assert config.log_level == "INFO"


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="не найден"):
        load_config(missing_path)


def test_load_config_raises_for_broken_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("project_name: [broken\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="YAML поврежден"):
        load_config(config_path)


def test_load_config_raises_for_missing_required_field(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="обязательные поля"):
        load_config(config_path)


def test_prepare_directories_creates_missing_paths(tmp_path: Path) -> None:
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

    config = load_config(config_path)
    statuses = prepare_directories(config)

    assert config.data_dir.is_dir()
    assert config.artifacts_dir.is_dir()
    assert config.temp_dir.is_dir()
    assert statuses == {
        "data": "создана",
        "artifacts": "создана",
        "temp": "создана",
    }
