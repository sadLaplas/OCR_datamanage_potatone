from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_FIELDS = (
    "project_name",
    "run_mode",
    "data_dir",
    "artifacts_dir",
    "temp_dir",
    "log_level",
)
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class AppConfig:
    project_name: str
    run_mode: str
    data_dir: Path
    artifacts_dir: Path
    temp_dir: Path
    log_level: str
    config_path: Path

    @property
    def directories(self) -> dict[str, Path]:
        return {
            "data": self.data_dir,
            "artifacts": self.artifacts_dir,
            "temp": self.temp_dir,
        }

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_dir / "manifest.json"


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).expanduser()

    if not path.exists():
        raise ConfigError(f"Файл конфигурации не найден: {path}")
    if not path.is_file():
        raise ConfigError(f"Путь к конфигурации должен указывать на файл: {path}")

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"YAML поврежден или не читается: {error}") from error
    except OSError as error:
        raise ConfigError(f"Не удалось прочитать файл конфигурации: {error}") from error

    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ConfigError("Конфигурация должна быть YAML-словарем.")

    missing_fields = [field for field in REQUIRED_FIELDS if field not in raw_config]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ConfigError(f"В конфигурации отсутствуют обязательные поля: {missing}")

    config_dir = path.resolve().parent
    project_name = _read_text_field(raw_config, "project_name")
    run_mode = _read_text_field(raw_config, "run_mode")
    log_level = _read_text_field(raw_config, "log_level").upper()

    if log_level not in VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigError(
            f"Недопустимый уровень логирования '{log_level}'. Ожидалось: {allowed}"
        )

    return AppConfig(
        project_name=project_name,
        run_mode=run_mode,
        data_dir=_resolve_directory(raw_config, "data_dir", config_dir),
        artifacts_dir=_resolve_directory(raw_config, "artifacts_dir", config_dir),
        temp_dir=_resolve_directory(raw_config, "temp_dir", config_dir),
        log_level=log_level,
        config_path=path.resolve(),
    )


def prepare_directories(config: AppConfig) -> dict[str, str]:
    statuses: dict[str, str] = {}

    for name, path in config.directories.items():
        existed_before = path.exists()

        if existed_before and not path.is_dir():
            raise ConfigError(f"Путь '{name}' должен указывать на директорию: {path}")

        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ConfigError(
                f"Не удалось подготовить директорию '{name}': {path}"
            ) from error

        statuses[name] = "готова" if existed_before else "создана"

    return statuses


def _read_text_field(raw_config: dict[str, object], field_name: str) -> str:
    value = raw_config.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Поле '{field_name}' должно быть непустой строкой.")

    return value.strip()


def _resolve_directory(
    raw_config: dict[str, object], field_name: str, config_dir: Path
) -> Path:
    raw_value = _read_text_field(raw_config, field_name)
    path = Path(raw_value).expanduser()

    if not path.is_absolute():
        path = (config_dir / path).resolve()
    else:
        path = path.resolve()

    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise ConfigError(
            f"Некорректный путь в поле '{field_name}': родительский путь не является директорией."
        )

    return path
