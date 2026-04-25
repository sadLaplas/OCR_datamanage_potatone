from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .ocr_dataset import OcrCharset, OcrDatasetError, load_line_image
from .ocr_evaluation import OcrEvaluationError, greedy_decode_logits
from .ocr_model import OcrCrnnModel, OcrModelConfig, OcrModelError
from .ocr_training import resolve_device


class OcrInferenceError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class OcrInferenceResult:
    image_path: Path
    checkpoint_path: Path
    prediction: str
    output_path: Path | None
    device: str


def run_ocr_line_inference(
    image_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    device_name: str = "cpu",
) -> OcrInferenceResult:
    actual_image_path = resolve_existing_file(
        image_path,
        missing_message="Файл изображения не найден",
        directory_message="Путь к изображению указывает на папку",
    )
    actual_checkpoint_path = resolve_existing_file(
        checkpoint_path,
        missing_message="Не нашел checkpoint",
        directory_message="Путь к checkpoint указывает на папку",
    )
    device = resolve_inference_device(device_name)
    checkpoint = load_checkpoint(actual_checkpoint_path, device)
    charset = load_charset_from_checkpoint(checkpoint)
    model_config = load_model_config_from_checkpoint(checkpoint)
    model = load_model_from_checkpoint(checkpoint, model_config, charset, device)

    image_tensor = prepare_line_image(actual_image_path, model_config, device)
    image_widths = torch.tensor([image_tensor.shape[-1]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits, output_lengths = model(image_tensor, image_widths)

    try:
        prediction = greedy_decode_logits(logits, output_lengths, charset)[0]
    except OcrEvaluationError as error:
        raise OcrInferenceError(str(error)) from error

    actual_output_path = save_inference_result(
        output_path=output_path,
        image_path=actual_image_path,
        checkpoint_path=actual_checkpoint_path,
        prediction=prediction,
    )

    return OcrInferenceResult(
        image_path=actual_image_path,
        checkpoint_path=actual_checkpoint_path,
        prediction=prediction,
        output_path=actual_output_path,
        device=device.type,
    )


def resolve_existing_file(
    path: str | Path,
    missing_message: str,
    directory_message: str,
) -> Path:
    actual_path = Path(path).expanduser().resolve()

    if not actual_path.exists():
        raise OcrInferenceError(f"{missing_message}: {actual_path}")
    if not actual_path.is_file():
        raise OcrInferenceError(f"{directory_message}: {actual_path}")
    return actual_path


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except (OSError, RuntimeError, ValueError) as error:
        raise OcrInferenceError("Не удалось загрузить checkpoint.") from error

    if not isinstance(checkpoint, dict):
        raise OcrInferenceError("Checkpoint выглядит неправильно.")
    return checkpoint


def load_charset_from_checkpoint(checkpoint: dict[str, object]) -> OcrCharset:
    raw_charset = checkpoint.get("charset")
    blank_index = checkpoint.get("blank_index")

    if not isinstance(raw_charset, list) or not raw_charset:
        raise OcrInferenceError("В checkpoint нет словаря.")
    if blank_index != 0:
        raise OcrInferenceError("Checkpoint использует другой пустой класс.")
    if not all(isinstance(char, str) and char for char in raw_charset):
        raise OcrInferenceError("Словарь в checkpoint поврежден.")

    try:
        return OcrCharset(tuple(raw_charset))
    except OcrDatasetError as error:
        raise OcrInferenceError(str(error)) from error


def load_model_config_from_checkpoint(checkpoint: dict[str, object]) -> OcrModelConfig:
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise OcrInferenceError("В checkpoint нет настроек модели.")

    try:
        conv_channels = tuple(int(value) for value in raw_config["conv_channels"])
        return OcrModelConfig(
            input_channels=int(raw_config["input_channels"]),
            conv_channels=conv_channels,
            lstm_hidden_size=int(raw_config["lstm_hidden_size"]),
            lstm_num_layers=int(raw_config["lstm_num_layers"]),
            dropout=float(raw_config["dropout"]),
            image_height=int(raw_config["image_height"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OcrInferenceError("Настройки модели в checkpoint повреждены.") from error


def load_model_from_checkpoint(
    checkpoint: dict[str, object],
    model_config: OcrModelConfig,
    charset: OcrCharset,
    device: torch.device,
) -> OcrCrnnModel:
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise OcrInferenceError("В checkpoint нет весов модели.")

    try:
        model = OcrCrnnModel(model_config, num_classes=charset.size)
        model.load_state_dict(state_dict)
    except (OcrModelError, RuntimeError) as error:
        raise OcrInferenceError("Checkpoint не подходит к этой модели.") from error

    return model.to(device)


def prepare_line_image(
    image_path: Path,
    model_config: OcrModelConfig,
    device: torch.device,
) -> torch.Tensor:
    try:
        line_image = load_line_image(image_path, model_config.image_height)
    except OcrDatasetError as error:
        raise OcrInferenceError("Не получилось открыть изображение.") from error

    # Модель ждет форму [B, C, H, W], поэтому добавляем размер батча.
    return line_image.unsqueeze(0).to(device)


def save_inference_result(
    output_path: str | Path | None,
    image_path: Path,
    checkpoint_path: Path,
    prediction: str,
) -> Path | None:
    if output_path is None:
        return None

    actual_output_path = Path(output_path).expanduser().resolve()
    payload = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "prediction": prediction,
        "confidence": None,
    }

    try:
        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        actual_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise OcrInferenceError("Не удалось сохранить результат.") from error

    return actual_output_path


def resolve_inference_device(device_name: str) -> torch.device:
    try:
        return resolve_device(device_name)
    except ValueError as error:
        raise OcrInferenceError(str(error)) from error
