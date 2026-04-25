import json
from pathlib import Path

import pytest
import torch

from docmanage.config import load_config, prepare_directories
from docmanage.ocr_dataset import build_charset_from_dataset
from docmanage.ocr_dataset_generation import generate_ocr_dataset
from docmanage.ocr_inference import (
    OcrInferenceError,
    run_ocr_line_inference,
)
from docmanage.ocr_model import OcrCrnnModel, OcrModelConfig


def test_inference_runs_on_one_line_image(tmp_path: Path) -> None:
    image_path, checkpoint_path = prepare_image_and_checkpoint(tmp_path)

    result = run_ocr_line_inference(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "result.json",
    )

    assert result.image_path == image_path.resolve()
    assert result.checkpoint_path == checkpoint_path.resolve()
    assert isinstance(result.prediction, str)
    assert result.output_path is not None
    assert result.output_path.exists()

    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["image_path"] == str(image_path.resolve())
    assert payload["checkpoint_path"] == str(checkpoint_path.resolve())
    assert payload["prediction"] == result.prediction
    assert payload["confidence"] is None


def test_inference_uses_greedy_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path, checkpoint_path = prepare_image_and_checkpoint(tmp_path)

    def fake_decode(*_args):
        return ["готово"]

    monkeypatch.setattr("docmanage.ocr_inference.greedy_decode_logits", fake_decode)

    result = run_ocr_line_inference(image_path=image_path, checkpoint_path=checkpoint_path)

    assert result.prediction == "готово"


def test_inference_raises_for_missing_checkpoint(tmp_path: Path) -> None:
    image_path, _ = prepare_image_and_checkpoint(tmp_path)

    with pytest.raises(OcrInferenceError, match="Не нашел checkpoint"):
        run_ocr_line_inference(
            image_path=image_path,
            checkpoint_path=tmp_path / "missing.pt",
        )


def test_inference_raises_for_missing_image(tmp_path: Path) -> None:
    _, checkpoint_path = prepare_image_and_checkpoint(tmp_path)

    with pytest.raises(OcrInferenceError, match="Файл изображения не найден"):
        run_ocr_line_inference(
            image_path=tmp_path / "missing.png",
            checkpoint_path=checkpoint_path,
        )


def test_inference_raises_for_bad_checkpoint(tmp_path: Path) -> None:
    image_path, _ = prepare_image_and_checkpoint(tmp_path)
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"wrong": "data"}, checkpoint_path)

    with pytest.raises(OcrInferenceError, match="В checkpoint нет словаря"):
        run_ocr_line_inference(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
        )


def prepare_image_and_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=17)
    image_path = next((dataset_dir / "images" / "train").glob("*.png"))
    charset = build_charset_from_dataset(dataset_dir)
    model_config = OcrModelConfig()
    model = OcrCrnnModel(model_config, num_classes=charset.size)
    checkpoint_path = tmp_path / "best_model.pt"
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
    return image_path, checkpoint_path


def prepare_test_config(tmp_path: Path):
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
    prepare_directories(config)
    return config
