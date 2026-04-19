from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from docmanage.config import load_config, prepare_directories
from docmanage.ocr_dataset import OcrLineDataset, build_charset_from_dataset, ocr_collate_fn
from docmanage.ocr_dataset_generation import generate_ocr_dataset
from docmanage.ocr_model import (
    OcrCrnnModel,
    OcrModelCheckResult,
    OcrModelConfig,
    OcrModelError,
    run_ocr_model_check,
)


def test_model_creates_and_runs_forward(tmp_path: Path) -> None:
    dataset_dir = prepare_dataset(tmp_path)
    charset = build_charset_from_dataset(dataset_dir)
    dataset = OcrLineDataset(dataset_dir, "train", charset)
    batch = ocr_collate_fn([dataset[0], dataset[1]])
    model = OcrCrnnModel(OcrModelConfig(), num_classes=charset.size)

    logits, output_lengths = model(batch["images"], batch["image_widths"])

    assert logits.ndim == 3
    assert logits.shape[0] == 2
    assert logits.shape[1] > 0
    assert logits.shape[2] == charset.size
    assert output_lengths.shape == (2,)
    assert torch.isfinite(logits).all()


def test_model_can_do_small_backward_step(tmp_path: Path) -> None:
    dataset_dir = prepare_dataset(tmp_path)
    charset = build_charset_from_dataset(dataset_dir)
    dataset = OcrLineDataset(dataset_dir, "train", charset)
    batch = ocr_collate_fn([dataset[0], dataset[1]])
    model = OcrCrnnModel(OcrModelConfig(), num_classes=charset.size)

    logits, _ = model(batch["images"], batch["image_widths"])
    value = logits.mean()
    value.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients


def test_model_check_works_with_dataset(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=13)

    result = run_ocr_model_check(config, dataset_path=dataset_dir, batch_size=3)

    assert isinstance(result, OcrModelCheckResult)
    assert result.batch_shape[0] == 3
    assert result.logits_shape[0] == 3
    assert result.logits_shape[1] > 0
    assert result.logits_shape[2] == result.num_classes


def test_model_check_raises_for_bad_batch_size(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=21)

    with pytest.raises(OcrModelError, match="Размер батча должен быть больше нуля"):
        run_ocr_model_check(config, dataset_path=dataset_dir, batch_size=0)


def test_model_raises_for_bad_config(tmp_path: Path) -> None:
    dataset_dir = prepare_dataset(tmp_path)
    charset = build_charset_from_dataset(dataset_dir)

    with pytest.raises(OcrModelError, match="LSTM"):
        OcrCrnnModel(
            OcrModelConfig(lstm_hidden_size=0),
            num_classes=charset.size,
        )


def test_dataloader_and_collate_make_expected_batch(tmp_path: Path) -> None:
    dataset_dir = prepare_dataset(tmp_path)
    charset = build_charset_from_dataset(dataset_dir)
    dataset = OcrLineDataset(dataset_dir, "train", charset)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=ocr_collate_fn)

    batch = next(iter(loader))

    assert batch["images"].ndim == 4
    assert batch["images"].shape[0] == 4
    assert batch["images"].shape[1] == 1
    assert batch["image_widths"].shape == (4,)
    assert batch["target_lengths"].shape == (4,)
    assert batch["targets"].ndim == 1


def prepare_dataset(tmp_path: Path) -> Path:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=3)
    return dataset_dir


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
