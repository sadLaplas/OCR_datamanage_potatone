import json
from pathlib import Path

import pytest
import torch

from docmanage.config import load_config, prepare_directories
from docmanage.ocr_dataset import build_charset_from_dataset
from docmanage.ocr_dataset_generation import generate_ocr_dataset
from docmanage.ocr_training import (
    OcrTrainingConfig,
    OcrTrainingError,
    compute_ctc_batch_loss,
    create_ctc_loss,
    create_ocr_dataloaders,
    resolve_device,
    run_ocr_training,
    train_one_epoch,
    validate_one_epoch,
)
from docmanage.ocr_model import OcrCrnnModel


def test_training_run_saves_checkpoint_and_history(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    output_dir = tmp_path / "ocr_training"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=7)

    result = run_ocr_training(
        config,
        dataset_path=dataset_dir,
        output_dir=output_dir,
        training_config=OcrTrainingConfig(
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=7,
            show_progress=False,
        ),
    )

    assert result.epochs_ran == 1
    assert len(result.history) == 1
    assert result.best_checkpoint_path.exists()
    assert result.history_path.exists()
    assert result.history[0].train_loss > 0
    assert result.history[0].val_loss > 0
    assert result.history[0].val_cer >= 0
    assert 0 <= result.history[0].exact_match <= 1
    assert result.history[0].error_matrix_path.exists()
    assert result.history[0].examples_path.exists()

    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert history["history"][0]["epoch"] == 1
    assert history["history"][0]["was_best"] is True
    assert "val_cer" in history["history"][0]
    assert "exact_match" in history["history"][0]
    assert "error_matrix_path" in history["history"][0]
    assert "examples_path" in history["history"][0]


def test_demo_run_uses_single_epoch(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=9)

    result = run_ocr_training(
        config,
        dataset_path=dataset_dir,
        output_dir=tmp_path / "ocr_training",
        training_config=OcrTrainingConfig(
            epochs=3,
            batch_size=4,
            device="cpu",
            demo=True,
            show_progress=False,
        ),
    )

    assert result.epochs_ran == 1
    assert len(result.history) == 1


def test_train_and_val_epoch_return_losses(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=5)

    charset = build_charset_from_dataset(dataset_dir)
    training_config = OcrTrainingConfig(
        epochs=1,
        batch_size=4,
        device="cpu",
        show_progress=False,
    )
    train_loader, val_loader = create_ocr_dataloaders(dataset_dir, charset, training_config)
    model = OcrCrnnModel(training_config.model_config, num_classes=charset.size)
    criterion = create_ctc_loss(charset)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    device = resolve_device("cpu")

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        max_batches=1,
        show_progress=False,
    )
    val_result = validate_one_epoch(
        model,
        val_loader,
        criterion,
        device,
        charset=charset,
        eval_dir=tmp_path / "eval",
        epoch=1,
        max_batches=1,
        show_progress=False,
    )

    assert train_loss > 0
    assert val_result.val_loss > 0
    assert val_result.val_cer >= 0
    assert val_result.error_matrix_path.exists()
    assert val_result.examples_path.exists()


def test_ctc_loss_is_computed_on_real_batch(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=11)

    charset = build_charset_from_dataset(dataset_dir)
    training_config = OcrTrainingConfig(epochs=1, batch_size=4, device="cpu")
    train_loader, _ = create_ocr_dataloaders(dataset_dir, charset, training_config)
    batch = next(iter(train_loader))
    model = OcrCrnnModel(training_config.model_config, num_classes=charset.size)
    criterion = create_ctc_loss(charset)

    loss = compute_ctc_batch_loss(model, batch, criterion, resolve_device("cpu"))

    assert float(loss.item()) > 0


def test_training_raises_for_bad_config(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    dataset_dir = tmp_path / "ocr_dataset"
    generate_ocr_dataset(config, output_dir=dataset_dir, demo=True, seed=3)

    with pytest.raises(OcrTrainingError, match="Размер батча должен быть больше нуля"):
        run_ocr_training(
            config,
            dataset_path=dataset_dir,
            output_dir=tmp_path / "ocr_training",
            training_config=OcrTrainingConfig(batch_size=0),
        )


def test_training_raises_for_missing_dataset(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)

    with pytest.raises(OcrTrainingError, match="Папка с датасетом не найдена"):
        run_ocr_training(
            config,
            dataset_path=tmp_path / "missing_dataset",
            output_dir=tmp_path / "ocr_training",
        )


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
