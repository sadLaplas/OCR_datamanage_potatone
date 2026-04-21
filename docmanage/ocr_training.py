from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import AppConfig
from .ocr_dataset import (
    OcrCharset,
    OcrDatasetError,
    OcrLineDataset,
    build_charset_from_dataset,
    ocr_collate_fn,
)
from .ocr_model import OcrCrnnModel, OcrModelConfig, OcrModelError


class OcrTrainingError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class OcrTrainingConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 0.001
    device: str = "cpu"
    num_workers: int = 0
    demo: bool = False
    seed: int = 42
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    model_config: OcrModelConfig = field(default_factory=OcrModelConfig)


@dataclass(slots=True, frozen=True)
class OcrTrainingEpoch:
    epoch: int
    train_loss: float
    val_loss: float
    was_best: bool


@dataclass(slots=True, frozen=True)
class OcrTrainingResult:
    dataset_path: Path
    output_dir: Path
    best_checkpoint_path: Path
    history_path: Path
    device: str
    best_val_loss: float
    epochs_ran: int
    history: tuple[OcrTrainingEpoch, ...]


def run_ocr_training(
    config: AppConfig,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    training_config: OcrTrainingConfig | None = None,
) -> OcrTrainingResult:
    actual_config = training_config or OcrTrainingConfig()
    validate_training_config(actual_config)

    dataset_dir = resolve_dataset_path(config, dataset_path)
    training_dir = resolve_training_output_dir(config, output_dir)
    prepare_training_directory(training_dir)

    set_training_seed(actual_config.seed)
    device = resolve_device(actual_config.device)

    try:
        charset = build_charset_from_dataset(dataset_dir)
        train_loader, val_loader = create_ocr_dataloaders(dataset_dir, charset, actual_config)
    except OcrDatasetError as error:
        raise OcrTrainingError(str(error)) from error

    try:
        model = OcrCrnnModel(actual_config.model_config, num_classes=charset.size).to(device)
    except OcrModelError as error:
        raise OcrTrainingError(str(error)) from error

    criterion = create_ctc_loss(charset)
    optimizer = Adam(model.parameters(), lr=actual_config.learning_rate)

    history: list[OcrTrainingEpoch] = []
    best_val_loss = float("inf")
    best_checkpoint_path = training_dir / "best_model.pt"
    history_path = training_dir / "history.json"

    epochs_to_run = 1 if actual_config.demo else actual_config.epochs
    train_limit = 2 if actual_config.demo else actual_config.max_train_batches
    val_limit = 1 if actual_config.demo else actual_config.max_val_batches

    for epoch_index in range(1, epochs_to_run + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=train_limit,
        )
        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=val_limit,
        )

        was_best = val_loss < best_val_loss
        if was_best:
            best_val_loss = val_loss
            save_best_checkpoint(
                best_checkpoint_path=best_checkpoint_path,
                model=model,
                charset=charset,
                training_config=actual_config,
                best_val_loss=best_val_loss,
            )

        history.append(
            OcrTrainingEpoch(
                epoch=epoch_index,
                train_loss=train_loss,
                val_loss=val_loss,
                was_best=was_best,
            )
        )

    save_training_history(
        history_path=history_path,
        dataset_path=dataset_dir,
        output_dir=training_dir,
        device=device.type,
        charset=charset,
        training_config=actual_config,
        history=history,
        best_val_loss=best_val_loss,
    )

    return OcrTrainingResult(
        dataset_path=dataset_dir,
        output_dir=training_dir,
        best_checkpoint_path=best_checkpoint_path,
        history_path=history_path,
        device=device.type,
        best_val_loss=best_val_loss,
        epochs_ran=epochs_to_run,
        history=tuple(history),
    )


def create_ocr_dataloaders(
    dataset_dir: str | Path,
    charset: OcrCharset,
    training_config: OcrTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    image_height = training_config.model_config.image_height

    train_dataset = OcrLineDataset(
        dataset_dir=dataset_dir,
        split="train",
        charset=charset,
        image_height=image_height,
    )
    val_dataset = OcrLineDataset(
        dataset_dir=dataset_dir,
        split="val",
        charset=charset,
        image_height=image_height,
    )

    # Train мешаем, чтобы батчи не были одинаковыми на каждой эпохе.
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(training_config.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=ocr_collate_fn,
    )
    # Val оставляем в стабильном порядке, здесь веса уже не меняются.
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(training_config.batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=ocr_collate_fn,
    )
    return train_loader, val_loader


def create_ctc_loss(charset: OcrCharset) -> nn.CTCLoss:
    return nn.CTCLoss(blank=charset.blank_index, zero_infinity=True)


def train_one_epoch(
    model: OcrCrnnModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: Adam,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        # Сначала чистим старые градиенты, потом считаем новый loss для текущего батча.
        optimizer.zero_grad(set_to_none=True)
        loss = compute_ctc_batch_loss(model, batch, criterion, device)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        batch_count += 1

    if batch_count == 0:
        raise OcrTrainingError("Train loader пустой.")

    return total_loss / batch_count


def validate_one_epoch(
    model: OcrCrnnModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0

    # Валидация только считает loss и не трогает веса модели.
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            loss = compute_ctc_batch_loss(model, batch, criterion, device)
            total_loss += float(loss.item())
            batch_count += 1

    if batch_count == 0:
        raise OcrTrainingError("Val loader пустой.")

    return total_loss / batch_count


def compute_ctc_batch_loss(
    model: OcrCrnnModel,
    batch: dict[str, object],
    criterion: nn.CTCLoss,
    device: torch.device,
) -> torch.Tensor:
    images, image_widths, targets, target_lengths = prepare_ctc_batch(batch, device)
    logits, output_lengths = model(images, image_widths)
    validate_ctc_lengths(output_lengths, target_lengths)

    # После модели логиты имеют форму [B, T, C]:
    # B — батч, T — временная ось по ширине, C — число классов.
    # Для CTC нужен порядок [T, B, C], поэтому меняем оси здесь.
    log_probs = logits.log_softmax(dim=2).permute(1, 0, 2).contiguous()

    try:
        # targets здесь уже склеены в один длинный вектор индексов,
        # а target_lengths хранит длину каждой строки в батче.
        loss = criterion(
            log_probs,
            targets,
            output_lengths.to(dtype=torch.long).cpu(),
            target_lengths.to(dtype=torch.long).cpu(),
        )
    except RuntimeError as error:
        raise OcrTrainingError("Не получилось посчитать CTC loss.") from error

    if not torch.isfinite(loss):
        raise OcrTrainingError("Loss сломался и стал плохим числом.")

    return loss


def prepare_ctc_batch(
    batch: dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        images = batch["images"].to(device)
        image_widths = batch["image_widths"].to(dtype=torch.long)
        targets = batch["targets"].to(device=device, dtype=torch.long)
        target_lengths = batch["target_lengths"].to(dtype=torch.long)
    except KeyError as error:
        raise OcrTrainingError(f"В батче не хватает поля: {error.args[0]}") from error

    if images.ndim != 4:
        raise OcrTrainingError("Батч картинок должен быть четырехмерным.")
    if targets.ndim != 1:
        raise OcrTrainingError("Целевые индексы должны быть одним длинным вектором.")
    # image_widths и target_lengths обе одномерные,
    # но первая относится к ширине после ресайза, а вторая — к длине текста.
    if image_widths.ndim != 1 or target_lengths.ndim != 1:
        raise OcrTrainingError("Длины батча выглядят неправильно.")
    if images.shape[0] != image_widths.shape[0] or images.shape[0] != target_lengths.shape[0]:
        raise OcrTrainingError("Размеры батча не сходятся между собой.")
    if torch.any(target_lengths <= 0):
        raise OcrTrainingError("В target_lengths не должно быть нулей.")

    return images, image_widths, targets, target_lengths


def validate_ctc_lengths(
    output_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> None:
    # output_lengths — длина временной оси после CNN,
    # target_lengths — длина исходной строки в символах.
    if torch.any(output_lengths <= 0):
        raise OcrTrainingError("У модели получилась пустая временная ось.")
    if output_lengths.shape != target_lengths.shape:
        raise OcrTrainingError("Длины выхода и цели не совпали по размеру батча.")
    if torch.any(target_lengths > output_lengths.cpu()):
        raise OcrTrainingError("Для части строк временная ось короче самой цели.")


def save_best_checkpoint(
    best_checkpoint_path: Path,
    model: OcrCrnnModel,
    charset: OcrCharset,
    training_config: OcrTrainingConfig,
    best_val_loss: float,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "charset": list(charset.characters),
        "blank_index": charset.blank_index,
        "model_config": asdict(training_config.model_config),
        "best_val_loss": best_val_loss,
    }

    # Сохраняем только лучший чекпоинт, чтобы не плодить лишние файлы.
    try:
        torch.save(checkpoint, best_checkpoint_path)
    except OSError as error:
        raise OcrTrainingError("Не удалось сохранить чекпоинт.") from error


def save_training_history(
    history_path: Path,
    dataset_path: Path,
    output_dir: Path,
    device: str,
    charset: OcrCharset,
    training_config: OcrTrainingConfig,
    history: list[OcrTrainingEpoch],
    best_val_loss: float,
) -> None:
    # Историю держим в одном простом json, чтобы потом быстро посмотреть,
    # как менялись train loss и val loss по эпохам.
    payload = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "device": device,
        "blank_index": charset.blank_index,
        "num_classes": charset.size,
        "training_config": {
            "epochs": training_config.epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "demo": training_config.demo,
            "seed": training_config.seed,
        },
        "best_val_loss": best_val_loss,
        "history": [
            {
                "epoch": item.epoch,
                "train_loss": item.train_loss,
                "val_loss": item.val_loss,
                "was_best": item.was_best,
            }
            for item in history
        ],
    }

    try:
        history_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise OcrTrainingError("Не удалось сохранить историю обучения.") from error


def resolve_dataset_path(config: AppConfig, dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        return (config.artifacts_dir / "ocr_dataset").resolve()

    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_training_output_dir(config: AppConfig, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return (config.artifacts_dir / "ocr_training").resolve()

    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def prepare_training_directory(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise OcrTrainingError("Путь для обучения занят файлом.")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OcrTrainingError("Не удалось создать папку для обучения.") from error


def validate_training_config(training_config: OcrTrainingConfig) -> None:
    if training_config.epochs <= 0:
        raise OcrTrainingError("Число эпох должно быть больше нуля.")
    if training_config.batch_size <= 0:
        raise OcrTrainingError("Размер батча должен быть больше нуля.")
    if training_config.learning_rate <= 0:
        raise OcrTrainingError("Learning rate должен быть больше нуля.")
    if training_config.num_workers < 0:
        raise OcrTrainingError("num_workers не может быть отрицательным.")
    if training_config.seed < 0:
        raise OcrTrainingError("Seed не должен быть отрицательным.")


def resolve_device(device_name: str) -> torch.device:
    normalized_name = device_name.strip().lower()
    if normalized_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized_name == "cpu":
        return torch.device("cpu")
    if normalized_name == "cuda":
        if not torch.cuda.is_available():
            raise OcrTrainingError("CUDA сейчас недоступна.")
        return torch.device("cuda")
    if normalized_name == "mps":
        if not torch.backends.mps.is_available():
            raise OcrTrainingError("MPS сейчас недоступен.")
        return torch.device("mps")

    raise OcrTrainingError("Устройство должно быть cpu, cuda, mps или auto.")


def set_training_seed(seed: int) -> None:
    torch.manual_seed(seed)
