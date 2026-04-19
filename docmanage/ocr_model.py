from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import AppConfig
from .ocr_dataset import (
    OcrDatasetError,
    OcrLineDataset,
    build_charset_from_dataset,
    ocr_collate_fn,
)


class OcrModelError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class OcrModelConfig:
    input_channels: int = 1
    conv_channels: tuple[int, ...] = (32, 64, 128, 128)
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    dropout: float = 0.1
    image_height: int = 32


@dataclass(slots=True, frozen=True)
class OcrModelCheckResult:
    dataset_path: Path
    split: str
    batch_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    output_lengths: tuple[int, ...]
    num_classes: int


class OcrCrnnModel(nn.Module):
    def __init__(self, config: OcrModelConfig, num_classes: int) -> None:
        super().__init__()
        validate_model_config(config, num_classes)
        self.config = config
        self.num_classes = num_classes
        channels = config.conv_channels

        self.features = nn.Sequential(
            conv_block(config.input_channels, channels[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(channels[0], channels[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(channels[1], channels[2]),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            conv_block(channels[2], channels[3]),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.encoder = nn.LSTM(
            input_size=channels[-1],
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(config.lstm_hidden_size * 2, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        image_widths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise OcrModelError("Батч картинок должен быть четырехмерным.")
        if images.shape[1] != self.config.input_channels:
            raise OcrModelError("Не совпал размер входа по каналам.")

        feature_map = self.features(images)
        if feature_map.ndim != 4 or feature_map.shape[2] != 1:
            raise OcrModelError("Не получилось собрать последовательность из карты признаков.")

        sequence = feature_map.squeeze(2).permute(0, 2, 1).contiguous()
        encoded_sequence, _ = self.encoder(sequence)
        logits = self.head(encoded_sequence)

        if image_widths is None:
            raw_widths = torch.full(
                (images.shape[0],),
                images.shape[-1],
                dtype=torch.long,
                device=images.device,
            )
        else:
            raw_widths = image_widths.to(images.device)

        output_lengths = self.compute_output_lengths(raw_widths)
        return logits, output_lengths

    def compute_output_lengths(self, image_widths: torch.Tensor) -> torch.Tensor:
        output_lengths = torch.div(image_widths, 2, rounding_mode="floor")
        output_lengths = torch.div(output_lengths, 2, rounding_mode="floor")
        return torch.clamp(output_lengths, min=1)


def conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


def validate_model_config(config: OcrModelConfig, num_classes: int) -> None:
    if config.input_channels <= 0:
        raise OcrModelError("Число входных каналов должно быть больше нуля.")
    if len(config.conv_channels) != 4 or any(channel <= 0 for channel in config.conv_channels):
        raise OcrModelError("Нужны четыре нормальных значения для conv_channels.")
    if config.lstm_hidden_size <= 0:
        raise OcrModelError("Размер LSTM должен быть больше нуля.")
    if config.lstm_num_layers <= 0:
        raise OcrModelError("Число слоев LSTM должно быть больше нуля.")
    if config.image_height < 16:
        raise OcrModelError("Высота входа должна быть хотя бы 16.")
    if not 0.0 <= config.dropout < 1.0:
        raise OcrModelError("Dropout должен быть в диапазоне от 0 до 1.")
    if num_classes <= 1:
        raise OcrModelError("Число классов должно быть больше одного.")


def run_ocr_model_check(
    config: AppConfig,
    dataset_path: str | Path | None = None,
    split: str = "train",
    batch_size: int = 4,
    model_config: OcrModelConfig | None = None,
) -> OcrModelCheckResult:
    dataset_dir = resolve_dataset_path(config, dataset_path)
    if batch_size <= 0:
        raise OcrModelError("Размер батча должен быть больше нуля.")

    actual_config = model_config or OcrModelConfig()

    try:
        charset = build_charset_from_dataset(dataset_dir)
        dataset = OcrLineDataset(
            dataset_dir=dataset_dir,
            split=split,
            charset=charset,
            image_height=actual_config.image_height,
        )
    except OcrDatasetError as error:
        raise OcrModelError(str(error)) from error

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        collate_fn=ocr_collate_fn,
    )
    batch = next(iter(loader))
    model = OcrCrnnModel(actual_config, num_classes=charset.size)
    model.eval()

    with torch.no_grad():
        logits, output_lengths = model(batch["images"], batch["image_widths"])

    validate_model_output(batch, logits, output_lengths, charset.size)

    return OcrModelCheckResult(
        dataset_path=dataset_dir,
        split=split,
        batch_shape=tuple(batch["images"].shape),
        logits_shape=tuple(logits.shape),
        output_lengths=tuple(int(value) for value in output_lengths.tolist()),
        num_classes=charset.size,
    )


def resolve_dataset_path(config: AppConfig, dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        return (config.artifacts_dir / "ocr_dataset").resolve()

    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def validate_model_output(
    batch: dict[str, object],
    logits: torch.Tensor,
    output_lengths: torch.Tensor,
    num_classes: int,
) -> None:
    images = batch["images"]

    if logits.ndim != 3:
        raise OcrModelError("Выход модели должен быть трехмерным.")
    if logits.shape[0] != images.shape[0]:
        raise OcrModelError("Размер батча в выходе не совпал.")
    if logits.shape[1] <= 0:
        raise OcrModelError("У модели не получилось дать временную ось.")
    if logits.shape[2] != num_classes:
        raise OcrModelError("Размер выхода не совпал со словарем.")
    if output_lengths.ndim != 1 or output_lengths.shape[0] != images.shape[0]:
        raise OcrModelError("Длины выхода выглядят неправильно.")
    if torch.any(output_lengths < 1):
        raise OcrModelError("Длины выхода не должны быть нулевыми.")
    if torch.any(output_lengths > logits.shape[1]):
        raise OcrModelError("Длины выхода больше самой последовательности.")
    if not torch.isfinite(logits).all():
        raise OcrModelError("В выходе модели появились плохие числа.")
