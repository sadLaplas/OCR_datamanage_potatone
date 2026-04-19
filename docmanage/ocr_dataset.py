from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class OcrDatasetError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class OcrCharset:
    characters: tuple[str, ...]
    blank_token: str = "<blank>"

    def __post_init__(self) -> None:
        if not self.characters:
            raise OcrDatasetError("Словарь символов пустой.")
        if self.blank_token in self.characters:
            raise OcrDatasetError("Служебный символ нельзя класть в словарь.")

    @property
    def size(self) -> int:
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        if not text:
            raise OcrDatasetError("Текст в выборке не должен быть пустым.")

        char_to_index = {char: index + 1 for index, char in enumerate(self.characters)}

        try:
            return [char_to_index[char] for char in text]
        except KeyError as error:
            raise OcrDatasetError(f"Неизвестный символ в тексте: {error.args[0]!r}") from error


@dataclass(slots=True, frozen=True)
class OcrSample:
    sample_id: str
    image_path: Path
    text: str
    split: str


class OcrLineDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        charset: OcrCharset,
        image_height: int = 32,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.split = split.strip()
        self.charset = charset
        self.image_height = image_height
        self.samples = load_samples(self.dataset_dir, self.split)

        if self.image_height < 16:
            raise OcrDatasetError("Высота изображения должна быть хотя бы 16.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = load_line_image(sample.image_path, self.image_height)
        target = torch.tensor(self.charset.encode(sample.text), dtype=torch.long)

        return {
            "sample_id": sample.sample_id,
            "text": sample.text,
            "image": image,
            "target": target,
        }


def build_charset_from_dataset(dataset_dir: str | Path) -> OcrCharset:
    samples = load_samples(dataset_dir, "train") + load_samples(dataset_dir, "val")
    characters = sorted({char for sample in samples for char in sample.text})

    if not characters:
        raise OcrDatasetError("Не получилось собрать словарь символов.")

    return OcrCharset(tuple(characters))


def load_samples(dataset_dir: str | Path, split: str) -> list[OcrSample]:
    dataset_path = Path(dataset_dir).expanduser().resolve()

    if not dataset_path.exists():
        raise OcrDatasetError(f"Папка с датасетом не найдена: {dataset_path}")
    if not dataset_path.is_dir():
        raise OcrDatasetError(f"Путь к датасету должен быть папкой: {dataset_path}")

    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "val"}:
        raise OcrDatasetError("Split должен быть train или val.")

    annotations_path = dataset_path / f"{normalized_split}.jsonl"
    if not annotations_path.exists():
        raise OcrDatasetError(f"Не найден файл аннотаций: {annotations_path}")

    try:
        lines = annotations_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OcrDatasetError(f"Не получилось прочитать аннотации: {annotations_path}") from error

    samples: list[OcrSample] = []
    for line in lines:
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise OcrDatasetError("Аннотации повреждены.") from error

        sample_id = str(record.get("sample_id", "")).strip()
        text = str(record.get("text", ""))
        image_path_value = str(record.get("image_path", "")).strip()

        if not sample_id or not text or not image_path_value:
            raise OcrDatasetError("В аннотациях не хватает полей.")

        image_path = (dataset_path / image_path_value).resolve()
        if not image_path.exists():
            raise OcrDatasetError(f"Не найдено изображение: {image_path}")

        samples.append(
            OcrSample(
                sample_id=sample_id,
                image_path=image_path,
                text=text,
                split=normalized_split,
            )
        )

    if not samples:
        raise OcrDatasetError(f"Split {normalized_split} пустой.")

    return samples


def load_line_image(image_path: Path, image_height: int) -> torch.Tensor:
    try:
        with Image.open(image_path) as image:
            grayscale_image = image.convert("L")
    except OSError as error:
        raise OcrDatasetError(f"Не получилось открыть изображение: {image_path}") from error

    if grayscale_image.height <= 0 or grayscale_image.width <= 0:
        raise OcrDatasetError(f"У изображения странный размер: {image_path}")

    scaled_width = max(8, round(grayscale_image.width * image_height / grayscale_image.height))
    resized_image = grayscale_image.resize(
        (scaled_width, image_height),
        resample=Image.Resampling.BILINEAR,
    )
    pixel_values = torch.tensor(list(resized_image.getdata()), dtype=torch.float32)
    image_tensor = pixel_values.view(image_height, scaled_width).unsqueeze(0) / 255.0
    return image_tensor


def ocr_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise OcrDatasetError("Батч пустой.")

    image_height = int(batch[0]["image"].shape[-2])
    max_width = max(int(item["image"].shape[-1]) for item in batch)
    images = torch.zeros(len(batch), 1, image_height, max_width, dtype=torch.float32)

    image_widths: list[int] = []
    targets: list[torch.Tensor] = []
    target_lengths: list[int] = []
    sample_ids: list[str] = []
    texts: list[str] = []

    for row_index, item in enumerate(batch):
        image = item["image"]
        width = int(image.shape[-1])
        images[row_index, :, :, :width] = image
        image_widths.append(width)

        target = item["target"]
        targets.append(target)
        target_lengths.append(int(target.numel()))
        sample_ids.append(str(item["sample_id"]))
        texts.append(str(item["text"]))

    return {
        "images": images,
        "image_widths": torch.tensor(image_widths, dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "sample_ids": sample_ids,
        "texts": texts,
    }
