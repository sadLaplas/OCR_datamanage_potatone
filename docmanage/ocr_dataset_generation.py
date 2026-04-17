from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .config import AppConfig

RUSSIAN_WORDS = [
    "договор",
    "счет",
    "акт",
    "номер",
    "дата",
    "сумма",
    "поставка",
    "товар",
    "клиент",
    "строка",
    "подпись",
    "печать",
    "архив",
    "копия",
    "лист",
    "заказ",
    "файл",
    "платеж",
    "реестр",
    "касса",
    "проект",
    "таблица",
    "позиция",
    "сервис",
    "документ",
    "форма",
    "папка",
    "склад",
    "отчет",
    "итого",
]
LATIN_WORDS = [
    "invoice",
    "date",
    "number",
    "total",
    "copy",
    "order",
    "client",
    "line",
    "amount",
    "record",
    "page",
    "draft",
]
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


class OcrDatasetError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class FontChoice:
    label: str
    path: str | None


@dataclass(slots=True, frozen=True)
class OcrSampleRecord:
    sample_id: str
    split: str
    image_path: str
    text: str
    text_length: int
    font_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "image_path": self.image_path,
            "text": self.text,
            "text_length": self.text_length,
            "font_name": self.font_name,
        }


@dataclass(slots=True, frozen=True)
class OcrDatasetResult:
    output_dir: Path
    total_samples: int
    train_samples: int
    val_samples: int
    train_annotations_path: Path
    val_annotations_path: Path
    metadata_path: Path
    fonts_used: list[str]
    demo_mode: bool


def generate_ocr_dataset(
    config: AppConfig,
    *,
    output_dir: str | Path | None = None,
    count: int = 100,
    val_ratio: float = 0.2,
    seed: int = 42,
    demo: bool = False,
) -> OcrDatasetResult:
    total_samples = 12 if demo else count
    validate_generation_params(total_samples, val_ratio)

    dataset_dir = resolve_output_dir(config, output_dir)
    prepare_output_dir(dataset_dir)
    font_choices = load_font_choices()
    use_russian = any(choice.path is not None for choice in font_choices)
    train_annotations_path = dataset_dir / "train.jsonl"
    val_annotations_path = dataset_dir / "val.jsonl"
    metadata_path = dataset_dir / "metadata.json"

    split_rng = random.Random(seed + 1)
    val_indexes = build_val_indexes(total_samples, val_ratio, split_rng)
    records: list[OcrSampleRecord] = []
    fonts_used: set[str] = set()

    for sample_index in range(total_samples):
        sample_id = f"sample_{sample_index + 1:06d}"
        split = "val" if sample_index in val_indexes else "train"
        text_rng = random.Random(seed + sample_index * 17)
        image_rng = random.Random(seed + sample_index * 41 + 7)
        text = generate_text_line(text_rng, use_russian=use_russian)
        image, font_name = render_text_line(text, image_rng, font_choices)
        relative_image_path = Path("images") / split / f"{sample_id}.png"
        save_sample_image(dataset_dir / relative_image_path, image)
        image.close()

        records.append(
            OcrSampleRecord(
                sample_id=sample_id,
                split=split,
                image_path=relative_image_path.as_posix(),
                text=text,
                text_length=len(text),
                font_name=font_name,
            )
        )
        fonts_used.add(font_name)

    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    save_annotations(train_annotations_path, train_records)
    save_annotations(val_annotations_path, val_records)
    save_metadata(
        metadata_path,
        {
            "total_samples": total_samples,
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "seed": seed,
            "val_ratio": val_ratio,
            "demo_mode": demo,
            "fonts_used": sorted(fonts_used),
            "train_annotations": train_annotations_path.name,
            "val_annotations": val_annotations_path.name,
            "images_dir": "images",
        },
    )

    return OcrDatasetResult(
        output_dir=dataset_dir,
        total_samples=total_samples,
        train_samples=len(train_records),
        val_samples=len(val_records),
        train_annotations_path=train_annotations_path,
        val_annotations_path=val_annotations_path,
        metadata_path=metadata_path,
        fonts_used=sorted(fonts_used),
        demo_mode=demo,
    )


def validate_generation_params(total_samples: int, val_ratio: float) -> None:
    if total_samples < 2:
        raise OcrDatasetError("Нужно хотя бы 2 примера.")
    if not 0 < val_ratio < 1:
        raise OcrDatasetError("val_ratio должен быть между 0 и 1.")


def resolve_output_dir(config: AppConfig, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return config.artifacts_dir / "ocr_dataset"
    return Path(output_dir).expanduser().resolve()


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise OcrDatasetError(f"Папка для датасета занята файлом: {output_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OcrDatasetError(f"Не удалось создать папку: {output_dir}") from error

    shutil.rmtree(output_dir / "images", ignore_errors=True)


def load_font_choices() -> list[FontChoice]:
    choices = [
        FontChoice(label=Path(path).stem, path=path)
        for path in FONT_CANDIDATES
        if Path(path).exists()
    ]
    if choices:
        return choices
    return [FontChoice(label="default", path=None)]


def build_val_indexes(
    total_samples: int, val_ratio: float, rng: random.Random
) -> set[int]:
    indexes = list(range(total_samples))
    rng.shuffle(indexes)
    val_count = max(1, int(total_samples * val_ratio))
    if val_count >= total_samples:
        val_count = total_samples - 1
    return set(indexes[:val_count])


def generate_text_line(rng: random.Random, *, use_russian: bool) -> str:
    words = RUSSIAN_WORDS if use_russian else LATIN_WORDS
    token_count = rng.randint(2, 5)
    tokens: list[str] = []

    for index in range(token_count):
        roll = rng.random()
        if roll < 0.18:
            token = str(rng.randint(1, 9999))
        elif roll < 0.28:
            token = f"{rng.randint(1, 31):02d}.{rng.randint(1, 12):02d}.{rng.randint(21, 26)}"
        else:
            token = rng.choice(words)

        if index == 0 and rng.random() < 0.35:
            token = token.capitalize()
        if index < token_count - 1 and rng.random() < 0.12:
            token = f"{token},"
        tokens.append(token)

    line = " ".join(tokens)
    if rng.random() < 0.18:
        line = f"№ {rng.randint(1, 999)} {line}"
    if rng.random() < 0.12:
        line = f"{line}."
    return line


def render_text_line(
    text: str, rng: random.Random, font_choices: list[FontChoice]
) -> tuple[Image.Image, str]:
    font_choice = rng.choice(font_choices)
    font_size = rng.randint(26, 40)
    font = load_font(font_choice, font_size)

    draft = Image.new("L", (1, 1), 255)
    draft_draw = ImageDraw.Draw(draft)
    bbox = draft_draw.textbbox((0, 0), text, font=font)
    draft.close()

    padding_x = rng.randint(12, 20)
    padding_y = rng.randint(8, 14)
    width = max(120, bbox[2] - bbox[0] + padding_x * 2)
    height = max(48, bbox[3] - bbox[1] + padding_y * 2)
    background = rng.randint(235, 255)
    text_color = rng.randint(0, 45)

    image = Image.new("L", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.text(
        (padding_x - bbox[0], padding_y - bbox[1]),
        text,
        font=font,
        fill=text_color,
    )

    image = apply_text_augmentations(image, rng)
    return image, font_choice.label


def load_font(font_choice: FontChoice, font_size: int):
    if font_choice.path is None:
        return ImageFont.load_default()

    try:
        return ImageFont.truetype(font_choice.path, font_size)
    except OSError as error:
        raise OcrDatasetError(f"Не нашел шрифт: {font_choice.path}") from error


def apply_text_augmentations(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image.copy()
    image.close()

    result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.96, 1.05))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.92, 1.12))

    if rng.random() < 0.45:
        blurred = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
        result.close()
        result = blurred

    angle = rng.uniform(-1.6, 1.6)
    if abs(angle) >= 0.2:
        rotated = result.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )
        result.close()
        result = rotated

    noised = add_noise(result, rng, amount=rng.uniform(0.002, 0.009))
    result.close()
    result = noised

    trimmed = trim_image(result, padding=rng.randint(6, 12))
    result.close()
    return trimmed


def add_noise(image: Image.Image, rng: random.Random, amount: float) -> Image.Image:
    noisy = image.copy()
    pixels = noisy.load()
    width, height = noisy.size
    noise_points = max(1, int(width * height * amount))

    for _ in range(noise_points):
        x_index = rng.randrange(width)
        y_index = rng.randrange(height)
        value = pixels[x_index, y_index] + rng.randint(-35, 35)
        pixels[x_index, y_index] = max(0, min(255, value))

    return noisy


def trim_image(image: Image.Image, padding: int) -> Image.Image:
    inverted = ImageOps.invert(image)
    bbox = inverted.getbbox()
    inverted.close()
    if bbox is None:
        return image.copy()

    cropped = image.crop(bbox)
    trimmed = ImageOps.expand(cropped, border=padding, fill=255)
    cropped.close()
    return trimmed


def save_sample_image(image_path: Path, image: Image.Image) -> None:
    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, format="PNG")
    except OSError as error:
        raise OcrDatasetError(f"Не удалось сохранить картинку: {image_path}") from error


def save_annotations(path: Path, records: list[OcrSampleRecord]) -> None:
    try:
        path.write_text(
            "".join(
                json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    except OSError as error:
        raise OcrDatasetError(f"Не удалось сохранить аннотации: {path}") from error


def save_metadata(path: Path, payload: dict[str, object]) -> None:
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise OcrDatasetError(f"Не удалось сохранить метаданные: {path}") from error
