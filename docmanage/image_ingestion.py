from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import AppConfig
from .documents import (
    DocumentRegistrationError,
    RegisteredDocument,
    find_registered_document_by_id,
    find_registered_document_by_path,
    register_documents,
    resolve_input_file,
)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG"}


class ImageIngestionError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class ImagePageRecord:
    document_id: str
    page_id: str
    page_number: int
    width: int
    height: int
    image_mode: str
    color_space: str
    has_alpha: bool
    file_format: str
    source_path: str
    normalized_image_path: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "page_id": self.page_id,
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "image_mode": self.image_mode,
            "color_space": self.color_space,
            "has_alpha": self.has_alpha,
            "file_format": self.file_format,
            "source_path": self.source_path,
            "normalized_image_path": self.normalized_image_path,
            "status": self.status,
        }


@dataclass(slots=True, frozen=True)
class ImageIngestionResult:
    document_id: str
    original_name: str
    absolute_path: str
    page_count: int
    page_manifest_path: Path
    normalized_image_path: Path
    pages: list[ImagePageRecord]


def ingest_image(config: AppConfig, source: str) -> ImageIngestionResult:
    document = resolve_image_document(config, source)
    file_path = Path(document.absolute_path)
    page_manifest_path = build_page_manifest_path(config, document.document_id)
    normalized_image_path = build_normalized_image_path(config, document.document_id)

    page_record = read_image_page(document, file_path, normalized_image_path)
    result = ImageIngestionResult(
        document_id=document.document_id,
        original_name=document.original_name,
        absolute_path=document.absolute_path,
        page_count=1,
        page_manifest_path=page_manifest_path,
        normalized_image_path=normalized_image_path,
        pages=[page_record],
    )
    save_page_manifest(result)
    return result


def resolve_image_document(config: AppConfig, source: str) -> RegisteredDocument:
    if looks_like_image_path(source):
        return resolve_image_document_from_path(config, source)

    document = find_registered_document_by_id(config, source)
    if document is None:
        raise ImageIngestionError(
            f"Не найден зарегистрированный документ или файл изображения: {source}"
        )
    if document.document_kind != "image":
        raise ImageIngestionError(f"Документ не является изображением: {document.original_name}")
    return document


def resolve_image_document_from_path(
    config: AppConfig, raw_path: str
) -> RegisteredDocument:
    file_path = resolve_input_file(raw_path)

    if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageIngestionError(f"Поддерживаются только PNG, JPG и JPEG: {raw_path}")

    existing_document = find_registered_document_by_path(config, file_path)
    if existing_document is not None:
        return existing_document

    try:
        registered_documents, _ = register_documents(config, [raw_path])
    except DocumentRegistrationError as error:
        raise ImageIngestionError(str(error)) from error

    return registered_documents[0]


def read_image_page(
    document: RegisteredDocument,
    file_path: Path,
    normalized_image_path: Path,
) -> ImagePageRecord:
    normalized_image: Image.Image | None = None

    try:
        with Image.open(file_path) as source_image:
            source_format = (source_image.format or "").upper()
            if source_format not in SUPPORTED_IMAGE_FORMATS:
                raise ImageIngestionError(
                    f"Поддерживаются только PNG, JPG и JPEG: {file_path}"
                )

            normalized_image = normalize_image(source_image)
            normalized_image.load()
            width, height = normalized_image.size
            if width <= 0 or height <= 0:
                raise ImageIngestionError(
                    f"Изображение имеет некорректный размер: {file_path}"
                )

            save_normalized_image(normalized_image, normalized_image_path)
            normalized_mode = normalized_image.mode
    except ImageIngestionError:
        raise
    except UnidentifiedImageError as error:
        raise ImageIngestionError(f"Файл не читается как изображение: {file_path}") from error
    except OSError as error:
        raise ImageIngestionError(
            f"Изображение повреждено или не читается: {file_path}"
        ) from error
    finally:
        if normalized_image is not None:
            normalized_image.close()

    return ImagePageRecord(
        document_id=document.document_id,
        page_id=build_page_id(document.document_id, 1),
        page_number=1,
        width=width,
        height=height,
        image_mode=normalized_mode,
        color_space=detect_color_space(normalized_mode),
        has_alpha="A" in normalized_mode,
        file_format=source_format.lower(),
        source_path=str(file_path),
        normalized_image_path=str(normalized_image_path),
        status="image_loaded",
    )


def normalize_image(source_image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(source_image)
    mode = image.mode

    if mode in {"RGB", "RGBA", "L"}:
        return image.copy()
    if mode in {"LA", "PA"}:
        return image.convert("RGBA")
    if mode in {"1"}:
        return image.convert("L")
    if mode == "P":
        if "transparency" in image.info:
            return image.convert("RGBA")
        return image.convert("RGB")
    return image.convert("RGB")


def save_normalized_image(image: Image.Image, normalized_image_path: Path) -> None:
    try:
        normalized_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(normalized_image_path, format="PNG")
    except OSError as error:
        raise ImageIngestionError(
            f"Не удалось сохранить нормализованное изображение: {normalized_image_path}"
        ) from error


def save_page_manifest(result: ImageIngestionResult) -> None:
    payload = {
        "document_id": result.document_id,
        "original_name": result.original_name,
        "absolute_path": result.absolute_path,
        "page_count": result.page_count,
        "pages": [page.to_dict() for page in result.pages],
    }

    try:
        result.page_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        result.page_manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ImageIngestionError(
            f"Не удалось сохранить page manifest: {result.page_manifest_path}"
        ) from error


def build_page_manifest_path(config: AppConfig, document_id: str) -> Path:
    return config.artifacts_dir / "documents" / document_id / "page_manifest.json"


def build_normalized_image_path(config: AppConfig, document_id: str) -> Path:
    return config.artifacts_dir / "documents" / document_id / "normalized_page.png"


def build_page_id(document_id: str, page_number: int) -> str:
    return f"{document_id}_page_{page_number:04d}"


def detect_color_space(image_mode: str) -> str:
    if image_mode == "L":
        return "grayscale"
    if image_mode == "RGBA":
        return "rgba"
    return "rgb"


def looks_like_image_path(source: str) -> bool:
    path = Path(source).expanduser()
    return (
        path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        or "/" in source
        or source.startswith("~")
    )
