from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import AppConfig
from .documents import (
    DocumentRegistrationError,
    RegisteredDocument,
    find_registered_document_by_id,
    find_registered_document_by_path,
    register_documents,
    resolve_input_file,
)


class PdfIngestionError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class PdfPageRecord:
    document_id: str
    page_id: str
    page_number: int
    width: float
    height: float
    rotation: int
    has_text_layer: bool
    raw_text: str
    char_count: int
    word_count: int
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "page_id": self.page_id,
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "has_text_layer": self.has_text_layer,
            "raw_text": self.raw_text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "status": self.status,
        }


@dataclass(slots=True, frozen=True)
class PdfIngestionResult:
    document_id: str
    original_name: str
    absolute_path: str
    page_count: int
    text_layer_page_count: int
    page_manifest_path: Path
    pages: list[PdfPageRecord]


def ingest_pdf(config: AppConfig, source: str) -> PdfIngestionResult:
    document = resolve_pdf_document(config, source)
    file_path = Path(document.absolute_path)
    page_manifest_path = build_page_manifest_path(config, document.document_id)

    try:
        with file_path.open("rb") as file_object:
            reader = PdfReader(file_object)
            pages = build_page_records(document, reader)
    except PdfReadError as error:
        raise PdfIngestionError(f"PDF поврежден или не открывается: {file_path}") from error
    except OSError as error:
        raise PdfIngestionError(f"Не удалось открыть PDF: {file_path}") from error

    result = PdfIngestionResult(
        document_id=document.document_id,
        original_name=document.original_name,
        absolute_path=document.absolute_path,
        page_count=len(pages),
        text_layer_page_count=sum(page.has_text_layer for page in pages),
        page_manifest_path=page_manifest_path,
        pages=pages,
    )
    save_page_manifest(result)
    return result


def resolve_pdf_document(config: AppConfig, source: str) -> RegisteredDocument:
    if looks_like_pdf_path(source):
        return resolve_pdf_document_from_path(config, source)

    document = find_registered_document_by_id(config, source)
    if document is None:
        raise PdfIngestionError(f"Не найден зарегистрированный документ или PDF-файл: {source}")
    if document.document_kind != "pdf":
        raise PdfIngestionError(f"Документ не является PDF: {document.original_name}")
    return document


def resolve_pdf_document_from_path(config: AppConfig, raw_path: str) -> RegisteredDocument:
    file_path = resolve_input_file(raw_path)

    if file_path.suffix.lower() != ".pdf":
        raise PdfIngestionError(f"Поддерживается только PDF: {raw_path}")

    existing_document = find_registered_document_by_path(config, file_path)
    if existing_document is not None:
        return existing_document

    try:
        registered_documents, _ = register_documents(config, [raw_path])
    except DocumentRegistrationError as error:
        raise PdfIngestionError(str(error)) from error

    return registered_documents[0]


def build_page_records(
    document: RegisteredDocument, reader: PdfReader
) -> list[PdfPageRecord]:
    page_records: list[PdfPageRecord] = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = normalize_page_text(page.extract_text() or "")
            width = round(float(page.mediabox.width), 2)
            height = round(float(page.mediabox.height), 2)
            rotation = int(page.rotation or 0)
        except Exception as error:
            raise PdfIngestionError(
                f"Не удалось обработать страницу {page_index} в PDF: {document.original_name}"
            ) from error

        has_text_layer = bool(raw_text)
        page_records.append(
            PdfPageRecord(
                document_id=document.document_id,
                page_id=build_page_id(document.document_id, page_index),
                page_number=page_index,
                width=width,
                height=height,
                rotation=rotation,
                has_text_layer=has_text_layer,
                raw_text=raw_text,
                char_count=len(raw_text),
                word_count=len(raw_text.split()) if raw_text else 0,
                status="text_extracted" if has_text_layer else "no_text",
            )
        )

    return page_records


def save_page_manifest(result: PdfIngestionResult) -> None:
    payload = {
        "document_id": result.document_id,
        "original_name": result.original_name,
        "absolute_path": result.absolute_path,
        "page_count": result.page_count,
        "text_layer_page_count": result.text_layer_page_count,
        "pages": [page.to_dict() for page in result.pages],
    }

    try:
        result.page_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        result.page_manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise PdfIngestionError(
            f"Не удалось сохранить page manifest: {result.page_manifest_path}"
        ) from error


def build_page_manifest_path(config: AppConfig, document_id: str) -> Path:
    return config.artifacts_dir / "documents" / document_id / "page_manifest.json"


def build_page_id(document_id: str, page_number: int) -> str:
    return f"{document_id}_page_{page_number:04d}"


def normalize_page_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def looks_like_pdf_path(source: str) -> bool:
    path = Path(source).expanduser()
    return path.suffix.lower() == ".pdf" or "/" in source or source.startswith("~")
