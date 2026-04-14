from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .config import AppConfig

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".csv": "spreadsheet",
    ".xlsx": "spreadsheet",
}


class DocumentRegistrationError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class RegisteredDocument:
    document_id: str
    original_name: str
    original_path: str
    absolute_path: str
    extension: str
    document_kind: str
    file_size_bytes: int
    sha256: str
    registered_at: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "original_name": self.original_name,
            "original_path": self.original_path,
            "absolute_path": self.absolute_path,
            "extension": self.extension,
            "document_kind": self.document_kind,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "registered_at": self.registered_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, raw_document: dict[str, object]) -> RegisteredDocument:
        return cls(
            document_id=_read_text_field(raw_document, "document_id"),
            original_name=_read_text_field(raw_document, "original_name"),
            original_path=_read_text_field(raw_document, "original_path"),
            absolute_path=_read_text_field(raw_document, "absolute_path"),
            extension=_read_text_field(raw_document, "extension"),
            document_kind=_read_text_field(raw_document, "document_kind"),
            file_size_bytes=_read_int_field(raw_document, "file_size_bytes"),
            sha256=_read_text_field(raw_document, "sha256"),
            registered_at=_read_text_field(raw_document, "registered_at"),
            status=_read_text_field(raw_document, "status"),
        )


def register_documents(
    config: AppConfig, input_paths: list[str]
) -> tuple[list[RegisteredDocument], Path]:
    if not input_paths:
        raise DocumentRegistrationError("Нужно передать хотя бы один файл.")

    manifest_path = config.manifest_path
    existing_documents = load_manifest(manifest_path)
    existing_paths = {document.absolute_path for document in existing_documents}
    seen_paths: set[str] = set()
    new_documents: list[RegisteredDocument] = []

    for raw_path in input_paths:
        file_path = resolve_input_file(raw_path)
        absolute_path = str(file_path)

        if absolute_path in seen_paths:
            raise DocumentRegistrationError(f"Файл передан повторно: {raw_path}")
        if absolute_path in existing_paths:
            raise DocumentRegistrationError(f"Файл уже зарегистрирован: {file_path}")

        new_documents.append(build_document_record(file_path, raw_path))
        seen_paths.add(absolute_path)

    save_manifest(
        manifest_path,
        config.project_name,
        existing_documents + new_documents,
    )
    return new_documents, manifest_path


def resolve_input_file(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()

    if not path.exists():
        raise DocumentRegistrationError(f"Файл не найден: {raw_path}")
    if not path.is_file():
        raise DocumentRegistrationError(f"Ожидался файл, а получена директория: {raw_path}")

    try:
        return path.resolve()
    except OSError as error:
        raise DocumentRegistrationError(f"Не удалось обработать путь: {raw_path}") from error


def build_document_record(file_path: Path, original_path: str) -> RegisteredDocument:
    extension = file_path.suffix.lower()
    document_kind = detect_document_kind(extension)

    try:
        file_size_bytes = file_path.stat().st_size
    except OSError as error:
        raise DocumentRegistrationError(f"Не удалось прочитать размер файла: {file_path}") from error

    file_checksum = calculate_sha256(file_path)
    document_id = generate_document_id(file_path, file_checksum)

    return RegisteredDocument(
        document_id=document_id,
        original_name=file_path.name,
        original_path=original_path,
        absolute_path=str(file_path),
        extension=extension,
        document_kind=document_kind,
        file_size_bytes=file_size_bytes,
        sha256=file_checksum,
        registered_at=current_timestamp(),
        status="registered",
    )


def detect_document_kind(extension: str) -> str:
    document_kind = SUPPORTED_EXTENSIONS.get(extension.lower())
    if document_kind is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentRegistrationError(
            f"Неподдерживаемое расширение файла '{extension}'. Поддерживаются: {supported}"
        )
    return document_kind


def generate_document_id(file_path: Path, file_checksum: str) -> str:
    source = f"{file_path}:{file_checksum}"
    digest = sha256(source.encode("utf-8")).hexdigest()
    return f"doc_{digest[:12]}"


def calculate_sha256(file_path: Path) -> str:
    digest = sha256()

    try:
        with file_path.open("rb") as file_object:
            while chunk := file_object.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise DocumentRegistrationError(f"Не удалось прочитать файл: {file_path}") from error

    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> list[RegisteredDocument]:
    if not manifest_path.exists():
        return []
    if not manifest_path.is_file():
        raise DocumentRegistrationError(
            f"Путь manifest должен указывать на файл: {manifest_path}"
        )

    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DocumentRegistrationError(
            f"Manifest поврежден или не читается: {error.msg}"
        ) from error
    except OSError as error:
        raise DocumentRegistrationError(
            f"Не удалось прочитать manifest: {manifest_path}"
        ) from error

    if not isinstance(raw_manifest, dict):
        raise DocumentRegistrationError("Manifest имеет некорректную структуру.")

    raw_documents = raw_manifest.get("documents")
    if not isinstance(raw_documents, list):
        raise DocumentRegistrationError("Manifest имеет некорректную структуру.")

    try:
        documents: list[RegisteredDocument] = []
        for item in raw_documents:
            if not isinstance(item, dict):
                raise ValueError("Некорректная запись документа.")
            documents.append(RegisteredDocument.from_dict(item))
        return documents
    except ValueError as error:
        raise DocumentRegistrationError("Manifest имеет некорректную структуру.") from error


def save_manifest(
    manifest_path: Path, project_name: str, documents: list[RegisteredDocument]
) -> None:
    payload = {
        "project_name": project_name,
        "updated_at": current_timestamp(),
        "documents": [document.to_dict() for document in documents],
    }

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise DocumentRegistrationError(
            f"Не удалось сохранить manifest: {manifest_path}"
        ) from error


def current_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _read_text_field(raw_document: dict[str, object], field_name: str) -> str:
    value = raw_document.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Поле '{field_name}' должно быть непустой строкой.")
    return value.strip()


def _read_int_field(raw_document: dict[str, object], field_name: str) -> int:
    value = raw_document.get(field_name)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Поле '{field_name}' должно быть неотрицательным числом.")
    return value
