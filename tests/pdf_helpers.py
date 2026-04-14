from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def create_pdf(path: Path, page_texts: list[str | None]) -> Path:
    writer = PdfWriter()

    for text in page_texts:
        page = writer.add_blank_page(width=300, height=400)
        if text is not None:
            add_text_to_page(writer, page, text)

    with path.open("wb") as file_object:
        writer.write(file_object)

    return path


def add_text_to_page(writer: PdfWriter, page, text: str) -> None:
    font_dictionary = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font_dictionary)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )

    stream = DecodedStreamObject()
    escaped_text = (
        text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    )
    stream.set_data(
        f"BT\n/F1 12 Tf\n36 360 Td\n({escaped_text}) Tj\nET".encode("utf-8")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
