from pathlib import Path

from PIL import Image, ImageDraw


def create_image(
    path: Path,
    *,
    size: tuple[int, int] = (120, 160),
    mode: str = "RGB",
    color: int | tuple[int, ...] = (240, 240, 240),
    image_format: str | None = None,
) -> Path:
    image = Image.new(mode, size, color)
    save_format = image_format or path.suffix.lstrip(".").upper()

    if save_format == "JPG":
        save_format = "JPEG"

    image.save(path, format=save_format)
    image.close()
    return path


def create_document_like_image(path: Path, *, size: tuple[int, int] = (240, 320)) -> Path:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)

    for line_index in range(8):
        top = 30 + line_index * 28
        draw.rectangle((24, top, size[0] - 24, top + 8), fill=(40, 40, 40))

    save_format = path.suffix.lstrip(".").upper()
    if save_format == "JPG":
        save_format = "JPEG"

    image.save(path, format=save_format)
    image.close()
    return path
