from pathlib import Path

from PIL import Image


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
