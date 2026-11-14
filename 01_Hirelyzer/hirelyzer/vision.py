"""Turn an image file into a data URL for the vision model, with input guards."""

import base64
from pathlib import Path

from hirelyzer.errors import EmptyDocumentError, InputTooLargeError, UnsupportedFileError

# File extension -> MIME type used in the data URL.
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
IMAGE_EXTENSIONS = frozenset(IMAGE_TYPES)


def image_to_data_url(path: Path, max_file_bytes: int) -> str:
    """Read an image file and return it as a base64 data URL, or raise a typed error."""
    mime = IMAGE_TYPES.get(path.suffix.lower())
    if mime is None:
        supported = ", ".join(sorted(IMAGE_TYPES))
        raise UnsupportedFileError(
            f"Unsupported image type {path.suffix!r} for {path.name}. Supported: {supported}"
        )
    size = path.stat().st_size
    if size == 0:
        raise EmptyDocumentError(f"{path.name} is empty.")
    if size > max_file_bytes:
        raise InputTooLargeError(
            f"{path.name} is {size / 1024 / 1024:.1f} MB; the limit is "
            f"{max_file_bytes / 1024 / 1024:.1f} MB."
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise UnsupportedFileError(f"Cannot read image {path.name}: {exc}") from exc
    # Whether the bytes really are an image is checked by the API; a bad one comes back as
    # an LLMCallError for that file only.
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
