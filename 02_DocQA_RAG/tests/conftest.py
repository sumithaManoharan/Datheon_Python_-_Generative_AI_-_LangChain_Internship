"""Shared pytest fixtures."""

from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest

PdfFactory = Callable[..., Path]


@pytest.fixture
def make_pdf(tmp_path: Path) -> PdfFactory:
    """Return a function that writes a small PDF into tmp_path.

    Each string in `pages` becomes one page; an empty string gives a blank page.
    """

    def _make(
        pages: list[str],
        name: str = "sample.pdf",
        password: str | None = None,
    ) -> Path:
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        path = tmp_path / name
        if password:
            doc.save(
                path,
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                user_pw=password,
                owner_pw=password,
            )
        else:
            doc.save(path)
        doc.close()
        return path

    return _make
