"""Turn the files a browser sent into workspace calls. No Streamlit imports."""

from collections.abc import Iterable
from typing import Protocol

from docqa_web.workspace import DocumentWorkspace, FileResult


class Upload(Protocol):
    """What the app needs from an uploaded file (Streamlit's UploadedFile has all of it)."""

    file_id: str
    name: str

    def getvalue(self) -> bytes:
        """The file's content."""
        ...


def ingest_new_uploads(
    workspace: DocumentWorkspace, uploads: Iterable[Upload], seen: set[str]
) -> list[FileResult]:
    """Ingest the uploads not handled before, and return one result per new file.

    Streamlit re-runs the script on every click and the uploader keeps returning the same
    files, so `seen` (file ids) stops a file being read twice. A file that failed is marked
    as seen too, so it is not retried, and re-reported, on every later click.
    """
    results: list[FileResult] = []
    for upload in uploads:
        if upload.file_id in seen:
            continue
        seen.add(upload.file_id)
        results.append(workspace.add_file(upload.name, upload.getvalue()))
    return results
