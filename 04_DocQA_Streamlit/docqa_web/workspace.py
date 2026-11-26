"""A private, temporary document workspace for one browser session.

Each workspace owns a temporary folder holding the uploaded files and a vector store of its
own, so nothing a visitor uploads touches the project's main store, and everything is deleted
when the workspace is reset or closed. This module has no Streamlit imports.
"""

import logging
import shutil
import tempfile
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from docqa.config import Settings
from docqa.errors import DocQAError
from docqa.loader import SUPPORTED_SUFFIXES
from docqa.pipeline import DocQAPipeline, build_pipeline

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
SESSION_PREFIX = "docqa-session-"


@dataclass(frozen=True)
class FileResult:
    """What happened to one uploaded file; `message` is safe to show to the user."""

    name: str
    ok: bool
    message: str


def _default_factory(settings: Settings) -> DocQAPipeline:
    """The real pipeline for a session; usage telemetry is not kept for temporary stores."""
    return build_pipeline(settings, record_telemetry=False)


class DocumentWorkspace:
    """Upload, ingest and forget documents for one session.

    The pipeline factory is injected so tests can pass a fake. `base_dir` is where the
    temporary folder is created (the system temp folder by default).
    """

    def __init__(
        self,
        settings: Settings,
        pipeline_factory: Callable[[Settings], DocQAPipeline] = _default_factory,
        base_dir: Path | None = None,
    ) -> None:
        self._settings = settings
        self._factory = pipeline_factory
        self._base_dir = base_dir
        self._start()

    def _start(self) -> None:
        """Create a fresh temporary folder, an empty store inside it, and its pipeline."""
        self._root = Path(tempfile.mkdtemp(prefix=SESSION_PREFIX, dir=self._base_dir))
        # Registered first, so the folder is also removed if building the pipeline fails.
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(self._root), True)
        self._folders: dict[str, Path] = {}
        self._next_id = 0
        self._chunks = 0
        session_settings = self._settings.model_copy(update={"store_dir": self._root / "chroma"})
        self.pipeline = self._factory(session_settings)

    @property
    def documents(self) -> tuple[str, ...]:
        """Names of the files that were ingested successfully, sorted."""
        return tuple(sorted(self._folders))

    @property
    def chunk_count(self) -> int:
        """How many passages are searchable right now."""
        return self._chunks

    def add_file(self, name: str, data: bytes) -> FileResult:
        """Save one uploaded file and ingest it; a problem is reported, never raised.

        Uploading the same name again replaces the earlier version.
        """
        clean = Path(name.replace("\\", "/")).name
        problem = self._check(clean, data)
        if problem:
            return FileResult(clean or name, False, problem)
        folder = self._folder_for(clean)
        (folder / clean).write_bytes(data)
        try:
            summary = self.pipeline.ingest(folder)
        except DocQAError as exc:
            return self._failed(clean, folder, str(exc))
        if summary.failures:
            return self._failed(clean, folder, "; ".join(summary.failures))
        self._chunks = summary.total_chunks
        return FileResult(clean, True, "added" if summary.embedded else "already loaded")

    def reset(self) -> None:
        """Forget every document: delete the temporary folder and start a fresh, empty one."""
        self.close()
        self._start()

    def close(self) -> None:
        """Delete the temporary folder; safe to call more than once."""
        self._finalizer()

    def _check(self, name: str, data: bytes) -> str | None:
        """Why this upload cannot be used, or None if it can."""
        if not name or name.startswith("."):
            return "not a valid file name"
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            return f"unsupported file type '{suffix}' (use {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        if not data:
            return "the file is empty"
        if len(data) > self._settings.max_file_mb * BYTES_PER_MB:
            return f"the file is larger than {self._settings.max_file_mb} MB"
        return None

    def _folder_for(self, name: str) -> Path:
        """The folder holding this file: reused for the same name, new for a new name."""
        folder = self._folders.get(name)
        if folder is None:
            folder = self._root / "docs" / str(self._next_id)
            self._next_id += 1
            folder.mkdir(parents=True)
            self._folders[name] = folder
        return folder

    def _failed(self, name: str, folder: Path, reason: str) -> FileResult:
        """Remove a file that could not be ingested so it is not retried, and report why."""
        logger.warning("could not add %s: %s", name, reason)
        shutil.rmtree(folder, ignore_errors=True)
        self._folders.pop(name, None)
        return FileResult(name, False, reason)
