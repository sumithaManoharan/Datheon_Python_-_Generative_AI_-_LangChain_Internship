"""A private, temporary document workspace for one browser session.

Each workspace owns a temporary folder holding the uploaded files and a vector store of its
own, so nothing a visitor uploads touches the project's main store, and everything is deleted
when the workspace is reset or closed. Files arrive one at a time, from a folder on this
computer, or inside a zip. This module has no Streamlit imports.
"""

import io
import logging
import shutil
import tempfile
import weakref
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from docqa.config import Settings
from docqa.errors import DocQAError
from docqa.loader import SUPPORTED_SUFFIXES
from docqa.pipeline import DocQAPipeline, build_pipeline

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
SESSION_PREFIX = "docqa-session-"
# One folder or zip adds at most this many files, and this many bytes in total.
MAX_BATCH_FILES = 200
MAX_BATCH_BYTES = 200 * BYTES_PER_MB
ZIP_ENCRYPTED_FLAG = 0x1
# Reading a file from disk or from inside a zip can fail in these ways.
READ_ERRORS = (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, zlib.error)

# One file of a batch: its display name, its size in bytes, and a function that reads it.
_Item = tuple[str, int, Callable[[], bytes]]


@dataclass(frozen=True)
class FileResult:
    """What happened to one file; `message` is safe to show to the user."""

    name: str
    ok: bool
    message: str


def _default_factory(settings: Settings) -> DocQAPipeline:
    """The real pipeline for a session; usage telemetry is not kept for temporary stores."""
    return build_pipeline(settings, record_telemetry=False)


def _eligible(path: PurePosixPath) -> bool:
    """True for a supported file that is not hidden or macOS zip metadata."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    return not any(part.startswith(".") or part == "__MACOSX" for part in path.parts)


def _unique_names(paths: list[PurePosixPath]) -> list[str]:
    """File names, with the folder path folded in only where two files share a name."""
    counts = Counter(path.name for path in paths)
    return [path.name if counts[path.name] == 1 else "_".join(path.parts) for path in paths]


def _zip_reader(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Callable[[], bytes]:
    """A function that reads one zip entry, refusing a password-protected one."""

    def read() -> bytes:
        if info.flag_bits & ZIP_ENCRYPTED_FLAG:
            raise PermissionError("password protected")
        with archive.open(info) as handle:
            return handle.read()

    return read


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
        problem = self._check(clean, len(data))
        if problem:
            return FileResult(clean or name, False, problem)
        return self._ingest(clean, data)

    def add_folder(self, path_text: str) -> list[FileResult]:
        """Add every supported file in a folder on this computer, including sub-folders.

        Hidden files and folders are skipped. Only meant for an app running locally: it
        reads whatever path it is given.
        """
        text = path_text.strip()
        folder = Path(text).expanduser()
        if not text or not folder.is_dir():
            return [FileResult(text or "(no folder)", False, "not a folder on this computer")]
        found = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and _eligible(PurePosixPath(path.relative_to(folder).as_posix()))
        )
        relative = [PurePosixPath(path.relative_to(folder).as_posix()) for path in found]
        items: list[_Item] = [
            (name, path.stat().st_size, path.read_bytes)
            for name, path in zip(_unique_names(relative), found, strict=True)
        ]
        return self._add_many(folder.name or text, items)

    def add_zip(self, name: str, data: bytes) -> list[FileResult]:
        """Add every supported file inside a zip; entries are never extracted to a path."""
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            return [FileResult(name, False, "not a valid zip file")]
        with archive:
            entries = [
                (PurePosixPath(info.filename.replace("\\", "/")), info)
                for info in archive.infolist()
                if not info.is_dir()
            ]
            entries = [(path, info) for path, info in entries if _eligible(path)]
            names = _unique_names([path for path, _info in entries])
            items: list[_Item] = [
                (label, info.file_size, _zip_reader(archive, info))
                for label, (_path, info) in zip(names, entries, strict=True)
            ]
            return self._add_many(name, items)

    def reset(self) -> None:
        """Forget every document: delete the temporary folder and start a fresh, empty one."""
        self.close()
        self._start()

    def close(self) -> None:
        """Delete the temporary folder; safe to call more than once."""
        self._finalizer()

    def _add_many(self, label: str, items: list[_Item]) -> list[FileResult]:
        """Check, read and ingest each item; one bad file never stops the rest."""
        if not items:
            suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
            return [FileResult(label, False, f"no supported files found (use {suffixes})")]
        results: list[FileResult] = []
        taken = total = skipped = 0
        for name, size, read in items:
            problem = self._check(name, size)
            if problem:
                results.append(FileResult(name, False, problem))
                continue
            if taken >= MAX_BATCH_FILES or total + size > MAX_BATCH_BYTES:
                skipped += 1
                continue
            try:
                data = read()
            except READ_ERRORS as exc:
                results.append(FileResult(name, False, f"could not be read: {exc}"))
                continue
            taken += 1
            total += size
            results.append(self._ingest(name, data))
        if skipped:
            limit = f"{MAX_BATCH_FILES} files or {MAX_BATCH_BYTES // BYTES_PER_MB} MB"
            results.append(
                FileResult(label, False, f"{skipped} more file(s) skipped; the limit is {limit}")
            )
        return results

    def _ingest(self, name: str, data: bytes) -> FileResult:
        """Save a checked file in its own folder and ingest it."""
        folder = self._folder_for(name)
        (folder / name).write_bytes(data)
        try:
            summary = self.pipeline.ingest(folder)
        except DocQAError as exc:
            return self._failed(name, folder, str(exc))
        if summary.failures:
            return self._failed(name, folder, "; ".join(summary.failures))
        self._chunks = summary.total_chunks
        return FileResult(name, True, "added" if summary.embedded else "already loaded")

    def _check(self, name: str, size: int) -> str | None:
        """Why a file of this name and size cannot be used, or None if it can."""
        if not name or name.startswith("."):
            return "not a valid file name"
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            return f"unsupported file type '{suffix}' (use {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        if size == 0:
            return "the file is empty"
        if size > self._settings.max_file_mb * BYTES_PER_MB:
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
