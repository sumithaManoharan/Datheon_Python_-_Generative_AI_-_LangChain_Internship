"""Tests for docqa_web.workspace: uploads, per-file errors, isolation and cleanup."""

import gc
import io
import zipfile
from pathlib import Path

import docqa_web.workspace as workspace_module
import pytest
from docqa.config import Settings
from docqa.errors import EmbeddingError
from docqa.pipeline import IngestSummary
from docqa_web.workspace import DocumentWorkspace

OK = IngestSummary(embedded=1, skipped=0, failures=(), total_chunks=5)
UNCHANGED = IngestSummary(embedded=0, skipped=1, failures=(), total_chunks=5)


class FakePipeline:
    """Records each ingest call and what was in the folder at that moment."""

    def __init__(self, results=()) -> None:
        self._results = list(results)
        self.folders: list[Path] = []
        self.contents: list[list[str]] = []

    def ingest(self, folder: Path) -> IngestSummary:
        self.folders.append(folder)
        self.contents.append(sorted(p.name for p in folder.iterdir()))
        if not self._results:
            return OK
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def made():
    """(pipelines, settings) captured from every pipeline the factory builds."""
    return [], []


def _workspace(tmp_path: Path, made, results=(), settings=None) -> DocumentWorkspace:
    pipelines, seen_settings = made

    def factory(session_settings):
        seen_settings.append(session_settings)
        pipeline = FakePipeline(results if not pipelines else ())
        pipelines.append(pipeline)
        return pipeline

    return DocumentWorkspace(settings or Settings(), factory, base_dir=tmp_path)


def test_a_file_is_saved_in_a_private_folder_and_ingested(tmp_path, made):
    workspace = _workspace(tmp_path, made)
    result = workspace.add_file("notes.md", b"# Notes\n\nhello")

    assert (result.name, result.ok, result.message) == ("notes.md", True, "added")
    pipeline = made[0][0]
    assert pipeline.contents == [["notes.md"]]
    assert tmp_path in pipeline.folders[0].parents
    assert workspace.documents == ("notes.md",)
    assert workspace.chunk_count == 5


def test_the_session_store_never_points_at_the_projects_main_store(tmp_path, made):
    original = Settings()
    _workspace(tmp_path, made, settings=original)

    session_store = made[1][0].store_dir
    assert session_store != original.store_dir
    assert tmp_path in session_store.parents


def test_an_unsupported_file_type_is_rejected_without_saving_or_ingesting(tmp_path, made):
    workspace = _workspace(tmp_path, made)
    result = workspace.add_file("program.exe", b"MZ")

    assert result.ok is False
    assert "unsupported file type '.exe'" in result.message
    assert made[0][0].folders == []
    assert workspace.documents == ()


def test_an_empty_file_is_rejected(tmp_path, made):
    result = _workspace(tmp_path, made).add_file("blank.md", b"")

    assert result.ok is False
    assert "empty" in result.message


def test_a_file_over_the_size_limit_is_rejected(tmp_path, made):
    workspace = _workspace(tmp_path, made, settings=Settings(max_file_mb=1))
    result = workspace.add_file("big.txt", b"x" * (1024 * 1024 + 1))

    assert result.ok is False
    assert "larger than 1 MB" in result.message
    assert made[0][0].folders == []


@pytest.mark.parametrize("name", ["", ".", "..", ".hidden.md"])
def test_an_invalid_file_name_is_rejected(tmp_path, made, name):
    result = _workspace(tmp_path, made).add_file(name, b"text")

    assert result.ok is False
    assert made[0][0].folders == []


@pytest.mark.parametrize("name", ["../../evil.md", "..\\..\\evil.md", "/etc/evil.md"])
def test_a_path_in_the_name_cannot_escape_the_session_folder(tmp_path, made, name):
    workspace = _workspace(tmp_path, made)
    result = workspace.add_file(name, b"text")

    assert result.ok is True
    assert result.name == "evil.md"
    assert made[0][0].contents == [["evil.md"]]
    assert tmp_path in made[0][0].folders[0].parents
    assert not (tmp_path.parent / "evil.md").exists()


def test_a_file_that_fails_to_load_is_reported_and_removed(tmp_path, made):
    failed = IngestSummary(
        embedded=0, skipped=0, failures=("scan.pdf: no text found",), total_chunks=0
    )
    workspace = _workspace(tmp_path, made, results=[failed])
    result = workspace.add_file("scan.pdf", b"%PDF-1.4")

    assert result.ok is False
    assert "no text found" in result.message
    assert workspace.documents == ()
    assert not made[0][0].folders[0].exists()


def test_an_error_raised_while_ingesting_is_reported_not_raised(tmp_path, made):
    workspace = _workspace(tmp_path, made, results=[EmbeddingError("embedding service down")])
    result = workspace.add_file("a.md", b"text")

    assert result.ok is False
    assert "embedding service down" in result.message
    assert workspace.documents == ()


def test_a_failed_file_does_not_stop_the_next_one(tmp_path, made):
    workspace = _workspace(tmp_path, made, results=[EmbeddingError("down")])

    assert workspace.add_file("a.md", b"text").ok is False
    assert workspace.add_file("b.md", b"text").ok is True
    assert workspace.documents == ("b.md",)


def test_uploading_the_same_name_again_reuses_its_folder(tmp_path, made):
    workspace = _workspace(tmp_path, made, results=[OK, UNCHANGED])

    first = workspace.add_file("a.md", b"text")
    second = workspace.add_file("a.md", b"text")

    pipeline = made[0][0]
    assert pipeline.folders[0] == pipeline.folders[1]
    assert (first.message, second.message) == ("added", "already loaded")
    assert workspace.documents == ("a.md",)


def test_different_names_get_different_folders(tmp_path, made):
    workspace = _workspace(tmp_path, made)
    workspace.add_file("a.md", b"one")
    workspace.add_file("b.md", b"two")

    pipeline = made[0][0]
    assert pipeline.folders[0] != pipeline.folders[1]
    assert workspace.documents == ("a.md", "b.md")


def test_reset_deletes_everything_and_starts_a_fresh_store(tmp_path, made):
    workspace = _workspace(tmp_path, made)
    workspace.add_file("a.md", b"text")
    old_folder = made[0][0].folders[0]
    old_store = made[1][0].store_dir

    workspace.reset()

    assert not old_folder.exists()
    assert workspace.documents == ()
    assert workspace.chunk_count == 0
    assert len(made[0]) == 2 and workspace.pipeline is made[0][1]
    assert made[1][1].store_dir != old_store


def test_close_removes_the_temporary_folder_and_can_be_repeated(tmp_path, made):
    workspace = _workspace(tmp_path, made)
    workspace.add_file("a.md", b"text")
    folder = made[0][0].folders[0]

    workspace.close()
    workspace.close()

    assert not folder.exists()
    assert list(tmp_path.iterdir()) == []


def test_the_folder_is_removed_when_the_pipeline_cannot_be_built(tmp_path):
    def broken(_settings):
        raise RuntimeError("no api key")

    with pytest.raises(RuntimeError):
        DocumentWorkspace(Settings(), broken, base_dir=tmp_path)

    gc.collect()
    assert list(tmp_path.iterdir()) == []


# ---------- folders and zips ----------


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _library(tmp_path: Path, files: dict[str, bytes]) -> Path:
    library = tmp_path / "library"
    for name, data in files.items():
        target = library / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return library


def test_a_folder_adds_supported_files_including_sub_folders(tmp_path, made):
    library = _library(
        tmp_path,
        {
            "a.md": b"alpha",
            "sub/b.txt": b"beta",
            "notes.exe": b"MZ",
            ".hidden/c.md": b"secret",
        },
    )
    workspace = _workspace(tmp_path, made)

    results = workspace.add_folder(str(library))

    assert [(r.name, r.ok) for r in results] == [("a.md", True), ("b.txt", True)]
    assert workspace.documents == ("a.md", "b.txt")


@pytest.mark.parametrize("text", ["", "   ", "does-not-exist"])
def test_something_that_is_not_a_folder_is_reported(tmp_path, made, text):
    workspace = _workspace(tmp_path, made)
    (result,) = workspace.add_folder(str(tmp_path / text) if text.strip() else text)

    assert result.ok is False
    assert "not a folder" in result.message
    assert made[0][0].folders == []


def test_a_file_path_is_not_accepted_as_a_folder(tmp_path, made):
    library = _library(tmp_path, {"a.md": b"alpha"})
    (result,) = _workspace(tmp_path, made).add_folder(str(library / "a.md"))

    assert result.ok is False
    assert "not a folder" in result.message


def test_files_with_the_same_name_in_different_folders_both_load(tmp_path, made):
    library = _library(tmp_path, {"one/a.md": b"first", "two/a.md": b"second"})
    workspace = _workspace(tmp_path, made)

    results = workspace.add_folder(str(library))

    assert [r.name for r in results] == ["one_a.md", "two_a.md"]
    assert workspace.documents == ("one_a.md", "two_a.md")


def test_a_folder_with_nothing_supported_is_reported(tmp_path, made):
    library = _library(tmp_path, {"program.exe": b"MZ"})
    (result,) = _workspace(tmp_path, made).add_folder(str(library))

    assert (result.name, result.ok) == ("library", False)
    assert "no supported files" in result.message


def test_one_bad_file_in_a_folder_does_not_stop_the_rest(tmp_path, made):
    library = _library(tmp_path, {"a.md": b"alpha", "b.md": b"beta"})
    workspace = _workspace(tmp_path, made, results=[EmbeddingError("down")])

    results = workspace.add_folder(str(library))

    assert [(r.name, r.ok) for r in results] == [("a.md", False), ("b.md", True)]
    assert workspace.documents == ("b.md",)


def test_an_oversized_file_in_a_folder_is_reported_and_never_ingested(tmp_path, made):
    library = _library(tmp_path, {"big.txt": b"x" * (1024 * 1024 + 1)})
    workspace = _workspace(tmp_path, made, settings=Settings(max_file_mb=1))

    (result,) = workspace.add_folder(str(library))

    assert result.ok is False
    assert "larger than 1 MB" in result.message
    assert made[0][0].folders == []


def test_a_batch_stops_at_the_file_limit_and_says_so(tmp_path, made, monkeypatch):
    monkeypatch.setattr(workspace_module, "MAX_BATCH_FILES", 2)
    library = _library(tmp_path, {"a.md": b"1", "b.md": b"2", "c.md": b"3"})
    workspace = _workspace(tmp_path, made)

    results = workspace.add_folder(str(library))

    assert [r.ok for r in results] == [True, True, False]
    assert results[-1].name == "library"
    assert "1 more file(s) skipped" in results[-1].message
    assert workspace.documents == ("a.md", "b.md")


def test_a_batch_stops_at_the_size_limit_and_says_so(tmp_path, made, monkeypatch):
    monkeypatch.setattr(workspace_module, "MAX_BATCH_BYTES", 10)
    library = _library(tmp_path, {"a.md": b"123456", "b.md": b"123456"})
    workspace = _workspace(tmp_path, made)

    results = workspace.add_folder(str(library))

    assert [r.ok for r in results] == [True, False]
    assert "1 more file(s) skipped" in results[-1].message
    assert workspace.documents == ("a.md",)


def test_a_zip_adds_supported_files_and_ignores_the_rest(tmp_path, made):
    data = _zip(
        {
            "a.md": b"alpha",
            "__MACOSX/._a.md": b"junk",
            "program.exe": b"MZ",
            "sub/b.txt": b"beta",
            "docs/": b"",
            "dir\\c.md": b"gamma",
        }
    )
    workspace = _workspace(tmp_path, made)

    results = workspace.add_zip("bundle.zip", data)

    assert [(r.name, r.ok) for r in results] == [("a.md", True), ("b.txt", True), ("c.md", True)]
    assert workspace.documents == ("a.md", "b.txt", "c.md")


def test_something_that_is_not_a_zip_is_reported(tmp_path, made):
    (result,) = _workspace(tmp_path, made).add_zip("bundle.zip", b"this is not a zip")

    assert (result.name, result.ok) == ("bundle.zip", False)
    assert "not a valid zip" in result.message


def test_a_zip_with_nothing_supported_is_reported(tmp_path, made):
    (result,) = _workspace(tmp_path, made).add_zip("bundle.zip", _zip({"program.exe": b"MZ"}))

    assert (result.name, result.ok) == ("bundle.zip", False)
    assert "no supported files" in result.message


def test_zip_entries_with_the_same_name_both_load(tmp_path, made):
    data = _zip({"one/a.md": b"first", "two/a.md": b"second"})
    results = _workspace(tmp_path, made).add_zip("bundle.zip", data)

    assert [r.name for r in results] == ["one_a.md", "two_a.md"]


def test_an_oversized_zip_entry_is_reported_and_never_ingested(tmp_path, made):
    data = _zip({"big.txt": b"x" * (1024 * 1024 + 1)})  # compresses to almost nothing
    workspace = _workspace(tmp_path, made, settings=Settings(max_file_mb=1))

    (result,) = workspace.add_zip("bundle.zip", data)

    assert result.ok is False
    assert "larger than 1 MB" in result.message
    assert made[0][0].folders == []


def test_a_password_protected_zip_entry_is_reported_and_the_rest_load(tmp_path, made):
    raw = bytearray(_zip({"secret.md": b"hidden", "open.md": b"visible"}))
    first_entry = raw.index(b"PK\x01\x02")  # central directory header of the first entry
    raw[first_entry + 8] |= 0x1  # the "encrypted" flag
    workspace = _workspace(tmp_path, made)

    results = workspace.add_zip("bundle.zip", bytes(raw))

    assert [(r.name, r.ok) for r in results] == [("secret.md", False), ("open.md", True)]
    assert "password protected" in results[0].message
    assert workspace.documents == ("open.md",)


def test_a_zip_batch_stops_at_the_file_limit(tmp_path, made, monkeypatch):
    monkeypatch.setattr(workspace_module, "MAX_BATCH_FILES", 1)
    data = _zip({"a.md": b"1", "b.md": b"2"})
    workspace = _workspace(tmp_path, made)

    results = workspace.add_zip("bundle.zip", data)

    assert [r.ok for r in results] == [True, False]
    assert results[-1].name == "bundle.zip"
    assert workspace.documents == ("a.md",)
