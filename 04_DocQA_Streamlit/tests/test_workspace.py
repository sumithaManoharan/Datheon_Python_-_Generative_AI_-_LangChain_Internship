"""Tests for docqa_web.workspace: uploads, per-file errors, isolation and cleanup."""

import gc
from pathlib import Path

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
