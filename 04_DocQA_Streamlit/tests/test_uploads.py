"""Tests for docqa_web.uploads."""

from types import SimpleNamespace

from docqa_web.uploads import ingest_new_uploads
from docqa_web.workspace import FileResult


class FakeWorkspace:
    def __init__(self, failing=()) -> None:
        self._failing = set(failing)
        self.added: list[tuple[str, bytes]] = []

    def add_file(self, name: str, data: bytes) -> FileResult:
        self.added.append((name, data))
        if name in self._failing:
            return FileResult(name, False, "could not read")
        return FileResult(name, True, "added")


def _upload(file_id: str, name: str, data: bytes = b"text") -> SimpleNamespace:
    return SimpleNamespace(file_id=file_id, name=name, getvalue=lambda: data)


def test_every_new_file_is_added_in_order():
    workspace, seen = FakeWorkspace(), set()
    results = ingest_new_uploads(
        workspace, [_upload("1", "a.md", b"A"), _upload("2", "b.md")], seen
    )

    assert [r.name for r in results] == ["a.md", "b.md"]
    assert workspace.added == [("a.md", b"A"), ("b.md", b"text")]
    assert seen == {"1", "2"}


def test_a_file_already_seen_is_not_added_again():
    workspace, seen = FakeWorkspace(), {"1"}
    results = ingest_new_uploads(workspace, [_upload("1", "a.md"), _upload("2", "b.md")], seen)

    assert [r.name for r in results] == ["b.md"]
    assert [name for name, _ in workspace.added] == ["b.md"]


def test_calling_again_with_the_same_files_does_nothing():
    workspace, seen = FakeWorkspace(), set()
    uploads = [_upload("1", "a.md")]
    ingest_new_uploads(workspace, uploads, seen)

    assert ingest_new_uploads(workspace, uploads, seen) == []
    assert len(workspace.added) == 1


def test_a_failed_file_is_reported_and_not_retried():
    workspace, seen = FakeWorkspace(failing={"bad.pdf"}), set()
    uploads = [_upload("1", "bad.pdf"), _upload("2", "good.md")]

    first = ingest_new_uploads(workspace, uploads, seen)
    second = ingest_new_uploads(workspace, uploads, seen)

    assert [(r.name, r.ok) for r in first] == [("bad.pdf", False), ("good.md", True)]
    assert second == []


def test_no_uploads_gives_no_results():
    assert ingest_new_uploads(FakeWorkspace(), [], set()) == []
