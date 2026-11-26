"""Tests for the Streamlit screen (docqa_web.ui), run headlessly with Streamlit's AppTest.

The real workspace and pipeline are replaced by fakes placed in `st.session_state`, so the
tests need no network, no API key and no browser.
"""

from pathlib import Path

from docqa.answer import Answer, Citation
from docqa.config import Settings
from docqa.conversation import ConversationMemory
from docqa.errors import LLMCallError
from docqa_web.workspace import FileResult
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
# The card markup. The bare name also appears in the stylesheet, which is injected as markdown.
CARD = 'class="dq-step"'
ANSWER = Answer(
    text="Hail damage is covered [1].",
    citations=(Citation(number=1, source="policy.pdf", snippet="Hail is covered.", page=3),),
    refused=False,
)


class FakeChatStream:
    """Pieces, then an answer; adds the exchange to memory when it finishes, like the real one."""

    def __init__(self, question, memory, pieces, answer, fail_at=None):
        self._question = question
        self._memory = memory
        self._pieces = pieces
        self._answer = answer
        self._fail_at = fail_at

    def __iter__(self):
        for number, piece in enumerate(self._pieces):
            if number == self._fail_at:
                raise LLMCallError("connection lost")
            yield piece
        self._memory.add(self._question, self._answer.text)

    @property
    def answer(self):
        return self._answer


class FakePipeline:
    def __init__(self, pieces=("Hail damage ", "is covered [1]."), answer=ANSWER, fail_at=None):
        self.questions: list[str] = []
        self.memories: list[ConversationMemory] = []
        self._pieces = pieces
        self._answer = answer
        self._fail_at = fail_at

    def chat_stream(self, question, memory):
        self.questions.append(question)
        self.memories.append(memory)
        return FakeChatStream(question, memory, self._pieces, self._answer, self._fail_at)


class FakeWorkspace:
    def __init__(self, documents=(), pipeline=None):
        self.documents = tuple(documents)
        self.chunk_count = 7 if documents else 0
        self.pipeline = pipeline or FakePipeline()
        self.resets = 0
        self.folders: list[str] = []
        self.folder_results = [FileResult("a.md", True, "added")]

    def add_folder(self, path):
        self.folders.append(path)
        return self.folder_results

    def add_zip(self, name, data):
        return []

    def add_file(self, name, data):
        return FileResult(name, True, "added")

    def reset(self):
        self.resets += 1
        self.documents = ()
        self.chunk_count = 0


def _app(workspace, memory=None) -> AppTest:
    """The app with a fake workspace in place; nothing is run yet."""
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.session_state["settings"] = Settings()
    app.session_state["workspace"] = workspace
    app.session_state["memory"] = memory if memory is not None else ConversationMemory()
    return app


def _markdown(app) -> list[str]:
    return [item.value for item in list(app.markdown) + list(app.sidebar.markdown)]


def _errors(app) -> list[str]:
    return [item.value for item in list(app.error) + list(app.sidebar.error)]


def _button(app, label: str):
    return next(button for button in app.sidebar.button if label in button.label)


def _ask(app, question: str):
    app.chat_input[0].set_value(question).run()
    return app


# ---------- first look ----------


def test_the_page_opens_without_an_error_and_shows_the_banner():
    app = _app(FakeWorkspace()).run()

    assert not app.exception
    assert any('class="dq-banner"' in text for text in _markdown(app))


def test_with_no_documents_the_how_it_works_cards_are_shown_and_chat_is_disabled():
    app = _app(FakeWorkspace()).run()

    assert any(CARD in text for text in _markdown(app))
    assert app.chat_input[0].proto.disabled is True


def test_with_documents_the_cards_go_away_and_the_status_pill_appears():
    app = _app(FakeWorkspace(documents=("a.md", "b.md"))).run()

    assert not any(CARD in text for text in _markdown(app))
    assert any("2 documents ready" in text for text in _markdown(app))
    assert app.chat_input[0].proto.disabled is False


# ---------- asking ----------


def test_a_question_shows_the_final_answer_and_its_sources():
    workspace = FakeWorkspace(documents=("policy.pdf",))
    app = _ask(_app(workspace).run(), "Is hail covered?")

    assert not app.exception
    assert len(app.chat_message) == 2
    assert workspace.pipeline.questions == ["Is hail covered?"]
    assert any("Hail damage is covered [1]." in text for text in _markdown(app))
    assert any("Sources (1)" in expander.label for expander in app.expander)
    assert any("policy.pdf" in text and "page 3" in text for text in _markdown(app))


def test_the_question_is_given_the_session_memory():
    memory = ConversationMemory()
    workspace = FakeWorkspace(documents=("policy.pdf",))
    _ask(_app(workspace, memory).run(), "Is hail covered?")

    assert workspace.pipeline.memories == [memory]
    assert [turn.question for turn in memory.turns] == ["Is hail covered?"]


def test_the_conversation_is_kept_across_questions():
    app = _app(FakeWorkspace(documents=("policy.pdf",))).run()
    _ask(app, "First?")
    _ask(app, "And then?")

    roles = [message.role for message in app.session_state["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert len(app.chat_message) == 4


def test_a_refusal_is_shown_as_the_refusal_with_no_sources():
    refusal = Answer(
        text="I could not find this in the loaded documents.", citations=(), refused=True
    )
    workspace = FakeWorkspace(documents=("a.md",), pipeline=FakePipeline(answer=refusal))
    app = _ask(_app(workspace).run(), "Who won the cup?")

    assert any("I could not find this" in text for text in _markdown(app))
    assert not any("Sources" in expander.label for expander in app.expander)


def test_a_failure_part_way_keeps_the_partial_text_and_shows_the_error():
    pipeline = FakePipeline(pieces=("Hail ", "damage"), fail_at=1)
    workspace = FakeWorkspace(documents=("a.md",), pipeline=pipeline)
    app = _ask(_app(workspace).run(), "Is hail covered?")

    assert not app.exception
    assert any("connection lost" in text for text in _errors(app))
    last = app.session_state["messages"][-1]
    assert (last.role, last.content, last.error) == ("assistant", "Hail ", "connection lost")


# ---------- the sidebar ----------


def test_clear_chat_forgets_the_conversation_and_the_memory():
    memory = ConversationMemory()
    app = _app(FakeWorkspace(documents=("a.md",)), memory).run()
    _ask(app, "Is hail covered?")

    _button(app, "Clear chat").click().run()

    assert app.session_state["messages"] == []
    assert len(memory) == 0
    assert len(app.chat_message) == 0


def test_remove_all_documents_resets_the_workspace_and_the_chat():
    workspace = FakeWorkspace(documents=("a.md",))
    app = _app(workspace).run()
    _ask(app, "Is hail covered?")

    _button(app, "Remove all documents").click().run()

    assert workspace.resets == 1
    assert app.session_state["messages"] == []
    assert app.session_state["uploader_key"] == 1
    assert any(CARD in text for text in _markdown(app))


def test_the_folder_box_loads_the_typed_path():
    workspace = FakeWorkspace()
    app = _app(workspace).run()

    app.sidebar.text_input[0].set_value("/data/policies").run()
    _button(app, "Load folder").click().run()

    assert workspace.folders == ["/data/policies"]


def test_a_file_that_failed_is_listed_by_name_in_the_sidebar():
    workspace = FakeWorkspace()
    workspace.folder_results = [FileResult("scan.pdf", False, "no text found")]
    app = _app(workspace).run()

    app.sidebar.text_input[0].set_value("/data/scans").run()
    _button(app, "Load folder").click().run()

    assert any("scan.pdf: no text found" in text for text in _errors(app))


# ---------- start-up problems ----------


def test_a_missing_api_key_is_shown_as_a_message_not_a_crash():
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.session_state["settings"] = Settings()  # no key set; the real workspace will refuse
    app.run()

    assert not app.exception
    assert any("OPENROUTER_API_KEY" in text for text in _errors(app))


def test_an_invalid_setting_is_shown_as_a_message_not_a_crash(monkeypatch):
    monkeypatch.setattr("docqa_web.ui.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("DOCQA_CHUNK_SIZE", "not-a-number")
    app = AppTest.from_file(str(APP), default_timeout=10).run()

    assert not app.exception
    assert any("Invalid configuration" in text for text in _errors(app))


def test_remove_all_documents_is_disabled_when_there_is_nothing_to_remove():
    app = _app(FakeWorkspace()).run()

    assert _button(app, "Remove all documents").disabled is True


def test_clear_chat_is_available_right_after_the_first_answer():
    app = _app(FakeWorkspace(documents=("a.md",))).run()
    _ask(app, "Is hail covered?")

    assert _button(app, "Clear chat").disabled is False
