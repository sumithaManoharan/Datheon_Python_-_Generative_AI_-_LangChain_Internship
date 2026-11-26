"""The Streamlit screen: documents in the sidebar, a streaming chat in the main area.

All per-visitor state lives in `st.session_state`, created once by `_init_state`. Streamlit
re-runs this script on every click, so nothing expensive is built outside session state.
"""

from dataclasses import dataclass

import streamlit as st
from docqa.answer import Citation
from docqa.config import load_settings
from docqa.conversation import ConversationMemory
from docqa.errors import ConfigError, DocQAError
from docqa.loader import SUPPORTED_SUFFIXES
from docqa.logging_setup import configure_logging
from dotenv import find_dotenv, load_dotenv
from streamlit.delta_generator import DeltaGenerator

from docqa_web.uploads import ZIP_SUFFIX, ingest_new_uploads
from docqa_web.workspace import DocumentWorkspace, FileResult

PAGE_TITLE = "DocQA Chat"
UPLOAD_TYPES = [suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES | {ZIP_SUFFIX})]
CURSOR = "▌"
THINKING = "_Searching your documents..._"


@dataclass(frozen=True)
class ChatMessage:
    """One message shown in the chat. An assistant message may carry sources or an error."""

    role: str
    content: str
    citations: tuple[Citation, ...] = ()
    error: str | None = None


def source_line(citation: Citation) -> str:
    """The source line shown under an answer, e.g. '**[1] guide.pdf** (page 3)'."""
    location = f" ({citation.location()})" if citation.location() else ""
    return f"**[{citation.number}] {citation.source}**{location}"


def render() -> None:
    """Draw the whole page for this run of the script."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="\U0001f4ac")
    _init_state()
    _render_sidebar()
    st.title("Chat with your documents")
    st.caption("Answers come only from the documents you upload, with sources.")
    _render_history()
    ready = bool(st.session_state.workspace.documents)
    if not ready:
        st.info("Upload one or more documents in the sidebar to start asking questions.")
    prompt = st.chat_input(
        "Ask a question about your documents",
        max_chars=st.session_state.settings.max_question_chars,
        disabled=not ready,
    )
    if prompt:
        _answer(prompt)


def _init_state() -> None:
    """Create what a session needs, once. A key that already exists is left alone."""
    state = st.session_state
    if "settings" not in state:
        load_dotenv(find_dotenv(usecwd=True))
        try:
            state.settings = load_settings()
        except ConfigError as exc:
            st.error(str(exc))
            st.stop()
        configure_logging(state.settings.log_level)
    if "workspace" not in state:
        try:
            state.workspace = DocumentWorkspace(state.settings)
        except ConfigError as exc:
            st.error(str(exc))
            st.stop()
    state.setdefault("memory", ConversationMemory())
    state.setdefault("messages", [])
    state.setdefault("upload_status", {})
    state.setdefault("seen_uploads", set())
    state.setdefault("uploader_key", 0)


def _render_sidebar() -> None:
    """Upload box, folder box, what went wrong, what is loaded, and the two reset buttons."""
    state = st.session_state
    workspace = state.workspace
    with st.sidebar:
        st.header("Documents")
        uploads = st.file_uploader(
            "Add files or a zip",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"uploader-{state.uploader_key}",
            help="Select or drop many files at once. A .zip is opened and its documents added.",
        )
        if uploads:
            with st.spinner("Reading and indexing your documents..."):
                _record(ingest_new_uploads(workspace, uploads, state.seen_uploads))
        folder = st.text_input(
            "Or a folder on this computer",
            key=f"folder-{state.uploader_key}",
            placeholder="/home/you/Documents/policies",
        )
        if st.button("Load folder", disabled=not folder.strip()):
            with st.spinner("Reading and indexing the folder..."):
                _record(workspace.add_folder(folder))
        _show_status(workspace)
        st.button("Clear chat", on_click=_clear_chat, disabled=not state.messages)
        st.button(
            "Remove all documents",
            on_click=_remove_documents,
            disabled=not (workspace.documents or state.upload_status),
        )


def _record(results: list[FileResult]) -> None:
    """Remember the latest result for each file name, so the sidebar can show problems."""
    st.session_state.upload_status.update({result.name: result for result in results})


def _show_status(workspace: DocumentWorkspace) -> None:
    """Each file that could not be added, then a summary of what is ready to search."""
    for result in st.session_state.upload_status.values():
        if not result.ok:
            st.error(f"{result.name}: {result.message}")
    if workspace.documents:
        st.success(
            f"{len(workspace.documents)} document(s) ready ({workspace.chunk_count} passages)"
        )
        with st.expander("Show documents"):
            for name in workspace.documents:
                st.text(name)


def _render_history() -> None:
    """Show every earlier message of this session."""
    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            _show_extras(message)


def _show_extras(message: ChatMessage) -> None:
    """Sources (as an expander) and an error line, when the message has them."""
    if message.citations:
        with st.expander(f"Sources ({len(message.citations)})"):
            for citation in message.citations:
                st.markdown(source_line(citation))
                st.text(citation.snippet)
    if message.error:
        st.error(message.error)


def _answer(prompt: str) -> None:
    """Show the question, stream the answer into the chat, and remember both."""
    state = st.session_state
    state.messages.append(ChatMessage("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(THINKING)
        reply = _generate(prompt, placeholder)
        _show_extras(reply)
    state.messages.append(reply)


def _generate(prompt: str, placeholder: DeltaGenerator) -> ChatMessage:
    """Stream one answer into `placeholder`; a failure becomes a message with an error."""
    state = st.session_state
    text = ""
    try:
        stream = state.workspace.pipeline.chat_stream(prompt, state.memory)
        for piece in stream:
            text += piece
            placeholder.markdown(text + CURSOR)
        answer = stream.answer
    except DocQAError as exc:
        placeholder.markdown(text or "_No answer was produced._")
        return ChatMessage("assistant", text, error=str(exc))
    placeholder.markdown(answer.text)
    return ChatMessage("assistant", answer.text, citations=answer.citations)


def _clear_chat() -> None:
    """Button callback: forget the visible chat and the conversation memory."""
    st.session_state.messages.clear()
    st.session_state.memory.clear()


def _remove_documents() -> None:
    """Button callback: delete every document and the chat that was about them."""
    state = st.session_state
    _clear_chat()
    state.workspace.reset()
    state.upload_status.clear()
    state.seen_uploads.clear()
    state.uploader_key += 1  # a new key gives an empty uploader
