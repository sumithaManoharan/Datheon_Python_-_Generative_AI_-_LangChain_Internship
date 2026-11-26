"""Look and feel: the stylesheet and the small HTML pieces the screen uses.

No Streamlit calls here, so it can be tested on its own. Anything that comes from a user
(file names, document locations) is HTML-escaped before it is put into a page.
"""

import html
from pathlib import Path

from docqa.answer import Citation

STYLE_PATH = Path(__file__).with_name("style.css")
TITLE = "\U0001f4c4 DocQA Chat"
TAGLINE = "Ask questions about your own documents. Every answer comes with its sources."
STEPS = (
    ("1 · Upload", "Drop files or a zip, or point to a folder on this computer."),
    ("2 · Ask", "Type a question in plain words. Follow-up questions work too."),
    ("3 · Check", "Each answer lists the file and page it came from."),
)


def load_css(path: Path = STYLE_PATH) -> str:
    """The stylesheet text."""
    return path.read_text(encoding="utf-8")


def style_tag(css: str) -> str:
    """The stylesheet wrapped so Streamlit can inject it into the page."""
    return f"<style>{css}</style>"


def banner_html() -> str:
    """The gradient header with the app name and a one-line promise."""
    return (
        '<div class="dq-banner">'
        f'<div class="dq-title">{html.escape(TITLE)}</div>'
        f'<div class="dq-tagline">{html.escape(TAGLINE)}</div>'
        "</div>"
    )


def steps_html() -> str:
    """Three cards that explain how to use the app, shown until a document is loaded."""
    cards = "".join(
        f'<div class="dq-step"><div class="dq-step-head">{html.escape(head)}</div>'
        f'<div class="dq-step-body">{html.escape(body)}</div></div>'
        for head, body in STEPS
    )
    return f'<div class="dq-steps">{cards}</div>'


def ready_pill_html(documents: int, passages: int) -> str:
    """A pill saying how many documents are ready to search."""
    noun = "document" if documents == 1 else "documents"
    return f'<div class="dq-pill">{documents} {noun} ready · {passages} passages</div>'


def source_chip_html(citation: Citation) -> str:
    """One source as a coloured chip: its number, then the file and where in it."""
    location = f" · {html.escape(citation.location())}" if citation.location() else ""
    return (
        f'<span class="dq-chip">[{citation.number}]</span>'
        f"<strong>{html.escape(citation.source)}</strong>{location}"
    )
