"""Tests for docqa_web.theme: the stylesheet and the HTML pieces."""

from docqa.answer import Citation
from docqa_web.theme import (
    STYLE_PATH,
    TAGLINE,
    banner_html,
    load_css,
    ready_pill_html,
    source_chip_html,
    steps_html,
    style_tag,
)


def test_the_stylesheet_ships_with_the_package_and_covers_the_main_pieces():
    css = load_css()

    assert STYLE_PATH.is_file()
    for selector in (".dq-banner", ".dq-step", ".dq-chip", ".dq-pill", "@media"):
        assert selector in css


def test_the_stylesheet_is_wrapped_in_a_style_tag():
    assert style_tag("a{}") == "<style>a{}</style>"


def test_the_banner_carries_the_name_and_the_tagline():
    banner = banner_html()

    assert "DocQA Chat" in banner
    assert TAGLINE in banner
    assert banner.startswith('<div class="dq-banner">')


def test_there_are_three_how_it_works_cards():
    assert steps_html().count('class="dq-step"') == 3


def test_the_pill_pluralises_and_shows_the_passage_count():
    assert "1 document ready" in ready_pill_html(1, 12)
    assert "3 documents ready" in ready_pill_html(3, 12)
    assert "12 passages" in ready_pill_html(3, 12)


def test_a_chip_shows_number_file_and_page():
    chip = source_chip_html(Citation(number=2, source="guide.pdf", snippet="s", page=3))

    assert "[2]" in chip
    assert "guide.pdf" in chip
    assert "page 3" in chip


def test_a_chip_for_text_shows_the_section_and_lines():
    citation = Citation(
        number=1, source="a.md", snippet="s", section="Intro", start_line=4, end_line=9
    )

    assert "Intro, lines 4-9" in source_chip_html(citation)


def test_a_chip_without_a_location_has_no_separator():
    assert "·" not in source_chip_html(Citation(number=1, source="a.md", snippet="s"))


def test_markup_in_a_file_name_or_section_cannot_reach_the_page():
    citation = Citation(
        number=1, source="<img src=x onerror=alert(1)>.md", snippet="s", section="<b>bold</b>"
    )
    chip = source_chip_html(citation)

    assert "<img" not in chip
    assert "<b>" not in chip
    assert "&lt;img" in chip
