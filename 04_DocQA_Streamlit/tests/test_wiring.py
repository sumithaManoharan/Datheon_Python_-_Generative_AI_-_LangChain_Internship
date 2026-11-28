"""Smoke test: Task 4 can import Task 2's package, its own package and Streamlit."""

import importlib


def test_task_packages_and_streamlit_import() -> None:
    for name in ("docqa", "docqa_web", "streamlit", "streamlit.testing.v1"):
        assert importlib.import_module(name) is not None
