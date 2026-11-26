"""Entry point for the DocQA chat web app.

Run from the repository root:

    PYTHONPATH=02_DocQA_RAG uv run streamlit run 04_DocQA_Streamlit/app.py
"""

from docqa_web.ui import render

render()
