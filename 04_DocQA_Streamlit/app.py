"""Entry point for the DocQA chat web app.

Run from this folder, so Streamlit finds `.streamlit/config.toml` (the colour theme):

    cd 04_DocQA_Streamlit
    PYTHONPATH=../02_DocQA_RAG uv run streamlit run app.py
"""

from docqa_web.ui import render

render()
