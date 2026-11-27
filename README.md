# Datheon Python + Generative AI & LangChain Internship

Four hands-on projects, each in its own folder with its own README that explains the design,
setup, usage, tests and limitations.

## Projects

| # | Project | What it does | README | Run |
|---|---|---|---|---|
| 1 | **Hirelyzer** | Turns resumes (PDF, DOCX, TXT, images) into JSON validated by a strict Pydantic schema, with retry, repair and evaluation | [`01_Hirelyzer/README.md`](01_Hirelyzer/README.md) | `cd 01_Hirelyzer && uv run python -m hirelyzer.cli <resume file>` |
| 2 | **DocQA (RAG)** | Answers questions strictly from your own documents, with source page citations and refusal | [`02_DocQA_RAG/README.md`](02_DocQA_RAG/README.md) | see its README |
| 3 | **CartSense** | A LangChain tool-calling agent that interviews you, searches the web, and calculates the real cost of owning a product | [`03_CartSense/README.md`](03_CartSense/README.md) | `cd 03_CartSense && uv run python -m cartsense.cli "I want to buy a phone"` |
| 4 | **DocQA Streamlit** | A streaming chat app for your own documents, built on Task 2 | [`04_DocQA_Streamlit/README.md`](04_DocQA_Streamlit/README.md) | `PYTHONPATH=02_DocQA_RAG uv run streamlit run 04_DocQA_Streamlit/app.py` |

## Setup

```bash
uv sync        # from the repository root; Python 3.12, one shared environment
```

Each project reads its settings from a `.env` file in its own folder. Copy that folder's
`.env.example` to `.env` and add your keys. `.env` files are git-ignored and never committed.

## Repository layout

```
.
├── 01_Hirelyzer/        Task 1: structured JSON extraction
├── 02_DocQA_RAG/        Task 2: document Q&A with RAG
├── 03_CartSense/        Task 3: multi-tool purchase-decision agent
├── 04_DocQA_Streamlit/  Task 4: Streamlit chat over Task 2
├── pyproject.toml       shared dependencies
└── uv.lock
```

## Notes

- Only synthetic sample data is committed. Real documents, `.env` files, logs and generated
  output stay out of git.
- Each project's tests use fakes and need no network or API credits. Each README shows how to
  run them.
