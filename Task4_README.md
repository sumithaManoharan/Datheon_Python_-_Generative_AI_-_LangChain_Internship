# Task 4: Interactive Full-Stack AI Chatbot

**Project:** DocQA Chat, in [`04_DocQA_Streamlit/`](04_DocQA_Streamlit/) (full documentation: [`04_DocQA_Streamlit/README.md`](04_DocQA_Streamlit/README.md))

## Objective

Build an interactive chatbot with a web interface. DocQA Chat is a Streamlit app on top of the
Task 2 RAG pipeline: upload your own documents, ask questions in plain words, and watch the
answer stream in token by token. Every answer comes only from your documents, with numbered
sources; when the documents do not contain the answer, the app says so.

## Key Concepts

- Streamlit user interface (sidebar, chat, source chips, custom theme)
- Streaming responses: the answer appears as the model writes it, then is replaced by the final text with citations
- Conversation memory: the last 6 exchanges, updated only after an answer has streamed to the end
- Session state management: chat history, memory, document store and upload results live in `st.session_state`
- RAG integration: reuses the Task 2 pipeline (`DocQAPipeline.chat_stream`)
- Per-session document store: each visitor gets a private temporary vector store
- Upload safety: file type, size and count limits, path-safe file names, zip files read in memory, HTML-escaped source chips
- LLM: `openai/gpt-oss-20b` through OpenRouter

## Features

- Upload many `.md`, `.txt` and `.pdf` files, a `.zip`, or type a folder path
- Streaming answers with a collapsible **Sources** list (file, page or section, snippet)
- Follow-up questions are understood (for example "what about the second one?")
- **Clear chat** and **Remove all documents** buttons
- Clear errors: a file that cannot be read is reported by name, a mid-answer network failure keeps the partial text, a missing API key shows a message instead of a crash

## How it works

```
browser -> sidebar (upload / zip / folder) -> workspace.py (check, save, ingest one file at a time)
question -> chat_input -> pipeline.chat_stream(question, memory)
   1. rewrite a follow-up as a standalone question
   2. retrieve the top passages and apply the relevance threshold
   3. stream the answer piece by piece
   4. at the end: final answer and citations, memory updated, telemetry written
```

## How to run: the notebook is the main guide

[`Task4_Streamlit_AI_Chatbot.ipynb`](Task4_Streamlit_AI_Chatbot.ipynb) is the main way to understand
and run this project. It walks through everything in order: objective and key concepts, what the app
does, the architecture, the source of every part of the web app, the streaming pieces in the Task 2
pipeline, the per-session store and its upload checks, how to run the app, screenshots, tests, design
decisions and limitations. It can also start the real Streamlit app from inside the notebook and show
it in a frame.

**1. Install and add your API key (once)**

```bash
uv sync                                                          # from the repository root (Python 3.12)
cp 02_DocQA_RAG/.env.example 04_DocQA_Streamlit/.env             # then set OPENROUTER_API_KEY in .env
```

**2. Open the notebook from the repository root**

```bash
uv run --with jupyter jupyter lab
```

Open `Task4_Streamlit_AI_Chatbot.ipynb` and choose **Run All Cells**. This does not change the project
dependencies. The notebook finds the app folder and the Task 2 folder by itself. The app is a web
server, so it is **not** started by "Run All". To start it inside the notebook, set `START_APP = True`
in the "Run the app" section and run that cell, and run the stop cell when you are finished. The
app needs the API key from step 1 to answer questions.

The notebook does not duplicate the code. It reads the project files and the Task 2 classes from
disk, so it always matches the tested code.

## Optional: detailed review

The items in this section are optional. They are for a reviewer who wants to go deeper than the notebook.

- **Full project documentation:** [`04_DocQA_Streamlit/README.md`](04_DocQA_Streamlit/README.md) covers the look and feel, the upload checks and safety rules, the layout, tests, known limits and design decisions.
- **Run the app in a terminal, without the notebook:**

```bash
cd 04_DocQA_Streamlit
PYTHONPATH=../02_DocQA_RAG uv run streamlit run app.py
```

  Streamlit prints a local address (usually http://localhost:8501). Start the app from this folder so
  it finds `.streamlit/config.toml`. `PYTHONPATH` points at the Task 2 package (`docqa`), which the app imports.

- **Tests:**

```bash
uv run pytest -q                 # offline tests, no network or API key needed
```

  The screen is tested headlessly with Streamlit's `AppTest` (streamed answer, refusal, mid-answer
  failure, Clear chat, Remove all documents, missing API key).

## Where things are

| Path | What |
|---|---|
| `Task4_Streamlit_AI_Chatbot.ipynb` | Notebook walkthrough of the whole project (repository root) |
| `04_DocQA_Streamlit/app.py` | Entry point |
| `04_DocQA_Streamlit/docqa_web/` | `ui.py`, `workspace.py`, `uploads.py`, `theme.py`, `style.css` |
| `04_DocQA_Streamlit/.streamlit/config.toml` | Colour theme |
| `04_DocQA_Streamlit/tests/` | Offline tests |
| `02_DocQA_RAG/` | The Task 2 pipeline this app imports |

## Known limits

The chat and documents belong to one browser session, so refreshing starts again. The folder box
reads any path on the machine running the app, so it is meant for local use, not a public server.
There is no login or sharing.

Back to the [project index](README.md).
