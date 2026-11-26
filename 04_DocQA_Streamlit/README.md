# DocQA Chat: a streaming web app for your own documents

A browser chat built with Streamlit on top of the DocQA pipeline from Task 2
(`../02_DocQA_RAG`). Upload documents, ask questions in plain words, and watch the answer
stream in token by token. Every answer comes only from your documents, with numbered
sources; when the documents do not contain the answer, the app says so.

## What it does

- **Bring your own documents.** Drop many `.md`, `.txt` and `.pdf` files at once, drop a `.zip`
  (its documents are added), or type a folder path to load everything in it and its
  sub-folders.
- **Streaming answers.** The reply appears as the model writes it. When it ends, the display
  is replaced by the final text, with `[n]` markers cleaned up, and a collapsible **Sources**
  list shows the file, the page or section, and a snippet for each citation.
- **Conversation memory.** The last 6 exchanges are remembered, so a follow-up such as
  "what about the second one?" is understood. The memory is updated only after an answer has
  streamed to the end, so a stopped or failed stream never leaves half an answer behind.
- **Clear session state.** Chat history, memory, the document store and the upload results all
  live in `st.session_state`. **Clear chat** forgets the conversation; **Remove all documents**
  also deletes the session's store.
- **Honest errors.** A file that cannot be read is reported by name and does not stop the
  others. A network failure mid-answer keeps the partial text and shows the error. A missing
  API key is shown as a message, not a crash.
- **A look of its own.** Colour theme, gradient header, chat bubbles, source chips and a
  "how it works" empty state (see "Look and feel").

## Setup and run

From the repository root, once:

```
uv sync
```

Put your OpenRouter key in a `.env` file (see `../02_DocQA_RAG/.env.example`; the app finds a
`.env` in the current folder or any parent). Then start the app **from this folder**, so
Streamlit finds `.streamlit/config.toml` (the theme):

```
cd 04_DocQA_Streamlit
PYTHONPATH=../02_DocQA_RAG uv run streamlit run app.py
```

`PYTHONPATH` points at the Task 2 package (`docqa`), which this app imports. Streamlit prints a
local address (usually http://localhost:8501) and opens it in your browser.

## How it works

```
 browser  --upload / zip / folder path-->  sidebar (ui.py)
                                              |
                                   uploads.py: which files are new?
                                              |
                                   workspace.py: check, save, ingest one file at a time
                                              |                  (temp folder + own Chroma store)
 question --> chat_input (ui.py)              v
      |                                  DocQAPipeline (Task 2)
      +--> pipeline.chat_stream(question, memory)
              1. rewrite a follow-up as a standalone question (only with history)
              2. retrieve the top passages, apply the relevance threshold
              3. stream the answer from the LLM, piece by piece      --> placeholder.markdown(...)
              4. at the end: final answer + citations, memory updated, telemetry written
```

The streaming itself lives in the Task 2 package as two additive pieces, so the command line and
the web app share one pipeline:

- `AnswerGenerator.answer_stream` returns an `AnswerStream`: iterate it for text pieces, then
  read `.answer` for the finished `Answer` (citations, refusal flag, token counts). Opening the
  stream is retried on transient errors; a failure part-way raises `LLMCallError` and leaves
  no answer.
- `DocQAPipeline.chat_stream` returns a `ChatStream` that wraps it and writes memory and
  telemetry only when the stream has been read to the end.

`answer()` and `chat()` were refactored to share their helpers with the streaming path and
behave as before.

## Each visitor gets a private store

`DocumentWorkspace` (`docqa_web/workspace.py`) gives every browser session a temporary folder and
its own vector store inside it, so nothing a visitor uploads touches the project's main store
(`02_DocQA_RAG/data/chroma`). The folder is deleted on **Remove all documents** and when the
session ends. Usage telemetry is not recorded for these temporary stores.

Each upload is checked and ingested on its own, and gets its own result:

| Check | Result |
|---|---|
| Not `.md`, `.txt`, `.pdf` (or `.zip`) | Reported as unsupported, nothing saved |
| Empty, or larger than `DOCQA_MAX_FILE_MB` (20 MB by default) | Reported, nothing saved |
| Cannot be read or embedded | Reported by name, removed, not retried |
| Same name uploaded again | Replaces the earlier version |
| Two files with the same name in different folders | Both load, as `one_a.md` and `two_a.md` |

A folder or zip adds at most 200 files or 200 MB; the app says how many were skipped.

**Safety.** File names are reduced to their last part, so `../../x.md` cannot write outside the
session folder, and zip entries are read in memory, never extracted to a path. Hidden files and
folders, macOS `__MACOSX` entries and password-protected zip entries are skipped or reported.
File names and locations are HTML-escaped before they are shown in a source chip.

## Look and feel

- `.streamlit/config.toml`: light theme with violet as the main colour; hides the Deploy button.
- `docqa_web/style.css`: accents only (gradient banner, violet and teal chat bubbles, coral and
  teal highlights, source chips, status pill, tinted sidebar, rounded buttons, a narrow-screen
  layout). It sets no page or text colours, so it works on top of any base theme.
- `docqa_web/theme.py`: the small HTML pieces (banner, cards, chips, pill), with no Streamlit
  calls so they are tested directly.

## Layout

```
04_DocQA_Streamlit/
  app.py                   entry point (three lines)
  .streamlit/config.toml   colour theme
  docqa_web/
    ui.py                  the screen: sidebar, chat, session state
    workspace.py           per-session temp store, uploads, folders, zips (no Streamlit)
    uploads.py             which uploaded files are new, zip routing (no Streamlit)
    theme.py, style.css    look and feel
  tests/                   offline tests
```

## Tests

The tests need no network and no API key, and run with the rest of the repository:

```
uv run pytest -q
```

They cover the streaming generator and pipeline (pieces in order, refusal, mid-stream failure,
retry on opening, memory and telemetry written only at the end), the workspace (every check in
the table above, isolation from the main store, cleanup, folders, zips, limits, a
password-protected entry) and the upload and theme helpers (including HTML escaping).

The screen itself (`ui.py`) is a thin layer over those modules and is checked by hand in the
browser.

## Known limits

- The chat and the uploaded documents belong to one browser session. Refreshing the page or
  closing the tab starts again, and the documents have to be uploaded again (and embedded
  again, which uses a few API calls).
- The folder box reads whatever path it is given on the machine running the app. It is meant for
  running locally; do not expose it on a public server.
- While an answer streams, the text is shown as the model wrote it. `[n]` markers are
  normalised, and a refusal is replaced by the fixed refusal sentence, only when it ends.
- The token counts of the rewrite call for follow-ups are not included in telemetry, and
  telemetry is off for the temporary stores.
- The limits of the Task 2 pipeline still apply: questions that need counting across many
  documents are refused, long synthesis answers are weak, and the relevance threshold was tuned
  on one corpus (see `../02_DocQA_RAG/README.md`).
- There is no login and no sharing: it is a single-user local app.
- Some of the stylesheet targets Streamlit's own page structure (the chat bubbles). A future
  Streamlit release could change that; the app would keep working and lose only those colours.

## Design decisions

- **A separate folder that imports Task 2.** Task 2 stays exactly as graded, and the web app is
  a thin front end. The only changes to Task 2 are additive (the streaming methods), with its
  tests.
- **Logic out of the screen.** Uploads, workspace and theme pieces are plain Python so they can
  be tested without a browser. The screen keeps all of its state in `st.session_state`.
- **One store per session, one file at a time.** It isolates visitors, gives a clear result per
  file, and means a bad file is removed instead of being retried on every click.
- **Accent-only styling.** No page or text colours in the CSS, so a different base theme cannot
  make text unreadable.
- **LangChain.** As in Task 2, LangChain is used only for the text splitter; retrieval,
  prompting and streaming use the OpenAI-compatible client directly, which keeps the streaming
  and retry behaviour explicit and testable.
