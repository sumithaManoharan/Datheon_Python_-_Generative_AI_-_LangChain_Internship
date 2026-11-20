# DocQA: Document Q&A with RAG, citations and refusal

Ask questions about a folder of documents and get answers that come **only** from those
documents, each with numbered sources. When the documents do not contain the answer, the
assistant says so instead of guessing.

The bundled corpus is 76 synthetic Markdown files about a fictional insurance-tech company
("Insurellm"). PDF and plain-text files are supported too.

## How it works

```
 documents (.md .txt .pdf)
        |  loader: NFKC-normalised text, page numbers for PDFs
        v
 chunking: heading-aware, 800 chars / 100 overlap, line spans kept
        |
        v  embeddings (OpenRouter, openai/text-embedding-3-small)
 Chroma vector store (cosine) -- idempotent: unchanged files are skipped
        ^
        |  question -> embed -> top-k chunks -> relevance threshold
        |                                   |
     ingest                                 v
                       grounded prompt -> LLM (openai/gpt-oss-20b)
                                   |
                                   v
                    answer + numbered citations, or the fixed refusal
```

Two layers protect against invented answers:

1. If no retrieved chunk passes the relevance threshold, the refusal is returned **without
   calling the LLM**.
2. Otherwise the prompt tells the model to answer only from the numbered context and to reply
   with the exact refusal sentence when the answer is absent. A refusal clears the citations.

**Citations.** PDFs cite file and page. Markdown and text cite file, section and line span
(for example `about.md | About Insurellm, lines 1-5`). Nothing is converted to PDF. PDF support
is covered by tests with generated PDFs and was also checked on real PDFs (see "Using your own
documents"); the bundled corpus is Markdown.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and an OpenRouter API key.

```bash
uv sync                    # from the repository root
cd 02_DocQA_RAG
cp .env.example .env       # then set OPENROUTER_API_KEY in .env
```

## Use

```bash
uv run python -m docqa ingest                      # embed data/source_md (first run: 76 files)
uv run python -m docqa ask "Who founded Insurellm?"
uv run python -m docqa ask --json "When was Insurellm founded?"
uv run python -m docqa chat                        # conversation; remembers the last 6 exchanges
uv run python -m docqa stats                       # timing and token usage of past questions
```

Log lines go to stderr, so `2>/dev/null` leaves only the result. Exit codes: `0` success
(a refusal is a success), `1` any error or any file that failed to ingest, `2` usage error.

Ingest (first run, then a second run that changes nothing):

```
Ingest complete: 76 embedded, 0 unchanged, 0 failed; 632 chunks in the store.
Ingest complete: 0 embedded, 76 unchanged, 0 failed; 632 chunks in the store.
```

An answer with its source:

```
Insurellm was founded by Avery Lancaster[1].

Sources:
[1] about.md | About Insurellm, lines 1-5
    "# About Insurellm Insurellm was founded by Avery Lancaster in 2015 as an insurance tech startup designed to disrupt an industry in need of innovative products. Its first product was Markellm, the mark..."
```

A question the documents cannot answer:

```
I could not find this in the loaded documents.
```

`--json` output (snippet shortened):

```json
{
  "text": "Insurellm was founded in 2015. [1]",
  "citations": [
    {
      "number": 1,
      "source": "overview.md",
      "snippet": "# Overview of Insurellm Insurellm is an innovative insurance tech firm with 32 employees...",
      "page": null,
      "section": "Overview of Insurellm",
      "start_line": 1,
      "end_line": 5
    }
  ],
  "refused": false,
  "prompt_tokens": 1373,
  "completion_tokens": 106
}
```

## Using your own documents

Put `.md`, `.txt` or `.pdf` files in a folder and ingest that folder into **its own store**, so
they do not mix with other documents. The usage log is written next to the store folder, so give
the store its own parent folder:

```bash
export DOCQA_STORE_DIR=$HOME/my_project/store      # created on the first ingest
uv run python -m docqa ingest --source-dir /path/to/my_documents
uv run python -m docqa ask "A question about them"
```

Checked on real files in a separate store: a text file (answers cite `notes.txt | line 1`), a
one-page PDF cheat sheet (correct answers citing `cheat_sheet.pdf | page 1`), and a 19-page
research paper (110 chunks; answers cited pages 2, 3 and 6, and a question outside the documents
was refused).

Things to know:

- Ingesting a folder **adds** to the store. There is no command to remove documents; delete the
  store folder to start again.
- The defaults (`DOCQA_RELEVANCE_THRESHOLD` 0.25, `DOCQA_TOP_K` 10) were tuned on the Insurellm
  corpus. If valid questions are refused on your documents, lower the threshold.
- PDF text comes from plain text extraction. Columns, tables and indentation can come out
  jumbled (the cheat sheet's headings were letter-spaced), and scanned PDFs without a text layer
  have nothing to read.
- Only `.md`, `.txt` and `.pdf` are read, and `--source-dir` takes a folder, not a single file.

## Conversation memory

`ask` answers each question on its own. `chat` is an interactive session that remembers the last
6 question-and-answer exchanges (`--turns N` changes that), so a follow-up can use a pronoun:

```
> Who won the IIOTY award in 2023?
> What is her job title?
```

Each follow-up is first rewritten by the LLM into a standalone question, which is what retrieval
searches with; the model then sees the earlier exchanges when it answers. Facts must still come
from the numbered context passages, and citation numbers from old answers are stripped from the
remembered text so they cannot be confused with the new ones. Type `/clear` to forget the history
and `/exit` (or Ctrl-D) to quit. An error on one question is reported and the session continues.

## Configuration

Set in `.env` or the environment. Invalid values stop the program with a clear error.

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | none | Required by `ingest` and `ask`; not needed by `stats` or the tests |
| `DOCQA_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `DOCQA_LLM_MODEL` | `openai/gpt-oss-20b` | Answer model |
| `DOCQA_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `DOCQA_CHUNK_SIZE` / `DOCQA_CHUNK_OVERLAP` | 800 / 100 | Characters per chunk and overlap |
| `DOCQA_TOP_K` | 10 | Chunks retrieved per question |
| `DOCQA_RELEVANCE_THRESHOLD` | 0.25 | Minimum relevance (1 minus cosine distance) to keep a chunk |
| `DOCQA_TEMPERATURE` | 0.0 | LLM temperature |
| `DOCQA_MAX_QUESTION_CHARS` | 500 | Longer questions are rejected |
| `DOCQA_MAX_FILE_MB` | 20 | Larger files are rejected |
| `DOCQA_EMBED_BATCH_SIZE` | 64 | Texts per embedding request |
| `DOCQA_MAX_ATTEMPTS` / `DOCQA_RETRY_BASE_DELAY` | 3 / 1.0 | Retry policy for transient API errors |
| `DOCQA_SOURCE_DIR` | `data/source_md` | Documents to ingest |
| `DOCQA_STORE_DIR` | `data/chroma` | Vector store location (git-ignored) |
| `DOCQA_LOG_LEVEL` | `INFO` | Log level |

Changing the chunk size or overlap re-ingests files automatically, because the stored key is
a hash of the file bytes plus both chunk settings.

## Telemetry

Every `ask` appends one JSON line to `data/telemetry.jsonl` (git-ignored). It holds stage
timings, token counts and the refused flag, never the question text. A write failure is
logged and never breaks an answer.

```json
{"timestamp": "2026-11-19T11:15:49+00:00", "question_chars": 22, "retrieved": 10, "refused": false, "retrieve_seconds": 1.0501, "generate_seconds": 2.2493, "prompt_tokens": 1429, "completion_tokens": 92}
```

```
$ uv run python -m docqa stats
Questions: 2 (1 refused)
Latency (s): retrieve 0.91 | generate 1.12 | total 2.04 (p95 3.30)
Tokens: 1,429 prompt, 92 completion
```

## Tests

```bash
uv run pytest -q          # the whole suite; no network and no API key needed
uv run ruff check docqa tests && uv run ruff format --check docqa tests
```

Every external part (embedder, LLM client, store) is injected, so tests use fakes. Nothing runs
at import time.

## Evaluation

Questions are in `evaluation/` (150 answerable, in 7 categories; 5 unrelated and 6 near-topic
questions that must be refused). Relevance and answer correctness are **keyword proxies**: a
chunk or an answer counts as covering a question when it contains the question's keywords. The
numbers compare settings with each other; they are not ground truth. A paraphrase or a different
number format can miss a keyword, and each answer-level number comes from a single run.

### Retrieval (`uv run python -m docqa.evaluate`)

| k | recall | precision | MRR | nDCG | spanning recall |
|---|---|---|---|---|---|
| 4 | 0.929 | 0.773 | 0.933 | 0.945 | 0.756 |
| 6 | 0.948 | 0.703 | 0.936 | 0.931 | 0.808 |
| 8 | 0.957 | 0.643 | 0.937 | 0.923 | 0.853 |
| 10 | 0.963 | 0.584 | 0.938 | 0.921 | 0.898 |

The file name counts as chunk text in this evaluation, so these are upper bounds.

### Full answers (`uv run python -m docqa.evaluate_answers`, makes real API calls)

| top_k | keyword coverage | all keywords | answerable but refused | cited | prompt tokens | seconds |
|---|---|---|---|---|---|---|
| 6 | 0.707 | 0.527 | 6.7% | 100% | 1,050 | 2.36 |
| 10 | 0.755 | 0.620 | 5.3% | 100% | 1,586 | 2.57 |

| category | k=6 | k=10 |
|---|---|---|
| comparative | 0.900 | 1.000 |
| direct_fact | 0.740 | 0.793 |
| holistic | 0.500 | 0.550 |
| numerical | 0.800 | 0.633 |
| relationship | 0.900 | 0.900 |
| spanning | 0.363 | 0.422 |
| temporal | 0.800 | 0.925 |

`top_k` 10 is the default: it improved coverage by 0.048 for 51% more prompt tokens.

**Refusals:** all 11 unanswerable questions were refused at both settings. The 5 unrelated ones
were stopped by the relevance threshold with no LLM call. The 6 near-topic ones passed the
threshold and were refused by the model, so for those the prompt is the only protection.
Among the answerable questions, none were refused by the threshold: all 8 refusals at `top_k` 10
(10 at `top_k` 6) came from the model.

## Known limits

- **Aggregation questions** ("total value of all Healthllm contracts", "average Carllm price",
  "employees under a salary") need every matching document. Top-k retrieval cannot gather them,
  and the assistant refuses. Refusing is safer than a wrong total, but these questions are not
  answered.
- **Hidden hops** such as "the salary of the founder of Insurellm" need two lookups. Retrieval
  often finds the right file but the wrong section, and some are refused.
- **Spanning questions** are the weakest category (coverage 0.42), although the facts are mostly
  retrieved. The model does not always use all of them.
- **Numerical questions got worse at k=10** (0.800 to 0.633), mostly two price-tier questions.
  More passages may mix up tiers; this is a guess from a single run.
- **Near-topic refusals rely on the model.** Only 6 such questions were tested.
- **Conversation memory lives only in the running `chat` session.** It is not saved between runs.
  Each follow-up costs one extra LLM call (the rewrite), and that call's tokens are not counted
  in the telemetry log.
- **Failed `ask` calls** (an embedding or LLM error) are not written to the telemetry log.

## Design decisions

- **LangChain only where it saves work.** `RecursiveCharacterTextSplitter` does the splitting.
  Chroma and the OpenAI-compatible client are used directly, so retries, citations and the
  refusal path stay explicit and testable.
- **Heading-aware chunks with line spans**, and the file name in the embedded header, so
  citations point to a section and lines, and a question that names a file or person matches it.
- **Idempotent ingestion.** Files are keyed by a content hash; a changed file is re-embedded
  first and its old chunks deleted second, so a failed API call keeps the previous version.
- **Per-file isolation.** One unreadable file or failed embedding is reported and the rest
  continue; the exit code is still `1`.
- **Retries only for transient errors** (rate limit, connection, timeout, 5xx) with exponential
  backoff and jitter. A bad API key or a bad request fails immediately.
- **Citation markers are parsed and normalised.** The model sometimes writes `【1】` or `[1, 2]`;
  both become ordinary `[n]` references, and only cited passages are listed as sources.
- **Typed errors.** Every failure the program expects is a `DocQAError` subclass, so the CLI
  turns it into a one-line message and exit code `1`.

## Layout

```
docqa/
  config.py        validated settings        loader.py       text and PDF loading
  chunking.py      heading-aware splitting   embeddings.py   embedder with retry
  store.py         Chroma, idempotent ingest retriever.py    top-k and threshold
  answer.py        prompt, citations, refusal  pipeline.py   ingest and ask
  cli.py           command line               telemetry.py   usage log and stats
  conversation.py  chat memory (last N turns)
  retrieval_metrics.py, evaluate.py, evaluate_answers.py     evaluation
  prompts/grounding.txt                        the system prompt
data/source_md/    the 76 documents (see data/README.md)
evaluation/        questions (see evaluation/README.md)
tests/             unit tests with fakes (no network)
```