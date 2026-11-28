# Task 2: Document Q&A Assistant with RAG & Vector Embeddings

**Project:** DocQA, in [`02_DocQA_RAG/`](02_DocQA_RAG/) (full documentation: [`02_DocQA_RAG/README.md`](02_DocQA_RAG/README.md))

## Objective

Build a document question-answering system that retrieves the relevant passages from a set of
documents and generates answers **only** from them, with numbered source citations. When the
documents do not contain the answer, the assistant says so instead of guessing.

## Key Concepts

- Retrieval-Augmented Generation (RAG)
- Document processing: Markdown, plain text and PDF loading (page numbers kept for PDFs)
- Text chunking: heading-aware, 800 characters with 100 overlap, line spans kept
- Vector embeddings (`openai/text-embedding-3-small` through OpenRouter)
- Semantic search: Chroma vector store (cosine), top-k retrieval and a relevance threshold
- Grounded prompting, citations and a fixed refusal sentence
- Retrieval and answer evaluation, telemetry, conversation memory

## Pipeline

```
documents (.md .txt .pdf) -> loader -> heading-aware chunking -> embeddings -> Chroma store
question -> embed -> top-k chunks -> relevance threshold -> grounded prompt -> LLM (openai/gpt-oss-20b)
         -> answer with numbered citations, or the fixed refusal
```

Two layers protect against invented answers:

1. If no retrieved chunk passes the relevance threshold, the refusal is returned without calling the LLM.
2. Otherwise the prompt tells the model to answer only from the numbered context. A refusal clears the citations.

## How to run: the notebook is the main guide

[`Task2_DocQA_RAG.ipynb`](Task2_DocQA_RAG.ipynb) is the main way to run and understand this project.
It walks through everything in order: objective and key concepts, setup, pipeline architecture, a
tour of every module, the grounding prompt, the document corpus, a live ingest, a cited answer, a
refusal, conversation memory, telemetry, evaluation results, tests, design decisions and limitations.

**1. Install and add your API key (once)**

```bash
uv sync                                          # from the repository root (Python 3.12)
cp 02_DocQA_RAG/.env.example 02_DocQA_RAG/.env   # then set OPENROUTER_API_KEY in .env
```

**2. Open the notebook from the repository root**

```bash
uv run --with jupyter jupyter lab
```

Open `Task2_DocQA_RAG.ipynb` and choose **Run All Cells**. This does not change the project
dependencies. The live cells need the API key from step 1. The first ingest embeds all 76 documents,
and the notebook then asks two questions and one out-of-scope question. The evaluation re-runs are
switched off by default because they make many API calls.

The notebook does not duplicate the code. It imports the `docqa` package, prints each module's source
with `inspect`, and runs the real command line tool, so it always matches the tested code.

## Optional: detailed review

The items in this section are optional. They are for a reviewer who wants to go deeper than the notebook.

- **Full project documentation:** [`02_DocQA_RAG/README.md`](02_DocQA_RAG/README.md) covers configuration, using your own documents, conversation memory, telemetry, the evaluation method, known limits and design decisions.
- **Command line, without the notebook:**

```bash
cd 02_DocQA_RAG
uv run python -m docqa ingest                      # embed data/source_md (76 files)
uv run python -m docqa ask "Who founded Insurellm?"
uv run python -m docqa chat                        # conversation with memory of the last 6 exchanges
uv run python -m docqa stats                       # timing and token usage of past questions
```

  Example answer with its source:

```
Insurellm was founded by Avery Lancaster[1].

Sources:
[1] about.md | About Insurellm, lines 1-5
```

  A question the documents cannot answer returns: `I could not find this in the loaded documents.`

- **Tests and evaluation:**

```bash
uv run pytest -q                       # offline tests, no API key needed
uv run python -m docqa.evaluate        # retrieval evaluation
uv run python -m docqa.evaluate_answers   # full answers (makes real API calls)
```


## Results

Evaluation set: 150 answerable questions in 7 categories, plus 5 unrelated and 6 near-topic
questions that must be refused. Scores are keyword proxies, not ground truth.

| top_k | recall | precision | MRR | nDCG |
|---|---|---|---|---|
| 4 | 0.929 | 0.773 | 0.933 | 0.945 |
| 10 (default) | 0.963 | 0.584 | 0.938 | 0.921 |

Full answers at `top_k` 10: keyword coverage 0.755, 100% of answers cited, all 11 unanswerable
questions refused. Known limits (aggregation and multi-hop questions) are listed in the project README.

## Where things are

| Path | What |
|---|---|
| `Task2_DocQA_RAG.ipynb` | Notebook walkthrough of the whole project (repository root) |
| `02_DocQA_RAG/docqa/` | Source code (loader, chunking, embeddings, store, retriever, answer, pipeline, cli, telemetry) |
| `02_DocQA_RAG/data/source_md/` | The 76 synthetic documents |
| `02_DocQA_RAG/evaluation/` | Evaluation questions |
| `02_DocQA_RAG/tests/` | Offline unit tests with fakes |

Back to the [project index](README.md).
