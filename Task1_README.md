# Task 1: Structured LLM Prompt Pipeline & JSON Extractor

**Project:** Hirelyzer, in [`01_Hirelyzer/`](01_Hirelyzer/) (full documentation: [`01_Hirelyzer/README.md`](01_Hirelyzer/README.md))

## Objective

Build a structured prompt pipeline that reads unstructured text (here: resumes) and returns
**valid JSON** that matches a strict schema. Output the model gets wrong is sent back for repair,
and a document that still fails is reported, never silently dropped.

## Key Concepts

- Prompt engineering: a parameterized template, few-shot examples and the schema inside the prompt
- Structured LLM output: JSON mode at `temperature=0.0`
- JSON extraction and validation: fenced or surrounded JSON is parsed, then checked with Pydantic
- Output repair loop: validation errors are sent back to the model (up to 2 repair rounds)
- Error handling: retries with exponential backoff for transient API errors only, typed exceptions,
  guards for missing, empty, oversized, corrupt and unsupported files
- Evaluation against ground truth, plus telemetry (latency, tokens, retries)

## Pipeline

```
input (.pdf / .docx / .txt / raw text / .jpg / .png)
   -> readers.py / vision.py   guards + text extraction
   -> prompts.py               template + few-shot + schema
   -> llm.py                   OpenAI-compatible client, retry/backoff, telemetry
   -> parser.py                strips fences and prose -> JSON
   -> schema.py                Pydantic validation
        invalid? -> send the error back to the model (max 2 repair rounds)
   -> ExtractionResult { status, data, errors, attempts, tokens, latency }
```

## Models

- Text input: `openai/gpt-oss-20b` through OpenRouter (any OpenAI-compatible endpoint works)
- Image input: `qwen/qwen3.7-flash`

## How to run: the notebook is the main guide

[`Task1_JSON_Extractor.ipynb`](Task1_JSON_Extractor.ipynb) is the main way to run and understand
this project. It walks through everything in order: objective and key concepts, setup, pipeline
architecture, a tour of every module, the Pydantic schema, the prompt and few-shot files, a live
extraction, error handling, telemetry, evaluation results, tests, design decisions and limitations.

**1. Install and add your API key (once)**

```bash
uv sync                                   # from the repository root (Python 3.12)
cp 01_Hirelyzer/.env.example 01_Hirelyzer/.env   # then put your API key in .env
```

**2. Open the notebook from the repository root**

```bash
uv run --with jupyter jupyter lab
```

Open `Task1_JSON_Extractor.ipynb` and choose **Run All Cells**. This does not change the project
dependencies. The live-run cells need the API key from step 1 and make one API call.

The notebook does not duplicate the code. It imports the `hirelyzer` package, prints each module's
source with `inspect`, and runs the real command line tool, so it always matches the tested code.

## Optional: detailed review

The items in this section are optional. They are for a reviewer who wants to go deeper than the notebook.

- **Full project documentation:** [`01_Hirelyzer/README.md`](01_Hirelyzer/README.md) covers configuration, the schema, evaluation method, telemetry, design decisions and limitations.
- **Command line, without the notebook:**

```bash
cd 01_Hirelyzer
uv run python -m hirelyzer.cli data/synthetic/resumes/01_rohan_deshmukh_classic_single_column.pdf
```

  The JSON is written to `output/<file name>.json` and a run summary is printed. A non-resume
  (an invoice, for example) is returned as `{"is_resume": false}`.

- **Tests and evaluation:**

```bash
uv run pytest -q                       # mocked tests, no network or API key needed
uv run python -m hirelyzer.evaluate    # run the model on the synthetic set and score it
```

## Results

Baseline on 15 synthetic documents (8 PDF, 3 images, 4 DOCX; one is an invoice), 0 failed documents:

| Metric | Score |
|---|---|
| Overall (mean) | 0.972 |
| is_resume, name, email, phone, location | 1.000 |
| Skills F1 | 0.962 |
| Experience F1 | 0.976 |
| Education F1 | 1.000 |

These are upper bounds: the documents are synthetic and the ground truth was written for them.

## Where things are

| Path | What |
|---|---|
| `Task1_JSON_Extractor.ipynb` | Notebook walkthrough of the whole project (repository root) |
| `01_Hirelyzer/hirelyzer/` | Source code (config, readers, prompts, llm, parser, schema, extractor, evaluate, cli) |
| `01_Hirelyzer/tests/` | Offline unit tests |
| `01_Hirelyzer/data/synthetic/` | Synthetic sample resumes and ground truth |
| `01_Hirelyzer/evaluation/` | Saved evaluation runs |

Back to the [project index](README.md).
