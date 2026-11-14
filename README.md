# Hirelyzer: Structured JSON Extraction from Resumes

Task 1 of the Datheon Python + Generative AI & LangChain internship. Hirelyzer reads a resume (PDF, DOCX, TXT, JPG/PNG or raw text), asks an LLM to extract its content, and returns JSON validated against a strict Pydantic schema. Output the model gets wrong is sent back for repair; a document that still fails is reported, never silently dropped. An evaluation script scores the output against ground truth and records every run.

## Quick start

```bash
uv sync                      # from the repository root (Python 3.12)
cd 01_Hirelyzer
cp .env.example .env         # then put your API key in .env
uv run python -m hirelyzer.cli data/synthetic/resumes/01_rohan_deshmukh_classic_single_column.pdf
```

The JSON is written to `output/<file name>.json` and a run summary is printed.

## Configuration

Settings come from environment variables (or `.env`); a missing key stops the program at startup with a readable message.

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` or `OPENAI_API_KEY` | required | API key (the first one set wins) |
| `HIRELYZER_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `HIRELYZER_TEXT_MODEL` | `openai/gpt-oss-20b` | Model for text input |
| `HIRELYZER_VISION_MODEL` | `qwen/qwen3.7-flash` | Model for image input |
| `HIRELYZER_TEMPERATURE` | `0.0` | Sampling temperature (0 to 2) |
| `HIRELYZER_MAX_RETRIES` | `3` | Retries on transient API errors |
| `HIRELYZER_REPAIR_ATTEMPTS` | `2` | Repair rounds for invalid output |
| `HIRELYZER_MAX_FILE_MB` / `HIRELYZER_MAX_TEXT_CHARS` | `10` / `60000` | Input size limits |

## Usage

```bash
uv run python -m hirelyzer.cli resume.pdf scan.jpg folder/     # files and folders (batch)
uv run python -m hirelyzer.cli --text "Jane Doe, Python developer ..."
cat resume.txt | uv run python -m hirelyzer.cli -               # stdin
uv run python -m hirelyzer.cli resumes/ --output-dir out --telemetry-log logs/run.jsonl -v
```

A bad file never stops a batch. Exit code: `0` all succeeded, `1` at least one document failed, `2` configuration or usage problem. A non-resume (an invoice, say) is detected and returned as `{"is_resume": false}` with every other field empty.

## How it works

```
input: .pdf / .docx / .txt / raw text            .jpg / .png
        |                                            |
   readers.py (guards, text extraction)       vision.py (guards, data URL)
        |                                            |
        +------------> prompts.py <------------------+   template + few-shot + schema
                          |
                    llm.py: call with retry/backoff, telemetry (temperature 0.0, JSON mode)
                          |
                    parser.py: fences/prose stripped -> JSON
                          |
                    schema.py: Pydantic validation
                          |  invalid? send the error back to the model (max 2 repair rounds)
                          v
                    ExtractionResult { status, data, errors, attempts, tokens, latency }
```

| Module | Job |
|---|---|
| `config.py` | Settings from the environment, validated once at startup |
| `errors.py` | Typed exceptions (`ConfigError`, `UnsupportedFileError`, `EmptyDocumentError`, `InputTooLargeError`, `LLMCallError`, `OutputValidationError`) |
| `readers.py`, `vision.py` | PDF/DOCX/TXT to text, images to data URLs; guards for missing, empty, oversized, corrupt and unsupported files |
| `prompts.py` + `prompts/` | Parameterized template; template and few-shot examples live in files and are validated at startup |
| `llm.py` | OpenAI-compatible client; retries only transient errors (429, timeouts, connection errors, 5xx) with exponential backoff and jitter; never retries auth or bad requests |
| `parser.py` | Raw model text to a JSON object (handles code fences, surrounding prose; truncated JSON fails loudly) |
| `schema.py` | Pydantic models and validators |
| `extractor.py` | The stages, the repair loop and the result envelope |
| `telemetry.py` | Per-request latency, tokens and retries |
| `evaluate.py` | Scoring against ground truth and run history |
| `cli.py` | Argument parsing, batch isolation, summary, exit codes |

Clients are passed in (dependency injection) and nothing runs at import time, so every stage is testable with fakes.

### Schema

`is_resume`, `language`, `name`, `headline`, `email`, `phone`, `location`, `skills`, `experience` (company, job_title, job_location, start_date, end_date, is_current, highlights), `education` (institution, degree, field_of_study, start_year, end_year, grade), `certifications`, `projects` (project_title, project_description, skills_used), `languages`.

Rules: unknown fields are rejected; email and phone patterns; dates `YYYY-MM`, years `YYYY`; end not before start; a current job has no end date; blanks become `null` and lists are stripped and de-duplicated. Everything except `is_resume` is optional, so a non-resume validates as `is_resume=false` with empty fields.

## Evaluation

```bash
uv run python -m hirelyzer.evaluate                         # run the model on data/synthetic and score it
uv run python -m hirelyzer.evaluate --outputs some/dir      # score saved outputs, no API calls
uv run python -m hirelyzer.evaluate --unlabeled my/dir      # resumes with no ground truth
```

Flat fields are exact matches after normalization (phone by digits, language names to codes). List fields use set precision, recall and F1. Jobs, schools and projects are paired by company and title, institution and degree, or project title, then compared field by field. Every run is saved under `evaluation/runs/<time>/` with the model, the prompt-file hash, tokens and latency, and appended to `evaluation/runs/history.csv`; the report shows the change since the previous run. A failed document counts as zero in "all docs".

For resumes without ground truth, accuracy cannot be computed. `--unlabeled` reports only what can be measured (valid-JSON rate, repair rounds, empty fields) and prints counts, never content.

### Baseline (15 synthetic documents: 8 PDF, 3 images, 4 DOCX; one is an invoice)

Run `2026-11-12T12-15-16`, model `openai/gpt-oss-20b` (images: `qwen/qwen3.7-flash`), 0 failed documents.

| Metric | Score | Metric | Score |
|---|---|---|---|
| **overall (mean)** | **0.972** | certifications F1 | 1.000 |
| is_resume | 1.000 | languages F1 | 0.857 |
| name / email / phone / location | 1.000 | experience F1 | 0.976 |
| language | 1.000 | experience fields | 1.000 |
| headline | 0.929 | experience highlights F1 | 0.975 |
| skills precision / recall | 1.000 / 0.950 | education F1 / fields | 1.000 / 0.955 |
| skills F1 | 0.962 | projects F1 / skills used F1 | 1.000 / 0.857 |

Tokens: 57,873 prompt + 31,357 completion. Average latency: 23.3 s per document (it varies a lot between runs).

**Read these as an upper bound.** The documents are synthetic and the ground truth was written for them. Real resumes will score lower.

## Telemetry

Each LLM request writes one JSON line (`logs/telemetry.jsonl` by default) and the CLI prints a summary:

```json
{"source": "text-input", "model": "openai/gpt-oss-20b", "pipeline": "text", "latency_ms": 1991.9, "prompt_tokens": 3072, "completion_tokens": 111, "retries": 0, "success": true, "error": "", "timestamp": "2026-11-11T11:48:29+00:00"}
```

## Tests

```bash
uv run pytest -q          # mocked, no network: parser, schema, retry/backoff, repair loop, readers, CLI, evaluation ...
uv run ruff check . && uv run ruff format --check .
```

## Design decisions

- **openai SDK, not LangChain**: the Task 1 brief names the OpenAI/Gemini SDKs.
- **JSON mode plus the schema in the prompt, then Pydantic validation**: the vision model has no JSON-schema enforcement, so the schema is not trusted to the API.
- **`temperature=0.0`** makes output deterministic-leaning, not hallucination-proof; run-to-run variation was observed.
- **Retries live in visible code** (the SDK's own retries are off), so they are logged and tested.
- **A failed extraction is a result, not an exception**, so batches continue and the summary lists what failed.

## Limitations

- Accuracy is measured on synthetic data only; there is no ground truth for real resumes.
- The image path sends the whole image to the model provider (the text path sends extracted text). Do not use either with documents you may not share.
- Scanned PDFs go through OCR in `pymupdf4llm`, which can misplace sections on complex layouts.
- `links` and `summary` appear in the ground truth but are not extracted.

## Approaches tried and dropped

ScaleDown Extract (confidence only for flat fields, and the provider route does not forward its document input), RapidOCR confidence gating (it measures how well pixels were read, not where a field belongs), and converting images to PDF (it adds no text layer).
