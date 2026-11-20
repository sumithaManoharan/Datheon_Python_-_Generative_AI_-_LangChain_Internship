# Evaluation data

- `tests.jsonl`: 150 questions with `keywords`, a `reference_answer` and a `category`
  (direct_fact, numerical, temporal, relationship, comparative, spanning, holistic).
- `unanswerable.jsonl`: 5 questions unrelated to the documents; the system must refuse them.
- `near_topic.jsonl`: 6 questions about Insurellm that the documents do not answer; the system
  must refuse them too.

Scoring is keyword-based (see `docqa/retrieval_metrics.py` and `docqa/evaluate_answers.py`), so
results are indicators for comparing settings, not ground truth. Generated reports go to
`results/`, which is not committed.

Run: `uv run python -m docqa.evaluate` (retrieval) and `uv run python -m docqa.evaluate_answers`
(full answers; makes real API calls).
