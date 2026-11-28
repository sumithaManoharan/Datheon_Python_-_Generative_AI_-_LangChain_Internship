# Task 3: Multi-Tool Autonomous AI Agent

**Project:** CartSense, in [`03_CartSense/`](03_CartSense/) (full documentation: [`03_CartSense/README.md`](03_CartSense/README.md))

## Objective

Build an autonomous AI agent that chooses and uses several tools to complete a task. CartSense
helps you decide what to buy: you type a request such as `I want to buy a dishwasher`, and the
agent builds or reads your wishlist, searches the web for current prices and specs, calculates
running costs and a value score in Python, prints its reasoning chain, and saves a recommendation
to `output/<category>.md`.

## Key Concepts

- LangChain tool-calling agent (`langchain.agents.create_agent`)
- Multi-tool task execution, with the reasoning chain (Thought / Action / Observation) printed live
- Web search tool: Tavily when a key is set, otherwise DuckDuckGo, with retry and backoff
- Calculator tool: safe arithmetic parsed with `ast` (never `eval`), plus `running_cost`, `value_score` and `tco`
- Local file tools: reading and saving a validated wishlist (Pydantic model)
- Guardrails enforced in code: step and search caps, duplicate-call detection, a final-answer gate
- LLM: `openai/gpt-oss-20b` through OpenRouter (configurable with `CARTSENSE_MODEL`)

## Tools

| Tool | Job |
|---|---|
| `read_wishlist` | Reads the saved wishlist for the category (local file reader) |
| `ask_user` | Interviews the user, at most 10 questions (interactive mode only) |
| `save_wishlist` | Saves validated JSON to `wishlists/` only (interactive mode only) |
| `web_search` | Tavily or DuckDuckGo, at most 12 searches per run |
| `calculator` | Safe arithmetic and the scoring functions |

## Agent flow

```
request -> agent loop (create_agent) -> final-answer gate -> output/<category>.md
              Thought / Action / Observation printed live, tools called as needed
```

The final-answer gate sends the model back to fix its answer if it skips required steps, cites a
URL that no search returned, has no product-page link, or quotes a price or score that no tool
result supports.

## How to run

```bash
uv sync                                              # from the repository root (Python 3.12)
cp 03_CartSense/.env.example 03_CartSense/.env       # set OPENROUTER_API_KEY (TAVILY_API_KEY is optional)

cd 03_CartSense
uv run python -m cartsense.cli "I want to buy a phone"
uv run python -m cartsense.cli --no-interactive "I want to buy a phone"
uv run pytest                                        # offline tests with scripted fake models
```

To try the bundled sample wishlist: `mkdir -p wishlists && cp data/synthetic/wishlists/phone.json wishlists/`.
An optional Gradio web UI is in `03_CartSense/ui/app.py` (see the project README).

## Where things are

| Path | What |
|---|---|
| `03_CartSense/cartsense/` | Source code (config, models, tools, scoring, agent, cli) |
| `03_CartSense/prompts/system.md` | The system prompt |
| `03_CartSense/ui/app.py` | Optional Gradio front end |
| `03_CartSense/tests/` | Offline tests, no network or search credits |

## Limitations

Price and spec checks confirm that a number appears in the search results, not that it belongs to
the product being discussed. Prices and stock change. A run can fail the final-answer gate and exit
with code 1 instead of producing an answer.

Back to the [project index](README.md).
