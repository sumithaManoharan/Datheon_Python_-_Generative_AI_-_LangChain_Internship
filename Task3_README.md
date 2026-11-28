# Task 3: Multi-Tool Autonomous AI Agent

**Project:** CartSense, in [`O3_CartSense/`](03_CartSense/) (full documentation: [`O3_CartSense/README.md`](O3_CartSense/README.md))

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
- Local file tools: reading and saving a wishlist validated by a Pydantic model
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

## How to run: the notebook is the main guide

[`O3_Autonomous_AI_Agent.ipynb`](O3_Autonomous_AI_Agent.ipynb) is the main way to run and
understand this project. It walks through everything in order: objective and key concepts, setup,
architecture and tools, a tour of every module, the system prompt, the Pydantic wishlist model, the
scoring functions, a live agent run, the saved recommendation, safeguards, tests, design decisions
and limitations.

**1. Install and add your API key (once)**

```bash
uv sync                                                  # from the repository root (Python 3.12)
cp 03_CartSense/.env.example O3_CartSense/.env     # set OPENROUTER_API_KEY (TAVILY_API_KEY is optional)
```

**2. Open the notebook from the repository root**

```bash
uv run --with jupyter jupyter lab
```

Open `O3_Autonomous_AI_Agent.ipynb` and choose **Run All Cells**. This does not change the
project dependencies. The live-run cell copies the sample phone wishlist into `wishlists/`, then runs
the agent in non-interactive mode. It needs the API key from step 1 and internet access, and one run
can use up to 12 searches (the Tavily free tier is credit-limited). A run can fail the final-answer
gate and exit with code 1. That is by design, and running the cell again usually works.

The notebook does not duplicate the code. It imports the `cartsense` package, prints each module's
source with `inspect`, and runs the real command line tool, so it always matches the tested code.

## Optional: detailed review

The items in this section are optional. They are for a reviewer who wants to go deeper than the notebook.

- **Full project documentation:** [`O3_CartSense/README.md`](O3_CartSense/README.md) covers setup, usage, safeguards, tests, the web UI and limitations.
- **Command line, without the notebook:**

```bash
cd O3_CartSense
uv run python -m cartsense.cli "I want to buy a phone"                    # interactive interview
uv run python -m cartsense.cli --no-interactive "I want to buy a phone"   # uses the saved wishlist
```

  To try the bundled sample wishlist: `mkdir -p wishlists && cp data/synthetic/wishlists/phone.json wishlists/`.

- **Web UI (Gradio):** `uv run --with "gradio>=5,<6" python ui/app.py` from `O3_CartSense/`.
- **Tests:**

```bash
uv run pytest                 # offline tests with scripted fake models, no search credits spent
```

## Where things are

| Path | What |
|---|---|
| `O3_Autonomous_AI_Agent.ipynb` | Notebook walkthrough of the whole project (repository root) |
| `03_CartSense/cartsense/` | Source code (config, models, tools, scoring, agent, cli) |
| `03_CartSense/prompts/system.md` | The system prompt |
| `03_CartSense/ui/app.py` | Optional Gradio front end |
| `03_CartSense/tests/` | Offline tests, no network or search credits |

## Limitations

Price and spec checks confirm that a number appears in the search results, not that it belongs to
the product being discussed. Prices and stock change. A run can fail the final-answer gate and exit
with code 1 instead of producing an answer.

Back to the [project index](README.md).
