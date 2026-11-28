# CartSense: a purchase-decision agent

A console agent that helps you choose what to buy. You type a request such as
`I want to buy a dishwasher`. The agent reads or builds your wishlist, searches the web for
current prices and specs, calculates running costs and a value score in Python, prints its
reasoning chain, and saves a recommendation to `output/<category>.md`.

This is Task 3 (Multi-Tool Autonomous AI Agent) of the Datheon internship. It is a LangChain
tool-calling agent (`langchain.agents.create_agent`) with five tools.

## How it works
request ──► agent loop (LangChain create_agent) ──► final-answer gate ──► output/<category>.md
│ Thought / Action / Observation printed live
├─ read_wishlist saved wishlist for the category (local file reader)
├─ ask_user interview, max 10 questions (interactive only)
├─ save_wishlist validated JSON, wishlists/ only (interactive only)
├─ web_search Tavily (if a key is set) or DuckDuckGo, with retry and backoff
└─ calculator safe arithmetic, running_cost, value_score, tco


| File | Job |
|---|---|
| `cartsense/config.py` | Environment settings, validated at startup |
| `cartsense/models.py` | Pydantic `Wishlist` model with constraints |
| `cartsense/tools.py` | The tools, input guards, retry, `ToolError` |
| `cartsense/scoring.py` | Pure functions: `running_cost()`, `value_score()`, `tco()` |
| `cartsense/agent.py` | Tool wiring, agent loop, step trace, final-answer gate |
| `cartsense/cli.py` | Command line, saving the recommendation |
| `prompts/system.md` | The system prompt |

## Setup

From the repository root:

```bash
uv sync
cp 03_CartSense/.env.example 03_CartSense/.env   # then edit it
```

`.env` needs `OPENROUTER_API_KEY`. `CARTSENSE_MODEL`, the base URL and the search region
(default `in-en`) are configurable. For better product links set `TAVILY_API_KEY`; the search
provider then defaults to Tavily. Without a key it uses DuckDuckGo. The Tavily free tier is
credit-limited, and one run can use up to 12 searches.

## Usage

Run from `03_CartSense/`:

```bash
uv run python -m cartsense.cli "I want to buy a phone"
uv run python -m cartsense.cli --no-interactive "I want to buy a phone"
```

- **Interactive** (a terminal): if a wishlist exists the agent asks "Reuse or update?";
  otherwise it interviews you (6 to 10 questions), saves your answers, then researches.
- **Non-interactive** (`--no-interactive`, or stdin is not a terminal): `ask_user` and
  `save_wishlist` are removed. The agent uses the saved wishlist or states its defaults.
- To try the sample: `mkdir -p wishlists && cp data/synthetic/wishlists/phone.json wishlists/`.
  Real wishlists in `wishlists/` and results in `output/` are git-ignored.

Exit codes: 0 done, 1 failed or step limit reached, 2 configuration error, 130 cancelled.

## Safeguards

Small models skip steps and invent numbers, so rules that matter are enforced in code:

- **Safe calculator:** parsed with `ast`, never `eval`; caps on length, exponent and result.
- **`save_wishlist`** writes only `wishlists/<cleaned category>.json`, validated, size-capped.
  No tool can move, edit or delete other files.
- **Caps:** at most 10 questions and 12 searches (8 general, 4 store searches).
- **Duplicate calls** return a short note instead of running again.
- **Final-answer gate:** an answer is rejected, and the model is sent back to fix it, if it
  skips required steps, cites a URL that no search returned, lacks a product-page link, or
  quotes a price or table score that no tool result supports.
- **Retries:** searches retry with backoff; the model client retries transient API errors.
- **Calculator log:** the saved file ends with every successful calculator call, verbatim.

## Tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

All tests use scripted fake models and fake search functions. No test touches the network or
spends search credits.

## Web UI

A Gradio front end for the same agent, in `ui/app.py`. It is presentation only, so the agent code is unchanged.

From `03_CartSense/`, after completing Setup:

```bash
uv run --with "gradio>=5,<6" python ui/app.py
```

Open the local URL that Gradio prints in the terminal (usually http://127.0.0.1:7860).

The UI reads the same `.env` settings as the CLI, so `OPENROUTER_API_KEY` must be set.

## Limitations

- Checks confirm that a number appears in the search results, not that it belongs to the
  model being discussed, so a price or spec can be attributed to the wrong product.
- Specs, ratings, warranty and dealbreaker claims are not verified against sources.
- A model with no product page in the results can still end up with a listing-page link.
- Prices and stock change; links may go out of stock.
- Results depend on the search provider and on the model's discipline; a run can fail the
  gate and exit with code 1 instead of producing an answer.
- "Thought" lines appear only when the model writes text before a tool call.
