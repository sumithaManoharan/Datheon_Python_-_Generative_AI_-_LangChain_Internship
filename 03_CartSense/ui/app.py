"""Gradio front end for CartSense. Presentation only: the agent code is not modified.

Run from 03_CartSense/:
    uv run --with "gradio>=5,<6" python ui/app.py
"""

import html
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from cartsense.agent import (  # noqa: E402
    AgentError,
    Step,
    build_model,
    build_requirements,
    build_tools,
    load_system_prompt,
    mode_note,
    run_agent,
)
from cartsense.cli import OUTPUT_DIR, guess_category, save_recommendation  # noqa: E402
from cartsense.config import ConfigError, load_config  # noqa: E402
from cartsense.tools import UserPrompter  # noqa: E402

ANSWER_TIMEOUT_SECONDS = 900
MAX_OBSERVATION_CHARS = 500
LABELS = {
    "thought": "💭 Thought",
    "action": "🔧 Action",
    "observation": "👁 Observation",
    "final": "✅ Final answer",
}

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.cs-hero { text-align: center; padding: 18px 0 6px; }
.cs-hero h1 { font-size: 2.3rem; margin: 0; letter-spacing: -0.02em; }
.cs-hero p { opacity: 0.75; margin: 6px 0 0; }
.cs-trace { max-height: 560px; overflow-y: auto; padding-right: 4px; }
.cs-card { border-radius: 12px; padding: 10px 14px; margin: 8px 0;
           border-left: 5px solid #94a3b8; background: rgba(148,163,184,0.12); }
.cs-card pre { white-space: pre-wrap; word-break: break-word; margin: 6px 0 0;
               font-size: 0.82rem; font-family: ui-monospace, monospace; }
.cs-tag { font-weight: 600; font-size: 0.85rem; }
.cs-thought { border-color: #a78bfa; background: rgba(167,139,250,0.14); }
.cs-action { border-color: #6366f1; background: rgba(99,102,241,0.14); }
.cs-observation { border-color: #14b8a6; background: rgba(20,184,166,0.12); }
.cs-final { border-color: #22c55e; background: rgba(34,197,94,0.16); }
.cs-empty { opacity: 0.6; padding: 24px; text-align: center; }
"""

_answers: "queue.Queue[str]" = queue.Queue()


def card(step: Step) -> str:
    """Render one step of the reasoning chain as an HTML card."""
    if step.kind == "final":
        body = "The answer passed every check."
    else:
        body = html.escape(step.text)
        if step.kind == "observation" and len(body) > MAX_OBSERVATION_CHARS:
            body = body[:MAX_OBSERVATION_CHARS] + " …"
    label = LABELS.get(step.kind, step.kind)
    return (
        f'<div class="cs-card cs-{step.kind}"><span class="cs-tag">{label}</span>'
        f"<pre>{body}</pre></div>"
    )


def render_trace(cards: list[str]) -> str:
    """Newest step first, inside a scrollable box."""
    if not cards:
        return '<div class="cs-empty">The reasoning chain will appear here.</div>'
    return '<div class="cs-trace">' + "".join(reversed(cards)) + "</div>"


def _worker(request: str, interactive: bool, events: "queue.Queue[tuple]") -> None:
    """Run the real agent in a thread; report steps, questions and the result as events."""

    def ask(prompt: str) -> str:
        events.put(("ask", prompt.strip().removeprefix("?").removesuffix(">").strip()))
        try:
            return _answers.get(timeout=ANSWER_TIMEOUT_SECONDS)
        except queue.Empty:
            raise EOFError from None

    try:
        config = load_config()
        prompt = f"{load_system_prompt()}\n\n{mode_note(interactive)}"
        prompter = UserPrompter(ask) if interactive else None
        tools = build_tools(
            prompter=prompter,
            region=config.search_region,
            require_price_evidence=True,
            provider=config.search_provider,
            tavily_api_key=config.tavily_api_key,
        )
        result = run_agent(
            build_model(config),
            config,
            request,
            tools=tools,
            system_prompt=prompt,
            on_step=lambda step: events.put(("step", step)),
            requirements=build_requirements(interactive),
        )
        events.put(("done", result))
    except (ConfigError, AgentError) as exc:
        events.put(("error", str(exc)))
    except Exception as exc:  # keep the page alive whatever happens
        events.put(("error", f"Unexpected error: {exc}"))


def start(request: str):
    """Start a run and stream the trace, interview questions and the final recommendation."""
    request = (request or "").strip()
    if not request:
        raise gr.Error("Type what you want to buy, for example: I want to buy a dishwasher")
    events: "queue.Queue[tuple]" = queue.Queue()
    while not _answers.empty():
        _answers.get_nowait()
    threading.Thread(target=_worker, args=(request, True, events), daemon=True).start()

    cards: list[str] = []
    busy = gr.update(interactive=False, value="Working…")
    yield render_trace(cards), gr.update(visible=False), gr.update(visible=False), "", busy
    while True:
        try:
            kind, payload = events.get(timeout=0.4)
        except queue.Empty:
            continue
        if kind == "step":
            cards.append(card(payload))
            yield render_trace(cards), gr.update(), gr.update(), gr.update(), gr.update()
        elif kind == "ask":
            question = f"### 💬 CartSense asks\n\n**{payload}**"
            yield (
                render_trace(cards),
                gr.update(value=question, visible=True),
                gr.update(visible=True),
                gr.update(),
                gr.update(),
            )
        else:
            break

    idle = gr.update(interactive=True, value="Find my best buy")
    hide = gr.update(visible=False)
    if kind == "error":
        yield render_trace(cards), hide, hide, f"### ⚠️ Could not run\n\n{payload}", idle
        return
    result = payload
    if result.status != "done":
        message = f"### ⚠️ No recommendation\n\n{result.final_answer}"
        yield render_trace(cards), hide, hide, message, idle
        return
    path = save_recommendation(result, request, guess_category(request), OUTPUT_DIR)
    yield render_trace(cards), hide, hide, path.read_text(encoding="utf-8"), idle


def send(answer: str):
    """Hand the buyer's reply to the waiting agent and clear the answer box."""
    _answers.put((answer or "").strip())
    return gr.update(value=""), gr.update(visible=False), gr.update(visible=False)


def build_app() -> gr.Blocks:
    """Lay out the page."""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS, title="CartSense") as demo:
        gr.HTML(
            '<div class="cs-hero"><h1>🛒 CartSense</h1>'
            "<p>An AI agent that interviews you, searches the web, scores the options in "
            "Python and shows its reasoning.</p></div>"
        )
        with gr.Row():
            request = gr.Textbox(
                label="What do you want to buy?",
                placeholder="I want to buy a dishwasher",
                scale=4,
            )
        run = gr.Button("Find my best buy", variant="primary", size="lg")
        gr.Examples(
            ["I want to buy a dishwasher", "I want to buy a phone", "I want to buy an office chair"],
            inputs=request,
        )
        question = gr.Markdown(visible=False)
        with gr.Row(visible=False) as answer_row:
            answer = gr.Textbox(show_label=False, placeholder="Type your answer…", scale=5)
            reply = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Reasoning chain")
                trace = gr.HTML(render_trace([]))
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Recommendation")
                final = gr.Markdown("The recommendation will appear here.")
        gr.Markdown(
            "<small>Each run uses web-search credits. Prices and stock change; "
            "check the product page before buying.</small>"
        )

        outputs = [trace, question, answer_row, final, run]
        run.click(start, [request], outputs)
        request.submit(start, [request], outputs)
        reply.click(send, answer, [answer, answer_row, question])
        answer.submit(send, answer, [answer, answer_row, question])
    return demo


if __name__ == "__main__":
    build_app().queue().launch()
