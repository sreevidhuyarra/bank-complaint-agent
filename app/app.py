"""Gradio UI — agent chat, a local explorer, and a corpus dashboard.

Three tabs, one process:

  Ask the team   the full agent run; needs GOOGLE_AI_API_KEY
  Explore        semantic search + risk scoring, entirely local; needs no key
  Dashboard      volume, severity mix and issue movement over the corpus

Explore and Dashboard deliberately work without an LLM. A cold Space with an
expired key still demonstrates the retrieval and psycholinguistics work, which
is the part that is actually mine.
"""

from __future__ import annotations

import os

import gradio as gr
import pandas as pd

from agents import orchestrator
from agents.kernel_setup import MissingApiKey, QuotaExhausted, llm_available
from data.ingest import get_index

SEVERITY_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟢"}

EXAMPLE_QUESTIONS = [
    "What are Wells Fargo customers most frustrated about with overdraft fees in "
    "the last two quarters, and how urgent does the language sound?",
    "Which American Express complaint issues are growing fastest, and how severe is the language?",
    "Are there signs of escalating legal threats in Wells Fargo mortgage complaints?",
    "Compare complaint volume and severity between Wells Fargo and American Express.",
]


# --------------------------------------------------------------------------- #
# Tab 1 — agent chat
# --------------------------------------------------------------------------- #

async def ask_agents(question: str, history: list, mode: str):
    question = (question or "").strip()
    if not question:
        yield history or [], "", "_Ask something first._"
        return

    history = (history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "_Routing to the agent team…_"},
    ]
    yield history, "", "_Running…_"

    try:
        brief = await orchestrator.answer(question, mode=mode)
    except (MissingApiKey, QuotaExhausted) as exc:
        # These carry an already-human-readable message — show it plainly
        # instead of wrapping it in a class-name/traceback-style dump.
        history[-1]["content"] = f"**Could not complete the run.**\n\n{exc}"
        yield history, "", "_Run failed._"
        return
    except Exception as exc:  # noqa: BLE001 - surface failures in the UI, don't crash the Space
        history[-1]["content"] = f"**Could not complete the run.**\n\n`{type(exc).__name__}: {exc}`"
        yield history, "", "_Run failed._"
        return

    footer = f"\n\n<sub>mode: {brief.mode}</sub>"
    if brief.error:
        footer += f"\n\n<sub>note: {brief.error}</sub>"
    history[-1]["content"] = brief.answer + footer
    yield history, "", brief.trace_markdown()


# --------------------------------------------------------------------------- #
# Tab 2 — local explorer
# --------------------------------------------------------------------------- #

def explore(query: str, company: str, product: str, severity: str, top_k: int):
    index = get_index()
    hits = index.search(
        query=query or "complaint",
        k=int(top_k),
        company="" if company == "All" else company,
        product=product or "",
        severity="" if severity == "All" else severity,
    )
    if not hits:
        return pd.DataFrame(columns=["complaint_id", "date", "company", "issue",
                                     "similarity", "risk_pct", "severity", "excerpt"]), \
            "_No complaints matched those filters._"

    table = pd.DataFrame(
        {
            "complaint_id": [h.complaint_id for h in hits],
            "date": [h.date_received for h in hits],
            "company": [h.company for h in hits],
            "issue": [h.issue for h in hits],
            "similarity": [round(h.score, 3) for h in hits],
            "risk_pct": [h.risk_percentile for h in hits],
            "severity": [f"{SEVERITY_EMOJI.get(h.severity, '')} {h.severity}" for h in hits],
            "excerpt": [h.excerpt(200) for h in hits],
        }
    )

    top = hits[0]
    score = index.scorer.score(top.narrative)
    detail = [
        f"### Top match · complaint {top.complaint_id}",
        f"**{top.company}** · {top.product}"
        f"{' / ' + top.sub_product if top.sub_product else ''} · {top.date_received}",
        f"**Issue:** {top.issue}",
        "",
        f"**Linguistic risk: {SEVERITY_EMOJI.get(score.severity, '')} {score.headline()}**",
        "",
        "| feature | strength |",
        "| --- | --- |",
    ]
    detail += [f"| {name} | {value:.2f} |" for name, value in score.features.items()]
    detail.append("")
    if score.slor_z is not None:
        detail.append(
            f"SLOR **{score.slor:.3f}** ({score.slor_z:+.2f} sd vs corpus mean) · "
            f"mean unigram surprisal {score.mean_surprisal:.2f} nats"
        )
    if score.drivers:
        detail.append(f"\n**Drivers:** {'; '.join(score.drivers)}")
    if score.matches:
        matched = "; ".join(f"_{k}_: {', '.join(v)}" for k, v in score.matches.items())
        detail.append(f"\n**Matched markers:** {matched}")
    detail.append(f"\n> {top.narrative[:1200]}{'…' if len(top.narrative) > 1200 else ''}")
    return table, "\n".join(detail)


# --------------------------------------------------------------------------- #
# Tab 3 — dashboard
# --------------------------------------------------------------------------- #

def dashboard(company: str):
    index = get_index()
    frame = index.frame if company == "All" else index.frame[index.frame["company"] == company]
    if frame.empty:
        empty = pd.DataFrame({"month": [], "complaints": []})
        return empty, empty, empty, "No complaints in scope."

    volume = (
        frame.groupby("month")
        .agg(complaints=("complaint_id", "count"), mean_risk=("risk_score", "mean"))
        .reset_index()
        .sort_values("month")
    )
    mix = (
        frame["severity"].value_counts()
        .rename_axis("severity").reset_index(name="complaints")
    )
    issues = (
        frame.groupby("issue")
        .agg(complaints=("complaint_id", "count"), mean_risk=("risk_score", "mean"))
        .reset_index()
        .sort_values("complaints", ascending=False)
        .head(10)
        .round({"mean_risk": 1})
    )
    summary = (
        f"**{len(frame):,} complaints** · {frame['date_received'].min().date()} → "
        f"{frame['date_received'].max().date()} · mean risk "
        f"**{frame['risk_score'].mean():.1f}/100** · "
        f"{(frame['severity'] == 'high').mean() * 100:.1f}% high severity\n\n"
        f"Severity bands for this corpus: medium ≥ {index.bands.medium}, "
        f"high ≥ {index.bands.high} (calibrated to the 70th/92nd percentile)."
    )
    return volume, mix, issues, summary


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def build_ui() -> gr.Blocks:
    index = get_index()
    companies = ["All"] + sorted(index.frame["company"].unique().tolist())
    key_banner = (
        ""
        if llm_available()
        else (
            "> ⚠️ **GOOGLE_AI_API_KEY is not set**, so the agent team is offline. "
            "The **Explore** and **Dashboard** tabs run entirely locally and still work."
        )
    )

    with gr.Blocks(title="Bank Complaint Intelligence Agent", fill_height=True) as demo:
        gr.Markdown(
            "# Bank Complaint Intelligence Agent\n"
            "Five Semantic Kernel agents — routing, semantic retrieval, psycholinguistic "
            "risk scoring, trend analysis and synthesis — over the public CFPB consumer "
            f"complaint database.\n\n**Indexed:** {index.describe()}\n\n{key_banner}"
        )

        with gr.Tab("Ask the team"):
            # Gradio 6 dropped `type=` — the OpenAI-style messages format this
            # code emits ({"role", "content"}) is now the only one.
            chat = gr.Chatbot(height=440, show_label=False)
            with gr.Row():
                question = gr.Textbox(
                    placeholder="Ask about a company, product, theme or trend…",
                    show_label=False,
                    scale=5,
                )
                mode = gr.Dropdown(
                    ["auto", "handoff", "pipeline"],
                    value="auto",
                    label="Mode",
                    scale=1,
                )
                send = gr.Button("Ask", variant="primary", scale=1)
            gr.Examples(EXAMPLE_QUESTIONS, inputs=question, label="Try one")
            with gr.Accordion("Agent trace — who did what", open=False):
                trace = gr.Markdown("_Run a question to see the handoffs._")

            send.click(ask_agents, [question, chat, mode], [chat, question, trace])
            question.submit(ask_agents, [question, chat, mode], [chat, question, trace])

        with gr.Tab("Explore"):
            gr.Markdown(
                "Semantic search plus the psycholinguistic scorer, running locally — "
                "no LLM involved, so this tab works with no API key."
            )
            with gr.Row():
                search_query = gr.Textbox(
                    label="Search",
                    value="overdraft fee charged after my deposit cleared",
                    scale=3,
                )
                search_company = gr.Dropdown(companies, value="All", label="Company")
                search_product = gr.Textbox(label="Product contains", placeholder="checking")
                search_severity = gr.Dropdown(
                    ["All", "high", "medium", "low"], value="All", label="Severity"
                )
                search_k = gr.Slider(3, 20, value=8, step=1, label="Results")
            search_button = gr.Button("Search", variant="primary")
            results = gr.Dataframe(label="Matches", wrap=True)
            detail = gr.Markdown()

            inputs = [search_query, search_company, search_product, search_severity, search_k]
            search_button.click(explore, inputs, [results, detail])
            search_query.submit(explore, inputs, [results, detail])

        with gr.Tab("Dashboard"):
            dash_company = gr.Dropdown(companies, value="All", label="Company")
            dash_summary = gr.Markdown()
            with gr.Row():
                volume_plot = gr.LinePlot(
                    x="month", y="complaints", title="Complaint volume by month", height=280
                )
                mix_plot = gr.BarPlot(
                    x="severity", y="complaints", title="Severity mix", height=280
                )
            issues_table = gr.Dataframe(label="Top issues", wrap=True)

            outputs = [volume_plot, mix_plot, issues_table, dash_summary]
            dash_company.change(dashboard, dash_company, outputs)
            demo.load(dashboard, dash_company, outputs)

        gr.Markdown(
            "<sub>Data: CFPB Consumer Complaint Database (public domain). Risk scores are "
            "linguistic signals, not determinations about any company or complainant.</sub>"
        )

    return demo


def main() -> None:
    # Render (and most non-HF-Spaces PaaS hosts) assign a dynamic port via the
    # PORT env var and route traffic to it; Hugging Face Spaces doesn't set
    # this, so the 7860 default (Gradio's own, and what Spaces expects) still
    # applies for local dev and HF hosting.
    port = int(os.environ.get("PORT", 7860))
    build_ui().launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
