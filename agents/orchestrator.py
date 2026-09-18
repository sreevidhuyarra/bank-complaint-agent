"""Orchestrator — routes an analyst question across the specialist team.

Two execution modes share the same five agents:

`handoff`   Semantic Kernel's HandoffOrchestration. The Orchestrator agent is
            given `transfer_to_*` functions and decides, per question, which
            specialists to involve and in what order. This is the pattern the
            project is really about, and the one AI-103 examines.

`pipeline`  A deterministic retrieval → risk → trend → synthesis sequence with
            no routing model in the loop. Handoff routing depends on the model
            emitting a transfer call at every step; on a free tier that
            occasionally stalls, and a portfolio demo that dies live is worse
            than one that degrades. `auto` runs handoff and falls back here.

Both modes end with the Synthesis agent writing the brief.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Callable

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents.orchestration.handoffs import (
    HandoffOrchestration,
    OrchestrationHandoffs,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent

from agents import retrieval_agent, risk_agent, synthesis_agent, trend_agent
from agents.retrieval_agent import format_hits
from agents.kernel_setup import (
    MissingApiKey,
    QuotaExhausted,
    build_kernel,
    default_arguments,
    is_transient_error,
    llm_available,
    with_rate_limit_retry,
)

NAME = "Orchestrator"

INSTRUCTIONS = """You are the Orchestrator on a bank complaint-analysis team. You
route work; you do not answer questions yourself and you have no data tools.

Your specialists:
- RetrievalAgent — finds complaint narratives by semantic search. Only it can
  read the complaint index, so almost every question starts here.
- LinguisticRiskAgent — scores complaint language for urgency, distress and
  escalation risk. Needs complaint IDs from RetrievalAgent first.
- TrendAgent — complaint volume, issue mix and quarter-over-quarter movement.
  Works from filters alone and does not need retrieval.
- SynthesisAgent — writes the final cited brief. Always last.

Routing rules:
- A question about what customers are saying, or about themes → RetrievalAgent.
- A question mentioning urgency, severity, frustration, tone, escalation or
  "how bad is it" → RetrievalAgent first, then LinguisticRiskAgent.
- A question about volume, growth, spikes, "rising", "more than last quarter"
  → TrendAgent.
- Most real questions need two or three of them. Route to each in turn.
- When the specialists have reported, transfer to SynthesisAgent. Never write
  the brief yourself and never call complete_task before Synthesis has run.
- Some questions ask for something none of your specialists can supply — e.g.
  real-time account transactions, backend fee-calculation logic, or auditing a
  specific account, none of which exist in the indexed CFPB complaint data.
  Never let that end the conversation with no brief: route to RetrievalAgent
  anyway to check for related complaint evidence, then transfer to
  SynthesisAgent regardless — it is instructed to say plainly when a question
  is outside what the data supports. A clear "this data can't answer that"
  brief is always the right outcome, never silence.
- If SynthesisAgent hands control back to you, do not restart from
  RetrievalAgent — the specialists already ran and their findings are still
  earlier in this same conversation. Transfer straight back to SynthesisAgent;
  re-running the whole chain wastes work and risks looping.

Transfer immediately. Do not narrate your routing decision at length.
"""


@dataclass
class Turn:
    """One agent's contribution, kept for the trace panel in the UI."""

    agent: str
    content: str


@dataclass
class Brief:
    question: str
    answer: str
    mode: str
    trace: list[Turn] = field(default_factory=list)
    error: str | None = None

    def trace_markdown(self) -> str:
        if not self.trace:
            return "_No agent activity recorded._"
        blocks = [f"**{turn.agent}**\n\n{turn.content}" for turn in self.trace]
        return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Team construction
# --------------------------------------------------------------------------- #

def build_orchestrator_agent() -> ChatCompletionAgent:
    return ChatCompletionAgent(
        kernel=build_kernel(),
        # "required": the Orchestrator's only job is routing — it must never
        # just answer in prose instead of calling a transfer_to_* function.
        arguments=default_arguments(tool_choice="required"),
        name=NAME,
        description="Routes analyst questions to the right specialist agents.",
        instructions=INSTRUCTIONS,
    )


def build_team() -> dict[str, ChatCompletionAgent]:
    """The five agents. Order matters: the first member receives the question."""
    return {
        NAME: build_orchestrator_agent(),
        retrieval_agent.NAME: retrieval_agent.build_agent(),
        risk_agent.NAME: risk_agent.build_agent(),
        trend_agent.NAME: trend_agent.build_agent(),
        synthesis_agent.NAME: synthesis_agent.build_agent(),
    }


def build_handoffs() -> OrchestrationHandoffs:
    """The routing graph.

    Specialists can hand back to the Orchestrator or straight on to Synthesis,
    which lets a simple question skip a routing round-trip. Only the Orchestrator
    can reach every specialist, so the routing decision stays in one place.
    """
    return (
        OrchestrationHandoffs()
        .add_many(
            source_agent=NAME,
            target_agents={
                retrieval_agent.NAME: "Find complaint narratives matching the question",
                risk_agent.NAME: "Score complaint language for urgency and escalation risk",
                trend_agent.NAME: "Compute complaint volume and quarter-over-quarter movement",
                synthesis_agent.NAME: "Write the final cited brief from what has been gathered",
            },
        )
        .add_many(
            source_agent=retrieval_agent.NAME,
            target_agents={
                NAME: "Hand back for routing when another specialist is needed",
                risk_agent.NAME: "Score the severity of the complaints just retrieved",
                trend_agent.NAME: "Get volume context for the complaints just retrieved",
                synthesis_agent.NAME: "Write the brief when retrieval alone answers the question",
            },
        )
        .add_many(
            source_agent=risk_agent.NAME,
            target_agents={
                NAME: "Hand back for routing when another specialist is needed",
                trend_agent.NAME: "Add volume context to the severity read",
                synthesis_agent.NAME: "Write the brief once severity has been scored",
            },
        )
        .add_many(
            source_agent=trend_agent.NAME,
            target_agents={
                NAME: "Hand back for routing when another specialist is needed",
                retrieval_agent.NAME: "Pull example complaints behind the trend",
                synthesis_agent.NAME: "Write the brief once the trend is established",
            },
        )
        .add_many(
            source_agent=synthesis_agent.NAME,
            target_agents={
                # Synthesis is normally a dead end by design — but the
                # Orchestrator can route straight to it on the very first move
                # (see NAME's edges above), and if that happens before any
                # specialist has actually gathered anything, Synthesis needs a
                # way out other than silently giving up via complete_task.
                NAME: "Hand back for real routing — no specialist findings "
                      "exist yet to synthesize",
            },
        )
    )


# --------------------------------------------------------------------------- #
# Handoff execution
# --------------------------------------------------------------------------- #

async def run_handoff(
    question: str,
    on_turn: Callable[[Turn], None] | None = None,
    timeout: float = 180.0,
) -> Brief:
    """Run the question through SK's handoff orchestration."""
    trace: list[Turn] = []

    def capture(message: ChatMessageContent) -> None:
        content = (message.content or "").strip()
        if not content:
            return  # tool-call-only turns carry no prose
        turn = Turn(agent=message.name or "agent", content=content)
        trace.append(turn)
        if on_turn:
            on_turn(turn)

    team = build_team()
    orchestration = HandoffOrchestration(
        members=list(team.values()),
        handoffs=build_handoffs(),
        agent_response_callback=capture,
    )

    runtime = InProcessRuntime()
    runtime.start()
    try:
        result = await orchestration.invoke(task=question, runtime=runtime)
        final = await result.get(timeout=timeout)
    finally:
        await runtime.stop_when_idle()

    answer = _best_answer(trace, final)
    error = None
    if not _is_real_brief(answer):
        # Handoff routing never actually reached Synthesis — surface that
        # honestly rather than showing SK's internal wrap-up sentinel as if it
        # were the analyst's brief. `auto` mode already retries/falls back on
        # this; a caller that asked for `handoff` explicitly gets the plain
        # explanation instead of a confusing raw message.
        error = "The Orchestrator ended the run without routing to a specialist."
        answer = (
            "No brief was produced — the handoff routing didn't reach the "
            "Synthesis agent. Try again, or use pipeline mode."
        )
    elif fake_ids := _fabricated_citations(answer):
        # Worse than a routing failure: this looks like a clean success. Never
        # show a fabricated citation as trustworthy — see _fabricated_citations.
        error = (
            f"The brief cited complaint ID(s) not found in the indexed data "
            f"(likely fabricated): {', '.join(fake_ids)}. Its claims are not verified."
        )
    return Brief(question=question, answer=answer, mode="handoff", trace=trace, error=error)


def _best_answer(trace: list[Turn], final) -> str:
    """Prefer the Synthesis agent's own words over the orchestration's wrap-up.

    SK ends a handoff run with `complete_task(task_summary=...)`, and whichever
    agent happens to call it writes that summary. The brief we want is the last
    thing SynthesisAgent actually said.
    """
    for turn in reversed(trace):
        if turn.agent == synthesis_agent.NAME:
            return turn.content
    if isinstance(final, list):
        final = final[-1] if final else None
    text = getattr(final, "content", None) or (str(final) if final else "")
    return text.strip() or (trace[-1].content if trace else "No response produced.")


# Semantic Kernel's own _complete_task hardcodes this exact prefix on every
# call, whatever the summary text says (handoffs.py:
# `content=f"Task is completed with summary: {task_summary}"`) — confirmed by
# reading the source, not guessed. A live run showed why checking for it
# generally beats matching one specific summary string: the framework's own
# "no handoff agent name provided" wrap-up is one possible summary, but
# Synthesis can also call complete_task itself with a confident-sounding
# summary of its own ("Successfully evaluated...") instead of ever writing the
# actual multi-section brief — same underlying problem (complete_task standing
# in for the real deliverable), different words. Matching the prefix catches
# every variant, past or future, instead of one observed wording.
_COMPLETE_TASK_PREFIX = "task is completed with summary:"


def _is_real_brief(text: str) -> bool:
    return bool(text) and len(text) > 80 and not text.lower().startswith(_COMPLETE_TASK_PREFIX)


# Complaint IDs are always cited as bracketed lists in this format, per
# Synthesis's own instructed shape: "[24747089, 24746002]". Matching that
# format (rather than any 6+ digit number anywhere) avoids false positives on
# percentages, dates, or dollar amounts elsewhere in the prose.
_CITATION_GROUP_RE = re.compile(r"\[([\d,\s]+)\]")


def _cited_complaint_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for group in _CITATION_GROUP_RE.finditer(text):
        for token in group.group(1).split(","):
            token = token.strip()
            if token.isdigit():
                ids.add(token)
    return ids


def _fabricated_citations(text: str) -> list[str]:
    """Cited complaint IDs that don't actually exist in the indexed corpus.

    A model under pressure to satisfy "every theme needs an ID" can invent a
    plausible-looking number instead of admitting it has none — confirmed live:
    a brief cited three IDs (7890123-7890125) that don't exist anywhere in the
    index, in the same response that correctly wrote "not assessed" for a
    different section. Prompt instructions alone don't reliably prevent this,
    so every brief's citations are checked against the real index — the one
    thing in this pipeline that's always ground truth — before it's ever shown
    as a clean success.
    """
    ids = _cited_complaint_ids(text)
    if not ids:
        return []
    from data.ingest import get_index

    real_ids = set(get_index().frame["complaint_id"].astype(str))
    return sorted(cid for cid in ids if cid not in real_ids)


# --------------------------------------------------------------------------- #
# Deterministic pipeline
# --------------------------------------------------------------------------- #

async def _ask(agent: ChatCompletionAgent, prompt: str) -> str:
    # Each pipeline call is independent, so a rate-limit 429 here is retried in
    # place rather than restarting the whole run — unlike a mid-handoff 429,
    # nothing upstream needs to be redone.
    response = await with_rate_limit_retry(lambda: agent.get_response(messages=prompt))
    return (response.message.content or "").strip()


_ISSUE_STOPWORDS = {
    "a", "an", "and", "or", "of", "the", "to", "your", "you", "in", "on",
    "for", "with", "including", "on", "about",
}


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower().replace("&", " and "))
    return {w for w in words if w not in _ISSUE_STOPWORDS and len(w) > 2}


def _issues_named_in(text: str, known_issues: list[str], limit: int = 4,
                     min_overlap: float = 0.6) -> list[str]:
    """Which real corpus issue-category strings are mentioned in `text`.

    Not an exact-substring check: Synthesis's own paraphrase of a Trend finding
    routinely drifts from the corpus's literal category string — the corpus
    says "Advertising and marketing, including promotional offers", a brief
    says "Advertising & marketing (promotional offers)". Matching is by
    significant-word overlap instead (stopwords and "&"/"and" normalised away),
    so a paraphrase that keeps the substantive words still matches.

    Longest-issue-first, and a match already covered by a longer match already
    picked is skipped — so "Advertising" doesn't win a separate slot from
    "Advertising and marketing, including promotional offers" when both hit.
    """
    text_words = _significant_words(text)
    scored = []
    for issue in known_issues:
        issue_words = _significant_words(issue)
        if not issue_words:
            continue
        overlap = len(issue_words & text_words) / len(issue_words)
        if overlap >= min_overlap:
            scored.append((overlap, len(issue), issue))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)  # best overlap, then most specific

    picked: list[str] = []
    for _, _, issue in scored:
        if any(issue.lower() in kept.lower() or kept.lower() in issue.lower() for kept in picked):
            continue
        picked.append(issue)
        if len(picked) >= limit:
            break
    return picked


def _targeted_issue_evidence(question: str, trend_text: str, per_issue: int = 2) -> str:
    """Look up real complaints for whatever issue categories Trend just named.

    Retrieval runs *before* Trend in the pipeline, seeded only by the analyst's
    raw question — it has no way to know which categories Trend's own
    aggregation will later flag as fastest-growing, so its semantic search can
    easily surface a different set of complaints than the ones Trend is
    actually talking about (Synthesis then has numbers with nothing to cite).
    This closes that gap without another LLM call: matching an issue name is a
    deterministic corpus lookup, not a judgment call, so it's cheap to just do
    directly against the index.
    """
    from data.ingest import get_index  # local import matches this module's existing pattern

    index = get_index()
    known_issues = index.frame["issue"].dropna().unique().tolist()
    matched = _issues_named_in(trend_text, known_issues)
    if not matched:
        return ""

    blocks = []
    for issue in matched:
        hits = index.search(query=question, k=per_issue, issue=issue)
        if hits:
            blocks.append(f'For "{issue}":\n{format_hits(hits)}')
    if not blocks:
        return ""

    return (
        "\n\n=== Additional complaints for the Trend agent's named issues "
        "(looked up directly against the index, not reviewed by Retrieval) ===\n\n"
        + "\n\n".join(blocks)
    )


async def run_pipeline(
    question: str,
    on_turn: Callable[[Turn], None] | None = None,
) -> Brief:
    """Retrieval → risk → trend → synthesis, with no routing model in the loop."""
    trace: list[Turn] = []

    def record(agent: str, content: str) -> None:
        turn = Turn(agent=agent, content=content)
        trace.append(turn)
        if on_turn:
            on_turn(turn)

    team = build_team()

    retrieval = await _ask(
        team[retrieval_agent.NAME],
        f"Analyst question: {question}\n\n"
        "Search the complaint index and report the most relevant complaints, "
        "keeping every complaint ID.",
    )
    record(retrieval_agent.NAME, retrieval)

    risk, trend = await asyncio.gather(
        _ask(
            team[risk_agent.NAME],
            f"Analyst question: {question}\n\n"
            f"The Retrieval agent found these complaints:\n\n{retrieval}\n\n"
            "Score them for linguistic risk and give the overall severity read.",
        ),
        _ask(
            team[trend_agent.NAME],
            f"Analyst question: {question}\n\n"
            f"Context from retrieval (for the company/product in play):\n\n{retrieval[:1500]}\n\n"
            "Report the volume trend for this slice, quarter over quarter.",
        ),
    )
    record(risk_agent.NAME, risk)
    record(trend_agent.NAME, trend)

    # Retrieval ran before Trend existed, so it had no way to target whatever
    # categories Trend goes on to name — see _targeted_issue_evidence for why
    # this closes that gap with a direct index lookup rather than another
    # LLM call.
    evidence_addendum = _targeted_issue_evidence(question, trend)
    if evidence_addendum:
        record("EvidenceLookup", evidence_addendum.strip())

    brief = await _ask(
        team[synthesis_agent.NAME],
        f"Analyst question: {question}\n\n"
        f"=== RetrievalAgent ===\n{retrieval}\n\n"
        f"=== LinguisticRiskAgent ===\n{risk}\n\n"
        f"=== TrendAgent ===\n{trend}{evidence_addendum}\n\n"
        "Write the brief.",
    )
    record(synthesis_agent.NAME, brief)
    error = None
    if fake_ids := _fabricated_citations(brief):
        # Same guard as run_handoff — pipeline mode builds Synthesis a clean
        # digest of real tool output, so this should be rarer here, but
        # Synthesis is still a free-text model call and can still fabricate.
        error = (
            f"The brief cited complaint ID(s) not found in the indexed data "
            f"(likely fabricated): {', '.join(fake_ids)}. Its claims are not verified."
        )
    return Brief(question=question, answer=brief, mode="pipeline", trace=trace, error=error)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

async def answer(
    question: str,
    mode: str = "auto",
    on_turn: Callable[[Turn], None] | None = None,
) -> Brief:
    """Answer a question. `mode` is 'auto', 'handoff' or 'pipeline'."""
    if not llm_available():
        raise MissingApiKey(
            "GROQ_API_KEY is not set — the agent team cannot run. The Explore tab "
            "still works without it, since search and risk scoring are local."
        )

    if mode == "pipeline":
        return await run_pipeline(question, on_turn)
    if mode == "handoff":
        return await run_handoff(question, on_turn)

    # A transient failure mid-handoff (a rate limit, or gpt-oss occasionally
    # mis-forming a tool call) can't be resumed from where it broke — the whole
    # run restarts. That's wasteful of the same tight token budget that likely
    # caused it, so only one restart is attempted before giving up to pipeline.
    note = ""
    for attempt in range(2):
        try:
            brief = await run_handoff(question, on_turn)
        except QuotaExhausted:
            # Pipeline hits the same model on the same key — falling back here
            # would just fail again in a different call, less informatively.
            raise
        except Exception as exc:  # noqa: BLE001 - any other handoff failure should degrade, not crash
            note = f"handoff failed: {type(exc).__name__}: {exc}"
            if is_transient_error(exc) and attempt == 0:
                continue
            break
        if brief.error is None:
            return brief
        note = brief.error
        break

    fallback = await run_pipeline(question, on_turn)
    fallback.mode = "pipeline (fallback)"
    # Preserve both: why handoff was abandoned, and any new problem pipeline
    # itself introduces (e.g. its own fabricated citation) — overwriting
    # fallback.error with just `note` would silently discard the latter.
    fallback.error = f"{note} | pipeline also: {fallback.error}" if fallback.error else note
    return fallback


def answer_sync(question: str, mode: str = "auto") -> Brief:
    return asyncio.run(answer(question, mode))


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or (
        "What are Wells Fargo customers most frustrated about with overdraft fees "
        "in the last two quarters, and how urgent does the language sound?"
    )
    result = answer_sync(question)
    print(f"\n=== mode: {result.mode} ===")
    if result.error:
        print(f"(note: {result.error})")
    print(f"\n{result.answer}\n")
