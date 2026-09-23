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

from semantic_kernel.agents import Agent
from semantic_kernel.agents.orchestration.handoffs import (
    HandoffOrchestration,
    OrchestrationHandoffs,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent, FunctionResultContent

from agents import retrieval_agent, risk_agent, synthesis_agent, trend_agent
from agents.retrieval_agent import format_hits
from agents.kernel_setup import (
    MissingApiKey,
    QuotaExhausted,
    build_agent,
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

Routing rules — check every clause of the question against all three,
independently, before routing anywhere:
- A question about what customers are saying, or about themes → RetrievalAgent.
- A question mentioning urgency, severity, frustration, tone, escalation or
  "how bad is it" → RetrievalAgent first, then LinguisticRiskAgent.
- A question about volume, growth, spikes, "rising", "more than last quarter"
  → TrendAgent.
- Most real questions need two or three of them — a question can ask about
  more than one of these at once, and a strong cue for one ("growing
  fastest") does not mean the others don't also apply. Confirmed live: "Which
  issues are growing fastest, and how severe is the language?" was routed to
  TrendAgent only, on the strength of "growing fastest", and Synthesis then
  wrote a Severity section from no real scoring at all, since
  LinguisticRiskAgent (and the RetrievalAgent it depends on) were never
  called. Treat each clause of a multi-part question as its own routing
  decision — "how severe" always means RetrievalAgent + LinguisticRiskAgent
  regardless of what else the question also asks about.
- When severity/urgency was asked about, RetrievalAgent must be followed by
  LinguisticRiskAgent specifically, before TrendAgent or Synthesis — not just
  "eventually". Confirmed live, even after the fix above got RetrievalAgent
  invoked, it transferred straight to TrendAgent on its own and
  LinguisticRiskAgent still never ran. LinguisticRiskAgent is the only one of
  the three with a hard dependency on RetrievalAgent's own output (it needs
  the complaint IDs Retrieval just found), so once Retrieval has real IDs in
  hand and severity is part of the question, LinguisticRiskAgent is next,
  every time — TrendAgent has no such dependency and can run whenever, so it
  never needs to jump the queue ahead of the one specialist that does.
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

def build_orchestrator_agent() -> Agent:
    return build_agent(
        name=NAME,
        description="Routes analyst questions to the right specialist agents.",
        instructions=INSTRUCTIONS,
        # "required": the Orchestrator's only job is routing — it must never
        # just answer in prose instead of calling a transfer_to_* function.
        tool_choice="required",
    )


def build_team(specialist_tool_choice: str = "required") -> dict[str, Agent]:
    """The five agents. Order matters: the first member receives the question.

    `specialist_tool_choice` only affects Retrieval/Risk/Trend (never the
    Orchestrator, which pipeline mode doesn't use anyway, or Synthesis, which
    is always "auto"). Handoff mode needs "required" here — without it, a
    specialist can answer in prose and silently end the whole conversation
    before ever calling a tool or handing off. Pipeline mode's `_ask()` calls
    each specialist exactly once for one directed task with no handoff to
    protect, so it passes "auto" instead: confirmed live, forcing tool_choice
    on that single-shot call can leave the agent no way to ever answer in
    text at all, since the round Gemini's connector falls back to when it
    gives up forcing still reuses the same forced tool list — this sidesteps
    that bug by simply not forcing where forcing was never needed.
    """
    return {
        NAME: build_orchestrator_agent(),
        retrieval_agent.NAME: retrieval_agent.build_agent(specialist_tool_choice),
        risk_agent.NAME: risk_agent.build_agent(specialist_tool_choice),
        trend_agent.NAME: trend_agent.build_agent(specialist_tool_choice),
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
    # Raw tool results, captured independently of whether an agent also wrote
    # prose about them — pipeline mode's citation-matching pool comes from
    # RetrievalAgent restating its findings as its own text response, but a
    # handoff-mode agent can call a tool and transfer immediately with no
    # accompanying prose, so `trace` alone (prose only, see `capture` below)
    # isn't a reliable source of real complaint IDs to match against. This is
    # actually more reliable than pipeline mode's approach where it applies:
    # it's the tool's own verbatim output, never reworded by the model.
    retrieved_hit_texts: list[str] = []
    _HIT_PRODUCING_FUNCTIONS = {"search_complaints", "score_complaints"}

    def capture(message: ChatMessageContent) -> None:
        for item in message.items:
            if isinstance(item, FunctionResultContent) and item.function_name in _HIT_PRODUCING_FUNCTIONS:
                retrieved_hit_texts.append(str(item.result))
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
    else:
        # Same gap _targeted_issue_evidence already closes for pipeline mode,
        # now applied here too: RetrievalAgent's own semantic search is one
        # generic query and can easily miss the specific issue categories
        # Risk/Trend's own tool output highlighted as significant — confirmed
        # live, a brief's Themes ended up entirely "not confirmed" even though
        # real evidence for those categories existed in the index, because
        # the one search Retrieval happened to run was about a different
        # sub-topic. Pipeline mode has direct access to Trend's own text for
        # this; handoff mode's equivalent is the specialists' prose already
        # captured in `trace`.
        specialist_text = " ".join(
            turn.content for turn in trace if turn.agent in (trend_agent.NAME, risk_agent.NAME)
        )
        evidence_addendum = _targeted_issue_evidence(question, specialist_text)

        # Same citation-attachment step pipeline mode uses (see its own
        # comment on why Synthesis no longer writes IDs itself) — the hits
        # pool here comes from actual tool results captured during the run
        # (see `capture` above) plus the targeted lookup above, not from
        # re-asking any agent to restate itself. Still followed by the
        # redaction safety net: attachment only ever inserts IDs it found in
        # real tool output, but Synthesis is free-text and can still write a
        # stray bracketed number despite being told not to.
        hits = _parse_retrieved_hits(*retrieved_hit_texts, evidence_addendum)
        answer = _attach_citations(answer, hits)
        # Never let a fabricated ID number reach the screen — see
        # _redact_fabricated_citations. `error` still reflects the fabrication
        # so `auto` mode's fallback logic treats this as a failure worth
        # retrying via pipeline, even though the displayed text is now safe.
        answer, fake_ids = _redact_fabricated_citations(answer)
        if fake_ids:
            error = (
                f"The brief originally cited complaint ID(s) not found in the "
                f"indexed data (likely fabricated): {', '.join(fake_ids)}. "
                "Those citations have been redacted below — treat the affected "
                "theme(s) as unconfirmed, not just missing a number."
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


def _real_complaint_ids() -> set[str]:
    from data.ingest import get_index

    return set(get_index().frame["complaint_id"].astype(str))


def _fabricated_citations(text: str) -> list[str]:
    """Cited complaint IDs that don't actually exist in the indexed corpus.

    A model under pressure to satisfy "every theme needs an ID" can invent a
    plausible-looking number instead of admitting it has none — confirmed live,
    twice, with different fake IDs both times, including after Synthesis's own
    prompt was tightened to explicitly forbid it — this is not something
    prompt wording alone reliably prevents. Every brief's citations are
    checked against the real index — the one thing in this pipeline that's
    always ground truth — before it's ever shown as a clean success.
    """
    ids = _cited_complaint_ids(text)
    if not ids:
        return []
    real_ids = _real_complaint_ids()
    return sorted(cid for cid in ids if cid not in real_ids)


def _redact_fabricated_citations(text: str) -> tuple[str, list[str]]:
    """Strip any cited complaint ID not found in the real index from `text`,
    replacing it with an explicit marker instead of silently keeping a
    plausible-looking fake number in front of the reader.

    This is a display-safety net, not a groundedness check: it guarantees no
    fabricated *ID number* ever reaches the screen, but it cannot verify that
    the theme's surrounding description is itself accurate — that would need
    checking the claim's substance against the retrieved narratives, a harder
    problem this doesn't attempt. Treat a redacted theme as unconfirmed
    overall, not just missing one number.
    """
    fake_ids = set(_fabricated_citations(text))
    if not fake_ids:
        return text, []

    def _replace(match: re.Match) -> str:
        tokens = [t.strip() for t in match.group(1).split(",")]
        kept = [t for t in tokens if t not in fake_ids]
        if len(kept) == len(tokens):
            return match.group(0)  # nothing fake in this group — leave as-is
        return "[" + ", ".join(kept) + ", unverified]" if kept else "[unverified — no confirmed complaint ID]"

    return _CITATION_GROUP_RE.sub(_replace, text), sorted(fake_ids)


# --------------------------------------------------------------------------- #
# Deterministic pipeline
# --------------------------------------------------------------------------- #

# Every specialist's own INSTRUCTIONS (shared with handoff mode, where
# they're accurate) tell it to call transfer_to_X or complete_task once
# it's done — functions HandoffOrchestration adds dynamically, which
# pipeline mode's direct get_response() call never does. Confirmed live: a
# model trying to follow that instruction with no such function actually
# available just types the call out as literal text instead of answering
# ("transfer_to_Orchestrator()"). Prepending this per-call note is far
# cheaper than forking every agent's core instructions by mode.
_PIPELINE_MODE_NOTE = (
    "Note: this is a single direct request in a pipeline with no routing "
    "model — there is no transfer_to_X or complete_task function available "
    "here, and nothing else runs after your reply. Ignore any instruction "
    "about transferring to another agent or calling complete_task; once "
    "you've done the task below, just write your findings as your answer.\n\n"
)


async def _ask(agent: Agent, prompt: str) -> str:
    # Each pipeline call is independent, so a rate-limit 429 here is retried in
    # place rather than restarting the whole run — unlike a mid-handoff 429,
    # nothing upstream needs to be redone.
    response = await with_rate_limit_retry(
        lambda: agent.get_response(messages=_PIPELINE_MODE_NOTE + prompt)
    )
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


# CFPB complaint IDs are consistently 7-8 digits in this corpus, and nothing
# else in this domain produces a bare run that long — dates split into
# 4/2/2-digit groups by hyphens, dollar amounts and percentiles are far
# shorter. Anchoring on digit-run length alone, not surrounding punctuation,
# is deliberate: confirmed live across three separate runs, RetrievalAgent's
# own instructions ask it to report IDs verbatim as "[id]", and it still
# reformats freely from run to run — "[id]" one time, "Complaint ID: id" with
# no brackets at all the next. A negative lookbehind excludes a "$"-prefixed
# run so an unusually large dollar figure can't be mistaken for an ID.
_HIT_ID_RE = re.compile(r"(?<!\$)\b(\d{7,8})\b")


def _parse_retrieved_hits(*texts: str) -> list[tuple[str, str]]:
    """Extract (complaint_id, surrounding_text) pairs from retrieval/evidence text.

    Matches any ID-shaped digit run anywhere in the text, then takes the span
    up to the next one as that ID's context — works regardless of whatever
    formatting the model chose around it, since the anchor is the digits
    themselves, not brackets or a label.
    """
    hits: list[tuple[str, str]] = []
    for text in texts:
        if not text:
            continue
        matches = list(_HIT_ID_RE.finditer(text))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            hits.append((match.group(1), text[match.start():end]))
    return hits


# `_significant_words` is alphabetic-only ("[a-z]+"), so a dollar amount like
# "$1,300.00" contributes nothing to it — yet a specific figure is often the
# one detail that actually distinguishes two similarly-worded themes ("$1300
# in fees" vs. "$35 on a single transaction"). Normalizing away commas and
# cents (both "$1,300.00" and "$1300" become "1300") lets a theme and its
# matching complaint agree on a figure even if one writes it with commas/cents
# and the other doesn't. A 2+ digit floor skips stray single digits (list
# markers, a lone "1") that would otherwise match almost anything.
_NUMERIC_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numeric_tokens(text: str) -> set[str]:
    tokens = set()
    for match in _NUMERIC_TOKEN_RE.finditer(text):
        cleaned = match.group(0).replace(",", "").split(".")[0]
        if len(cleaned) >= 2:
            tokens.add(cleaned)
    return tokens


def _best_matching_ids(theme_text: str, hits: list[tuple[str, str]],
                        limit: int = 3, min_shared_words: int = 2) -> list[str]:
    """Which retrieved complaints' text most overlaps this theme's own wording.

    Deterministic keyword overlap, not another model call — the same pattern
    `_issues_named_in` already uses for matching Trend's issue names. A
    minimum shared-word count (not just "any overlap") avoids attaching an ID
    on the strength of one generic word ("account", "fee") shared with
    everything in the corpus.
    """
    theme_words = _significant_words(theme_text) | _numeric_tokens(theme_text)
    if not theme_words:
        return []
    scored = []
    for complaint_id, block in hits:
        block_words = _significant_words(block) | _numeric_tokens(block)
        shared = len(theme_words & block_words)
        if shared >= min_shared_words:
            scored.append((shared, complaint_id))
    scored.sort(key=lambda t: t[0], reverse=True)

    picked: list[str] = []
    seen: set[str] = set()
    for _, complaint_id in scored:
        if complaint_id in seen:
            continue
        seen.add(complaint_id)
        picked.append(complaint_id)
        if len(picked) >= limit:
            break
    return picked


# Matches from Synthesis's own "**Themes** —" heading up to the next "**Xxx**"
# heading (or end of text), so only Themes content gets citations attached —
# Answer/Severity/Trend/What I'd check next are left untouched. The header
# group deliberately stops at the heading markup itself (not ".*?" up to the
# first newline) — confirmed live, when Synthesis writes Themes as one
# paragraph with no newline until the section ends, a lazy ".*?(?:\n|$)"
# header swallows the *entire paragraph* into the header group, leaving
# nothing in the body for the paragraph-fallback path to match against.
# The dash-matching stays on the heading's own line ("[ \t]*", never "\s*",
# before it) and requires 2+ hyphens or a real em-dash for a plain "-" —
# confirmed live, an earlier version using "\s*[—-]*\s*" let the leading
# `\s*` cross the newline after "**Themes**" and then let "[—-]*" consume the
# *first bullet's own leading hyphen* as if it were heading punctuation,
# silently stripping that one bullet of its "- " marker so the bullet regex
# below could no longer see it as a bullet at all.
_THEMES_SECTION_RE = re.compile(r"(\*\*Themes\*\*[ \t]*(?:—+|-{2,})?[ \t]*\n?)(.*?)(?=\n\*\*\w|\Z)", re.DOTALL)
# No trailing `\s*` — under MULTILINE, a greedy `\s*` before `$` can cross
# into and consume a following blank line (`\s` matches `\n` too), which ate
# the blank line before the next section's heading in testing.
_BULLET_LINE_RE = re.compile(r"^-\s+\S.*$", re.MULTILINE)


def _attach_citations(brief_text: str, hits: list[tuple[str, str]]) -> str:
    """Insert real complaint-ID citations into Synthesis's Themes bullets.

    Synthesis is no longer asked to write IDs at all (see synthesis_agent.py's
    module docstring for why) — this is the step that actually adds them,
    matching each bullet's own wording against what Retrieval genuinely
    returned. A bullet with no confident match is marked as such rather than
    left silently uncited or backed by a guess — deliberately including the
    case where `hits` is empty (no tool that returns complaint IDs was even
    called this run, e.g. a pure-trend question routed only to TrendAgent):
    confirmed live, an earlier version of this function returned the brief
    completely unchanged whenever hits was empty, which meant a theme with
    zero retrieved evidence behind it looked identical to a properly-cited
    one. Every theme is always marked one way or the other, never left silent.
    """

    def _bullet_replacer(match: re.Match) -> str:
        line = match.group(0)
        ids = _best_matching_ids(line, hits)
        if ids:
            return f"{line} [{', '.join(ids)}]"
        return f"{line} [no closely-matching complaint found]"

    def _themes_replacer(match: re.Match) -> str:
        header, body = match.group(1), match.group(2)
        if not _BULLET_LINE_RE.search(body):
            # Synthesis wrote Themes as a paragraph instead of the instructed
            # bulleted list — confirmed live, this happens. Rather than
            # silently attach nothing, treat the whole body as one block so
            # at least some real evidence still gets cited.
            stripped = body.strip()
            if not stripped:
                return match.group(0)
            ids = _best_matching_ids(stripped, hits)
            suffix = f" [{', '.join(ids)}]" if ids else " [no closely-matching complaint found]"
            return header + body.rstrip() + suffix + "\n"
        return header + _BULLET_LINE_RE.sub(_bullet_replacer, body)

    return _THEMES_SECTION_RE.sub(_themes_replacer, brief_text, count=1)


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

    team = build_team(specialist_tool_choice="auto")

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
    # Synthesis no longer writes complaint-ID citations itself (see
    # synthesis_agent.py's module docstring) — this is the step that actually
    # attaches them, by matching each Themes bullet's own wording against what
    # Retrieval and the targeted issue lookup genuinely returned. It never has
    # the opportunity to invent an ID, because it's never asked to produce one.
    hits = _parse_retrieved_hits(retrieval, evidence_addendum)
    brief = _attach_citations(brief, hits)
    # Kept as a safety net, not the primary mechanism: catches the rare case
    # where Synthesis writes a bracketed number itself despite being told not
    # to, rather than one this step attached.
    brief, fake_ids = _redact_fabricated_citations(brief)
    record(synthesis_agent.NAME, brief)
    error = (
        f"The brief originally cited complaint ID(s) not found in the indexed "
        f"data (likely fabricated): {', '.join(fake_ids)}. Those citations "
        "have been redacted below — treat the affected theme(s) as "
        "unconfirmed, not just missing a number."
    ) if fake_ids else None
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
            "GOOGLE_AI_API_KEY is not set — the agent team cannot run. The Explore "
            "tab still works without it, since search and risk scoring are local."
        )

    if mode == "pipeline":
        return await run_pipeline(question, on_turn)
    if mode == "handoff":
        return await run_handoff(question, on_turn)

    # A transient failure mid-handoff (a rate limit, or the model occasionally
    # mis-forming a tool call) can't be resumed from where it broke — the whole
    # run restarts. That's wasteful of the same budget that likely caused it,
    # so only one restart is attempted before giving up to pipeline.
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
