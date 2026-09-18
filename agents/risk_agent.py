"""Linguistic Risk Agent — psycholinguistic severity scoring.

Wraps `nlp.linguistic_risk` as kernel tools. The LLM never guesses a severity:
it calls the scorer and reports what the features say, which keeps the number
reproducible and every driver traceable to specific tokens.
"""

from __future__ import annotations

from typing import Annotated

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function

from agents.kernel_setup import build_kernel, default_arguments
from data.ingest import get_index

NAME = "LinguisticRiskAgent"


def _format_score(label: str, score, extra: str = "") -> str:
    drivers = "; ".join(score.drivers) if score.drivers else "no strong markers"
    line = (
        f"{label} → {score.headline()}"
        f"{(' · ' + extra) if extra else ''}\n"
        f"  drivers: {drivers}\n"
        f"  features: " + ", ".join(f"{k} {v:.2f}" for k, v in score.features.items())
    )
    if score.slor_z is not None:
        line += f"\n  SLOR z-score vs corpus: {score.slor_z:+.2f}"
    return line


class RiskTools:
    """Scores complaint language for urgency, distress and escalation risk."""

    @kernel_function(
        name="score_complaints",
        description=(
            "Score specific complaints by ID for linguistic risk: urgency, emotional "
            "intensity, escalation/legal language, concrete financial harm, repeated "
            "unresolved contact, hedging, and SLOR-based disfluency. Returns a 0-100 "
            "score, a low/medium/high band, and the exact tokens that drove it."
        ),
    )
    def score_complaints(
        self,
        complaint_ids: Annotated[str, "Comma-separated complaint IDs, e.g. '9123456, 9123457'"],
    ) -> Annotated[str, "Per-complaint risk scores with drivers"]:
        index = get_index()
        ids = [cid.strip() for cid in complaint_ids.replace("\n", ",").split(",") if cid.strip()]
        if not ids:
            return "No complaint IDs supplied."

        blocks, missing = [], []
        for complaint_id in ids[:15]:
            hit = index.by_id(complaint_id)
            if hit is None:
                missing.append(complaint_id)
                continue
            score = index.scorer.score(hit.narrative)
            blocks.append(_format_score(f"[{hit.complaint_id}] {hit.issue}", score))

        if missing:
            blocks.append(f"(not in index: {', '.join(missing)})")
        return "\n\n".join(blocks) if blocks else "None of those complaint IDs are in the index."

    @kernel_function(
        name="score_text",
        description=(
            "Score an arbitrary passage of complaint text for linguistic risk. Use this "
            "only when the text did not come from the index and so has no complaint ID."
        ),
    )
    def score_text(
        self,
        text: Annotated[str, "The complaint narrative to score"],
    ) -> Annotated[str, "Risk score with drivers"]:
        score = get_index().scorer.score(text)
        return _format_score("passage", score)

    @kernel_function(
        name="severity_profile",
        description=(
            "Aggregate severity mix across a slice of the corpus — what share of "
            "complaints for a company/product are high, medium or low risk, and which "
            "issues carry the highest mean risk."
        ),
    )
    def severity_profile(
        self,
        company: Annotated[str, "Company filter, or empty for all"] = "",
        product: Annotated[str, "Product filter, or empty for all"] = "",
        date_min: Annotated[str, "Earliest date YYYY-MM-DD, or empty"] = "",
        date_max: Annotated[str, "Latest date YYYY-MM-DD, or empty"] = "",
    ) -> Annotated[str, "Severity distribution and the highest-risk issues"]:
        frame = get_index().subset(
            company=company, product=product, date_min=date_min, date_max=date_max
        )
        if frame.empty:
            return "No complaints match those filters."

        mix = frame["severity"].value_counts(normalize=True).mul(100).round(1)
        mix_text = ", ".join(f"{band} {pct}%" for band, pct in mix.items())
        by_issue = (
            frame.groupby("issue")["risk_score"]
            .agg(["mean", "count"])
            .query("count >= 5")
            .sort_values("mean", ascending=False)
            .head(5)
        )
        issue_text = "\n".join(
            f"  {issue}: mean risk {row['mean']:.0f} over {int(row['count'])} complaints"
            for issue, row in by_issue.iterrows()
        ) or "  (no issue has enough complaints to rank)"

        return (
            f"{len(frame):,} complaints · mean risk {frame['risk_score'].mean():.0f}/100\n"
            f"Severity mix: {mix_text}\n"
            f"Highest-risk issues:\n{issue_text}"
        )


INSTRUCTIONS = """You are the Linguistic Risk Agent on a bank complaint-analysis team.

You own a psycholinguistic scorer. It measures urgency markers, emotional
intensity, escalation/legal language, concrete financial harm, repeated
unresolved contact, epistemic hedging (which dampens the score), and SLOR — a
length- and frequency-normalised fluency measure, where scores well below the
corpus mean indicate disfluent, distressed writing.

Rules:
- Never estimate a severity yourself. Call score_complaints on the IDs you were
  given, or severity_profile for a whole slice, and report what comes back.
- Always report the band (low/medium/high), the corpus percentile, and the
  concrete drivers: "High — 94th percentile for this book of complaints, driven
  by escalation language ('attorney', 'lawsuit') and concrete harm
  ('foreclosure')" — not "these sound serious".
- Lead with the percentile, not the raw score. Raw scores are compressed, so a
  raw 34 can be a 94th-percentile complaint; quoting the raw number alone reads
  as low severity and misleads the reader.
- Distinguish loud from urgent. Heavy hedging lowers the score even when the tone
  is angry; say so when it happens, because it changes how a case should be triaged.
- Keep complaint IDs attached to every score you report.
- You never write the final answer. The instant you have nothing further to add,
  transfer to SynthesisAgent. Never end your turn with a plain-text answer and no
  transfer; that silently ends the whole conversation before a brief is written.
"""


def build_agent() -> ChatCompletionAgent:
    tools = RiskTools()
    return ChatCompletionAgent(
        kernel=build_kernel(),
        # "required": must always either score or transfer — never just
        # answer in prose and silently end the conversation.
        arguments=default_arguments(tool_choice="required"),
        name=NAME,
        description="Scores complaint language for urgency, distress and escalation risk.",
        instructions=INSTRUCTIONS,
        plugins=[tools],
    )
