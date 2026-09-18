"""Trend Agent — pandas aggregations over the complaint corpus.

Volume, pace, issue mix and quarter-over-quarter movement. Deliberately
arithmetic: the LLM's job here is to read the table out loud, not to estimate.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function

from agents.kernel_setup import build_kernel, default_arguments
from data.ingest import get_index

NAME = "TrendAgent"


def _pct_change(new: float, old: float) -> str:
    if old == 0:
        return "n/a (no complaints in the prior period)"
    return f"{(new - old) / old * 100:+.0f}%"


class TrendTools:
    """Aggregate statistics over complaint volume, pace and issue mix."""

    @kernel_function(
        name="complaint_volume",
        description=(
            "Complaint counts over time for a company/product slice, grouped by month "
            "or quarter, with the mean linguistic-risk score per period."
        ),
    )
    def complaint_volume(
        self,
        company: Annotated[str, "Company filter, or empty for all"] = "",
        product: Annotated[str, "Product filter, or empty for all"] = "",
        group_by: Annotated[str, "'month' or 'quarter'"] = "month",
    ) -> Annotated[str, "Volume per period with mean risk"]:
        frame = get_index().subset(company=company, product=product)
        if frame.empty:
            return "No complaints match those filters."

        column = "quarter" if group_by.lower().startswith("q") else "month"
        grouped = (
            frame.groupby(column)
            .agg(complaints=("complaint_id", "count"), mean_risk=("risk_score", "mean"))
            .sort_index()
        )
        lines = [
            f"  {period}: {int(row.complaints):>4} complaints · mean risk {row.mean_risk:.0f}"
            for period, row in grouped.iterrows()
        ]
        scope = f"{company or 'all companies'} / {product or 'all products'}"
        return f"Complaint volume by {column} — {scope} ({len(frame):,} total):\n" + "\n".join(lines)

    @kernel_function(
        name="quarter_over_quarter",
        description=(
            "Compare the two most recent quarters for a company/product slice: volume "
            "change, mean-risk change, and whether the latest quarter is still in progress."
        ),
    )
    def quarter_over_quarter(
        self,
        company: Annotated[str, "Company filter, or empty for all"] = "",
        product: Annotated[str, "Product filter, or empty for all"] = "",
    ) -> Annotated[str, "Quarter-over-quarter comparison"]:
        frame = get_index().subset(company=company, product=product)
        if frame.empty:
            return "No complaints match those filters."

        quarters = sorted(frame["quarter"].unique())
        if len(quarters) < 2:
            return f"Only one quarter ({quarters[0]}) is in scope — no comparison possible."

        previous_q, latest_q = quarters[-2], quarters[-1]
        latest = frame[frame["quarter"] == latest_q]
        previous = frame[frame["quarter"] == previous_q]

        # The newest quarter is usually partial, which makes a raw count comparison
        # look like a decline. Compare daily pace as well and say which is which.
        period = pd.Period(latest_q, freq="Q")
        observed_days = max((latest["date_received"].max() - period.start_time).days + 1, 1)
        full_days = (period.end_time - period.start_time).days + 1
        partial = observed_days < full_days

        latest_pace = len(latest) / observed_days
        previous_pace = len(previous) / ((pd.Period(previous_q, freq="Q").end_time
                                          - pd.Period(previous_q, freq="Q").start_time).days + 1)

        scope = f"{company or 'all companies'} / {product or 'all products'}"
        lines = [
            f"Quarter-over-quarter — {scope}",
            f"  {previous_q}: {len(previous):,} complaints · mean risk {previous['risk_score'].mean():.0f}",
            f"  {latest_q}: {len(latest):,} complaints · mean risk {latest['risk_score'].mean():.0f}",
            f"  Volume change: {_pct_change(len(latest), len(previous))}",
            f"  Mean-risk change: {latest['risk_score'].mean() - previous['risk_score'].mean():+.1f} points",
            f"  Daily pace: {previous_pace:.2f} → {latest_pace:.2f} complaints/day "
            f"({_pct_change(latest_pace, previous_pace)})",
        ]
        if partial:
            lines.append(
                f"  NOTE: {latest_q} is incomplete ({observed_days} of {full_days} days "
                f"observed), so the raw volume change understates it — use the daily pace."
            )
        return "\n".join(lines)

    @kernel_function(
        name="top_issues",
        description=(
            "The most common complaint issues in a company/product slice, with each "
            "issue's share of volume and mean linguistic-risk score."
        ),
    )
    def top_issues(
        self,
        company: Annotated[str, "Company filter, or empty for all"] = "",
        product: Annotated[str, "Product filter, or empty for all"] = "",
        date_min: Annotated[str, "Earliest date YYYY-MM-DD, or empty"] = "",
        date_max: Annotated[str, "Latest date YYYY-MM-DD, or empty"] = "",
        limit: Annotated[int, "How many issues to return, 1-15"] = 6,
    ) -> Annotated[str, "Ranked issues with share and mean risk"]:
        frame = get_index().subset(
            company=company, product=product, date_min=date_min, date_max=date_max
        )
        if frame.empty:
            return "No complaints match those filters."

        grouped = (
            frame.groupby("issue")
            .agg(n=("complaint_id", "count"), mean_risk=("risk_score", "mean"))
            .sort_values("n", ascending=False)
            .head(max(1, min(int(limit), 15)))
        )
        lines = [
            f"  {issue}: {int(row.n)} ({row.n / len(frame) * 100:.0f}%) · mean risk {row.mean_risk:.0f}"
            for issue, row in grouped.iterrows()
        ]
        scope = f"{company or 'all companies'} / {product or 'all products'}"
        return f"Top issues — {scope} ({len(frame):,} complaints):\n" + "\n".join(lines)

    @kernel_function(
        name="rising_issues",
        description=(
            "Issues whose share of complaints grew most between the two most recent "
            "quarters — the early-warning view."
        ),
    )
    def rising_issues(
        self,
        company: Annotated[str, "Company filter, or empty for all"] = "",
        product: Annotated[str, "Product filter, or empty for all"] = "",
        limit: Annotated[int, "How many issues to return, 1-10"] = 5,
    ) -> Annotated[str, "Issues gaining share quarter over quarter"]:
        frame = get_index().subset(company=company, product=product)
        quarters = sorted(frame["quarter"].unique()) if not frame.empty else []
        if len(quarters) < 2:
            return "Not enough quarters in scope to measure movement."

        previous_q, latest_q = quarters[-2], quarters[-1]
        latest = frame[frame["quarter"] == latest_q]["issue"].value_counts(normalize=True)
        previous = frame[frame["quarter"] == previous_q]["issue"].value_counts(normalize=True)
        counts = frame[frame["quarter"] == latest_q]["issue"].value_counts()

        movement = (latest.sub(previous, fill_value=0) * 100).sort_values(ascending=False)
        # Ignore issues too rare for a share change to mean anything.
        movement = movement[[i for i in movement.index if counts.get(i, 0) >= 3]]
        if movement.empty:
            return "No issue has enough volume for a meaningful share change."

        lines = [
            f"  {issue}: {previous.get(issue, 0) * 100:.0f}% → {latest.get(issue, 0) * 100:.0f}% "
            f"of quarterly volume ({delta:+.1f} pts, {int(counts.get(issue, 0))} complaints)"
            for issue, delta in movement.head(max(1, min(int(limit), 10))).items()
        ]
        return f"Share movement {previous_q} → {latest_q}:\n" + "\n".join(lines)


INSTRUCTIONS = """You are the Trend Agent on a bank complaint-analysis team.

You own the aggregate view: how many complaints, about what, moving which way.
Your tools compute real counts from the indexed corpus.

Rules:
- Never estimate a number. Call a tool and quote what it returns.
- Prefer daily pace over raw counts when a tool flags the newest quarter as
  incomplete — a partial quarter always looks like a decline otherwise.
- Say plainly when a movement is too small or too thinly-sampled to be real.
  "Up 2 complaints on a base of 7" is noise and should be labelled as such.
- Report one or two sentences of trend, with the numbers in them. You are not
  writing the final brief — the Synthesis agent does that.
- The instant you have nothing further to add, transfer to SynthesisAgent. Never
  end your turn with a plain-text answer and no transfer; that silently ends the
  whole conversation before a brief is written.
"""


def build_agent() -> ChatCompletionAgent:
    tools = TrendTools()
    return ChatCompletionAgent(
        kernel=build_kernel(),
        arguments=default_arguments(),
        name=NAME,
        description="Computes complaint volume, issue mix and quarter-over-quarter movement.",
        instructions=INSTRUCTIONS,
        plugins=[tools],
    )
