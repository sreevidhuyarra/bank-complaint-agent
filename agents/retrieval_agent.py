"""Retrieval Agent — the only agent that touches the complaint index.

Owns semantic search over CFPB narratives. Everything it returns carries a
complaint ID so the Synthesis agent can cite it.
"""

from __future__ import annotations

from typing import Annotated

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function

from agents.kernel_setup import build_kernel, default_arguments
from data.ingest import ComplaintHit, get_index
from nlp.linguistic_risk import ordinal

NAME = "RetrievalAgent"


def format_hits(hits: list[ComplaintHit], excerpt_chars: int = 220) -> str:
    if not hits:
        return "No complaints matched those filters. Try a broader company or product term."
    lines = []
    for hit in hits:
        lines.append(
            f"[{hit.complaint_id}] {hit.date_received} · {hit.company} · "
            f"{hit.product}{' / ' + hit.sub_product if hit.sub_product else ''} · "
            f"issue: {hit.issue} · similarity {hit.score:.2f} · "
            f"pre-scored risk {hit.severity} ({ordinal(hit.risk_percentile)} pct)\n"
            f'  "{hit.excerpt(excerpt_chars)}"'
        )
    return "\n\n".join(lines)


class RetrievalTools:
    """Semantic search over the indexed CFPB complaint narratives."""

    @kernel_function(
        name="search_complaints",
        description=(
            "Semantic search over CFPB consumer-complaint narratives. Returns matching "
            "complaint excerpts with their complaint IDs, dates, products and issues. "
            "Use loose natural terms for company and product — 'Wells Fargo', 'overdraft'."
        ),
    )
    def search_complaints(
        self,
        query: Annotated[str, "What to search for, e.g. 'overdraft fees charged after a deposit cleared'"],
        company: Annotated[str, "Company name filter, or empty for all companies"] = "",
        product: Annotated[str, "Product or sub-product filter, e.g. 'checking' or 'credit card'"] = "",
        issue: Annotated[str, "Exact or partial issue-category filter — use this when you already "
                              "know the category (e.g. from a Trend agent finding like 'Fees or "
                              "interest') and want complaints specifically from it, or empty for none"] = "",
        date_min: Annotated[str, "Earliest complaint date as YYYY-MM-DD, or empty"] = "",
        date_max: Annotated[str, "Latest complaint date as YYYY-MM-DD, or empty"] = "",
        top_k: Annotated[int, "How many complaints to return, 1-8. Keep this small — "
                              "each one is replayed to every later agent in the "
                              "conversation, and the model's token budget is tight."] = 4,
    ) -> Annotated[str, "Matching complaint excerpts with complaint IDs"]:
        hits = get_index().search(
            query=query,
            k=max(1, min(int(top_k), 8)),
            company=company,
            product=product,
            issue=issue,
            date_min=date_min,
            date_max=date_max,
        )
        return format_hits(hits)

    @kernel_function(
        name="dataset_scope",
        description=(
            "Describe what the complaint index actually covers: companies, date range "
            "and total complaint count. Call this before claiming data is unavailable."
        ),
    )
    def dataset_scope(self) -> Annotated[str, "Coverage of the indexed dataset"]:
        index = get_index()
        companies = index.meta.get("companies") or sorted(index.frame["company"].unique())
        products = index.frame["product"].value_counts().head(8)
        return (
            f"Indexed: {index.describe()}\n"
            f"Companies: {', '.join(companies)}\n"
            f"Top products: {', '.join(f'{p} ({n})' for p, n in products.items())}"
        )


INSTRUCTIONS = """You are the Retrieval Agent on a bank complaint-analysis team.

Your only job is to find the complaint narratives that answer the question, using
the search_complaints tool. You do not interpret severity and you do not compute
trends — other agents do that.

Rules:
- Always call search_complaints before answering. Never invent complaint text.
- Translate the analyst's wording into filters: a bank name goes in `company`, a
  product like overdraft/checking/credit card goes in `product`, a stated period
  goes in date_min/date_max. If the Trend agent already named a specific issue
  category, use the `issue` filter to pull complaints from exactly that category
  rather than relying on semantic similarity alone.
- If a search returns nothing, widen it once (drop the product filter, then the
  company filter) and say what you relaxed. Call dataset_scope if you suspect the
  question is outside the indexed coverage.
- Report the complaints you found verbatim, keeping every complaint ID intact.
  IDs are how the final brief cites its evidence, so never paraphrase them away.
- When the question also needs a severity read or a volume trend, hand off to the
  agent that owns it rather than guessing.
"""


def build_agent() -> ChatCompletionAgent:
    tools = RetrievalTools()
    return ChatCompletionAgent(
        # The agent registers `plugins` on its kernel itself — adding them to the
        # kernel here as well would expose every tool to the model twice.
        kernel=build_kernel(),
        arguments=default_arguments(),
        name=NAME,
        description="Finds relevant CFPB complaint narratives by semantic search.",
        instructions=INSTRUCTIONS,
        plugins=[tools],
    )
