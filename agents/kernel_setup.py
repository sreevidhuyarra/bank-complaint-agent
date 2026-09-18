"""Semantic Kernel wiring — one shared chat service for the whole agent team.

Two providers are supported, selected by `config.LLM_PROVIDER`:

`groq`    (default) Every agent reasons through Groq's free endpoint. Groq
          exposes an OpenAI-compatible API, so SK's OpenAI connector drives it
          unchanged — retargeting it away from api.openai.com means
          constructing an `openai.AsyncOpenAI` client against Groq's base URL
          and injecting it via `async_client=`.

`google`  Gemini via Google AI Studio, through SK's own native
          `GoogleAIChatCompletion` connector (no OpenAI-shim trick needed —
          Anthropic and Google both ship first-party SK connectors, since
          their wire formats aren't OpenAI-compatible the way Groq's is).

Switching providers is entirely contained to this file: every agent, tool, and
the orchestrator's handoff graph are unchanged either way.
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

import openai
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import KernelArguments

from config import (
    GOOGLE_MODEL,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_PROVIDER,
    google_api_key,
    groq_api_key,
)

SERVICE_ID = "llm"  # provider-agnostic now that more than one is supported


class MissingApiKey(RuntimeError):
    """Raised when no GROQ_API_KEY is present in the environment."""


class QuotaExhausted(RuntimeError):
    """Raised instead of retrying when a provider's suggested wait is too long
    to sit through inside a live request — a daily-quota 429 rather than the
    routine per-minute one (see `with_rate_limit_retry`)."""


@dataclass
class LlmConfig:
    provider: str = LLM_PROVIDER    # "groq" or "google" — see config.LLM_PROVIDER
    model: str | None = None        # None -> provider's default (filled in below)
    base_url: str = GROQ_BASE_URL   # only meaningful for the groq (OpenAI-shim) path
    temperature: float = 0.2        # analyst briefs should be reproducible
    # Every tool-calling-capable model on Groq's free tier shares the same tight
    # 8,000 tokens/minute cap (verified against this project's own key — see
    # "Rate limits" in the README). A lower ceiling here leaves more of that
    # budget for the next agent's turn in a handoff chain. Google's free tier
    # is request-count-limited rather than token-limited, so this ceiling is
    # purely about output length there, not quota protection.
    max_tokens: int = 900

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = GOOGLE_MODEL if self.provider == "google" else GROQ_MODEL


def build_chat_service(config: LlmConfig | None = None) -> ChatCompletionClientBase:
    """Create the shared chat-completion service for whichever provider is configured."""
    config = config or LlmConfig()
    if config.provider == "google":
        return _build_google_chat_service(config)
    return _build_groq_chat_service(config)


def _build_groq_chat_service(config: LlmConfig) -> OpenAIChatCompletion:
    from openai import AsyncOpenAI

    from agents.groq_compat import tool_alias_http_client

    api_key = groq_api_key()
    if not api_key:
        raise MissingApiKey(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and set it in your environment (or as a Space secret) before running the agents."
        )

    return OpenAIChatCompletion(
        service_id=SERVICE_ID,
        ai_model_id=config.model,
        async_client=AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url,
            # See agents/groq_compat.py — Groq's gpt-oss models sometimes call a
            # tool by its bare name, dropping the plugin prefix SK adds; this
            # patches the outgoing request so both forms validate.
            http_client=tool_alias_http_client(),
        ),
    )


def _build_google_chat_service(config: LlmConfig) -> "PatchedGoogleAIChatCompletion":
    # Imported lazily so a Groq-only install never needs google-genai on disk.
    from agents.google_compat import PatchedGoogleAIChatCompletion

    api_key = google_api_key()
    if not api_key:
        raise MissingApiKey(
            "GOOGLE_AI_API_KEY is not set. Get a free key at https://aistudio.google.com "
            "and set it in your environment (or as a Space secret) before running the agents."
        )

    return PatchedGoogleAIChatCompletion(
        service_id=SERVICE_ID,
        gemini_model_id=config.model,
        api_key=api_key,
    )


def build_kernel(plugins: dict[str, object] | None = None,
                 config: LlmConfig | None = None) -> Kernel:
    """A kernel carrying the Groq service and any tool plugins passed in."""
    kernel = Kernel()
    kernel.add_service(build_chat_service(config))
    for name, plugin in (plugins or {}).items():
        kernel.add_plugin(plugin, plugin_name=name)
    return kernel


def execution_settings(config: LlmConfig | None = None,
                       auto_tools: bool = True) -> PromptExecutionSettings:
    """Low temperature and auto tool-calling for every agent.

    `ChatCompletionAgent` already defaults to `FunctionChoiceBehavior.Auto`, but
    it does not set a temperature, and an analyst brief that changes wording on
    every run is not something a reviewer can sign off on.
    """
    config = config or LlmConfig()

    if config.provider == "google":
        from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatPromptExecutionSettings

        settings: PromptExecutionSettings = GoogleAIChatPromptExecutionSettings(
            service_id=SERVICE_ID,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,  # Google's field name differs from OpenAI's
        )
    else:
        settings = OpenAIChatPromptExecutionSettings(
            service_id=SERVICE_ID,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if auto_tools:
        # SK forces tool_choice="none" once a turn exceeds this many automatic
        # invocation rounds, to make the model wrap up. gpt-oss models on Groq
        # sometimes try to call a tool anyway on that final round, which Groq
        # then rejects outright ("Tool choice is none, but model called a
        # tool"). A higher ceiling makes that forced cutoff rare in practice for
        # the 1-3 tool calls a turn here actually needs. Untested whether Gemini
        # ever needs this same headroom — kept the same for both providers until
        # proven otherwise.
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto(maximum_auto_invoke_attempts=10)
    return settings


def default_arguments(config: LlmConfig | None = None) -> KernelArguments:
    """KernelArguments carrying the shared execution settings."""
    return KernelArguments(settings=execution_settings(config))


def llm_available(provider: str | None = None) -> bool:
    """Whether the agent team can run at all. The UI uses this to degrade gracefully."""
    provider = provider or LLM_PROVIDER
    return google_api_key() is not None if provider == "google" else groq_api_key() is not None


# --------------------------------------------------------------------------- #
# Resilience: retrying transient Groq failures
# --------------------------------------------------------------------------- #
# Two failure classes here are routine, not exceptional, on this stack:
#
# 1. Rate limiting. Every tool-calling model on Groq's free tier shares two
#    ceilings, measured against a real key: a tight per-minute one (8,000
#    tokens/minute, 1,000 requests/minute) and a separate per-day one (200,000
#    tokens/day, observed on the same key). A per-minute 429 recovers in well
#    under a second and is worth retrying in place. A per-day 429 can quote a
#    wait of several minutes — sitting through that inside a live request would
#    just hang the UI, so it's raised immediately as `QuotaExhausted` instead
#    (see `_MAX_INLINE_WAIT`).
# 2. gpt-oss occasionally ignores a forced `tool_choice: "none"` (issued once a
#    turn's automatic tool-call budget is spent, see `execution_settings`) and
#    tries to call a tool anyway; Groq then rejects the whole request with a
#    400. This is model sampling noise, not a deterministic bug — a fresh
#    attempt usually just answers in prose as asked.

# Groq quotes a wait as "37ms", "165ms", "3m8.352s", or (in principle) something
# with an hours component — variable units concatenated with no separator, so
# each unit is matched independently rather than assuming a fixed format.
# "ms" must be tried before "m"/"s" alone, or "500ms" mis-parses as 500 minutes.
_WAIT_UNIT_RE = re.compile(r"([\d.]+)\s*(ms|h|m|s)(?![a-z])", re.IGNORECASE)
_WAIT_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
# Above this, don't sit through it inline — a multi-minute sleep inside a live
# request just hangs the UI with no feedback; raise QuotaExhausted instead so
# the caller can show the wait time and let the user decide when to retry.
_MAX_INLINE_WAIT = 20.0
# Streaming-level generation glitches from Groq's gpt-oss models, all raised as
# a plain openai.APIError from inside the SSE stream (not an HTTP status error)
# and all clearing on a fresh sampling attempt rather than being deterministic.
_TRANSIENT_GENERATION_MARKERS = (
    "tool choice is none, but model called a tool",
    "failed to parse tool call arguments as json",
)
T = TypeVar("T")


def is_rate_limit_error(exc: BaseException) -> bool:
    """Whether `exc` is (or wraps) a Groq/OpenAI 429 rate-limit error."""
    return _rate_limit_error(exc) is not None


def is_transient_error(exc: BaseException) -> bool:
    """Whether `exc` is a known-retryable Groq failure — rate limit or a
    streaming-level generation glitch (see module docstring above)."""
    return _is_transient(exc)


def _find_cause(exc: BaseException, predicate: Callable[[BaseException], bool]) -> BaseException | None:
    """Walk `exc.__cause__` looking for a match.

    SK wraps the original openai exception as `ServiceResponseException(msg, ex)`
    — walking the cause chain unwraps that without coupling this to SK's
    internals.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if predicate(exc):
            return exc
        exc = exc.__cause__
    return None


def _google_error(exc: BaseException) -> BaseException | None:
    """Find a google-genai APIError in the cause chain, if that package is even
    installed — a Groq-only install has no reason to depend on it."""
    try:
        from google.genai.errors import APIError as GoogleAPIError
    except ImportError:
        return None
    return _find_cause(exc, lambda e: isinstance(e, GoogleAPIError))


def _rate_limit_error(exc: BaseException) -> BaseException | None:
    if (found := _find_cause(exc, lambda e: isinstance(e, openai.RateLimitError))) is not None:
        return found
    google_exc = _google_error(exc)
    # google-genai's APIError.code carries the HTTP status directly (429 here),
    # per google.genai.errors.APIError — read from its installed source, not
    # from a live 429 we've actually seen yet (unlike Groq's wait-time regex
    # below, which was tuned against a real captured error).
    if google_exc is not None and getattr(google_exc, "code", None) == 429:
        return google_exc
    return None


def _is_transient(exc: BaseException) -> bool:
    if _rate_limit_error(exc) is not None:
        return True
    google_exc = _google_error(exc)
    if google_exc is not None and getattr(google_exc, "code", 0) >= 500:
        return True  # a 5xx from Google is generically worth a fresh attempt
    return _find_cause(
        exc,
        lambda e: isinstance(e, openai.APIError)
        and any(marker in str(e).lower() for marker in _TRANSIENT_GENERATION_MARKERS),
    ) is not None


def _suggested_wait_seconds(exc: BaseException, default: float | None = None) -> float | None:
    """Parse a provider's "retry after" hint into seconds, or `default` if absent.

    Groq quotes this in its error message text ("try again in 3m8.352s"), which
    this regex is tuned against real captured errors for. Google's equivalent
    hasn't been observed live yet in this project — a `Retry-After` response
    header is checked as a reasonable guess, but treat this path as unverified
    until it's been seen against a real 429.
    """
    google_exc = _google_error(exc)
    if google_exc is not None:
        header = getattr(getattr(google_exc, "response", None), "headers", {}) or {}
        retry_after = header.get("retry-after") or header.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return default

    text = str(exc)
    start = text.lower().find("try again in")
    matches = list(_WAIT_UNIT_RE.finditer(text[start:] if start != -1 else text))
    if not matches:
        return default
    return sum(float(m.group(1)) * _WAIT_UNIT_SECONDS[m.group(2).lower()] for m in matches)


async def with_rate_limit_retry(
    call: Callable[[], Awaitable[T]],
    max_attempts: int = 4,
    min_wait: float = 0.5,
) -> T:
    """Run `call`, retrying transient Groq failures (see module docstring above).

    A per-minute 429's hinted wait is usually well under a second, but that hint
    reflects the instant the error was raised, not the instant we retry — a
    floor and an increasing backoff make the retry likely to land after the
    window has actually freed up rather than repeating the same 429 immediately.

    A per-day 429 quotes a wait far past what's reasonable to sleep through
    inside a live request; that raises `QuotaExhausted` immediately instead of
    consuming retry attempts on a wait that won't have elapsed by the last one.
    """
    last_error: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 - narrowed by _is_transient below
            if not _is_transient(exc):
                raise
            suggested = _suggested_wait_seconds(exc)
            if suggested is not None and suggested > _MAX_INLINE_WAIT:
                provider = "Google AI Studio" if _google_error(exc) is not None else "Groq"
                raise QuotaExhausted(
                    f"{provider}'s rate limit won't clear for about {_format_wait(suggested)} "
                    "- this is likely a daily quota, not a per-minute one. Wait for it to "
                    "reset, or switch provider/model/key (see config.LLM_PROVIDER)."
                ) from exc
            if attempt == max_attempts - 1:
                raise
            last_error = exc
            wait = max(min_wait, suggested or min_wait) * (2 ** attempt)
            wait += random.uniform(0, 0.25)  # jitter so parallel calls don't retry in lockstep
            await asyncio.sleep(wait)
    raise last_error  # pragma: no cover - loop always returns or raises above


def _format_wait(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes:.0f}m{rest:.0f}s"
