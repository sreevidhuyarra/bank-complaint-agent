"""Semantic Kernel wiring — one shared chat service for the whole agent team.

Every agent reasons through Gemini via Google AI Studio, through SK's own
native `GoogleAIChatCompletion` connector (patched — see agents/google_compat.py
for a hardcoded-role bug found live against a real key). This project
previously also supported Groq's free tier and two Azure OpenAI surfaces; both
were removed after extensive live debugging turned up provider-specific
reliability problems (Groq's tight per-minute/per-day token budget forcing a
lot of retry/fallback machinery; Azure's Responses API needing a genuinely
different SK `Agent` class and hitting several deployment-specific bugs with
no clean fix) that Gemini didn't share — one well-tested path beats three
partially-tested ones.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import KernelArguments

from config import GOOGLE_MODEL, google_api_key

SERVICE_ID = "llm"


class MissingApiKey(RuntimeError):
    """Raised when no GOOGLE_AI_API_KEY is present in the environment."""


class QuotaExhausted(RuntimeError):
    """Raised instead of retrying when Google's suggested wait is too long to
    sit through inside a live request — a daily-quota 429 rather than the
    routine per-minute one (see `with_rate_limit_retry`)."""


@dataclass
class LlmConfig:
    model: str = GOOGLE_MODEL
    temperature: float = 0.2        # analyst briefs should be reproducible
    max_tokens: int = 900            # output-length ceiling; Synthesis overrides this higher


def build_chat_service(config: LlmConfig | None = None) -> ChatCompletionClientBase:
    """Create the shared Gemini chat-completion service."""
    from agents.google_compat import PatchedGoogleAIChatCompletion

    config = config or LlmConfig()
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
    """A kernel carrying the Gemini service and any tool plugins passed in."""
    kernel = Kernel()
    kernel.add_service(build_chat_service(config))
    for name, plugin in (plugins or {}).items():
        kernel.add_plugin(plugin, plugin_name=name)
    return kernel


def build_agent(
    *,
    name: str,
    description: str,
    instructions: str,
    plugins: list[object] | None = None,
    tool_choice: str = "auto",
    config: LlmConfig | None = None,
):
    from semantic_kernel.agents import ChatCompletionAgent

    return ChatCompletionAgent(
        kernel=build_kernel(config=config),
        arguments=default_arguments(config, tool_choice),
        name=name,
        description=description,
        instructions=instructions,
        plugins=plugins,
    )


def execution_settings(config: LlmConfig | None = None,
                       tool_choice: str = "auto") -> PromptExecutionSettings:
    """Low temperature and tool-calling behavior for an agent.

    `tool_choice`:
      "auto"     model may call a tool or just answer in text. Right for
                 SynthesisAgent, whose whole job is answering in text — it has
                 one optional escape-hatch function it should use rarely, not
                 be forced into every turn.
      "required" model MUST call one of its declared functions this turn —
                 maps to Gemini's `function_calling_config.mode = "ANY"` (see
                 google/shared_utils.py's FUNCTION_CHOICE_TYPE_TO_GOOGLE_
                 FUNCTION_CALLING_MODE). Right for the Orchestrator and the
                 three specialists: every one of them is designed to always
                 end a turn by calling either a domain tool or a
                 transfer_to_* function, never by just answering in prose.
                 `Auto` permits that "just answer instead of routing" failure
                 — the "Orchestrator ended the run without routing to a
                 specialist" sentinel this project hit repeatedly is exactly
                 that: the model was *allowed* to skip the tool call, and
                 sometimes did. `Required` removes the option instead of
                 hoping a prompt instruction is enough.
      "none"     no tools exposed at all.
    """
    from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatPromptExecutionSettings

    config = config or LlmConfig()
    settings: PromptExecutionSettings = GoogleAIChatPromptExecutionSettings(
        service_id=SERVICE_ID,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )

    if tool_choice == "required":
        settings.function_choice_behavior = FunctionChoiceBehavior.Required(maximum_auto_invoke_attempts=10)
    elif tool_choice == "auto":
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto(maximum_auto_invoke_attempts=10)
    # "none" (or anything else): leave function_choice_behavior unset.
    return settings


def default_arguments(config: LlmConfig | None = None, tool_choice: str = "auto") -> KernelArguments:
    """KernelArguments carrying the shared execution settings."""
    return KernelArguments(settings=execution_settings(config, tool_choice))


def llm_available() -> bool:
    """Whether the agent team can run at all. The UI uses this to degrade gracefully."""
    return google_api_key() is not None


# --------------------------------------------------------------------------- #
# Resilience: retrying transient Google failures
# --------------------------------------------------------------------------- #
# Two failure classes here are routine, not exceptional:
#
# 1. Rate limiting. A per-minute 429 usually recovers in well under a second
#    and is worth retrying in place. A 429 with a long suggested wait (a daily
#    quota, not a per-minute one) would just hang the UI if slept through
#    inside a live request, so it's raised immediately as `QuotaExhausted`
#    instead (see `_MAX_INLINE_WAIT`).
# 2. Transient 5xx errors — generically worth a fresh attempt rather than
#    failing the whole turn.
T = TypeVar("T")

# Above this, don't sit through it inline — a multi-minute sleep inside a live
# request just hangs the UI with no feedback; raise QuotaExhausted instead so
# the caller can show the wait time and let the user decide when to retry.
_MAX_INLINE_WAIT = 20.0


def is_rate_limit_error(exc: BaseException) -> bool:
    """Whether `exc` is (or wraps) a Google 429 rate-limit error."""
    return _rate_limit_error(exc) is not None


def is_transient_error(exc: BaseException) -> bool:
    """Whether `exc` is a known-retryable Google failure (see module docstring above)."""
    return _is_transient(exc)


def _find_cause(exc: BaseException, predicate: Callable[[BaseException], bool]) -> BaseException | None:
    """Walk `exc.__cause__` looking for a match.

    SK wraps the original google-genai exception as `ServiceResponseException(msg, ex)`
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
    from google.genai.errors import APIError as GoogleAPIError

    return _find_cause(exc, lambda e: isinstance(e, GoogleAPIError))


def _rate_limit_error(exc: BaseException) -> BaseException | None:
    # google-genai's APIError.code carries the HTTP status directly (429 here),
    # per google.genai.errors.APIError, read from its installed source.
    google_exc = _google_error(exc)
    if google_exc is not None and getattr(google_exc, "code", None) == 429:
        return google_exc
    return None


def _is_transient(exc: BaseException) -> bool:
    if _rate_limit_error(exc) is not None:
        return True
    google_exc = _google_error(exc)
    return google_exc is not None and getattr(google_exc, "code", 0) >= 500


def _suggested_wait_seconds(exc: BaseException, default: float | None = None) -> float | None:
    """Parse Google's "retry after" hint into seconds, or `default` if absent.

    Unverified against a real captured 429 — this reads a `Retry-After`
    response header as a reasonable guess, but treat this path as unconfirmed
    until it's been seen live.
    """
    google_exc = _google_error(exc)
    if google_exc is None:
        return default
    header = getattr(getattr(google_exc, "response", None), "headers", {}) or {}
    retry_after = header.get("retry-after") or header.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return default


async def with_rate_limit_retry(
    call: Callable[[], Awaitable[T]],
    max_attempts: int = 4,
    min_wait: float = 0.5,
) -> T:
    """Run `call`, retrying transient Google failures (see module docstring above).

    A per-minute 429's hinted wait is usually well under a second, but that hint
    reflects the instant the error was raised, not the instant we retry — a
    floor and an increasing backoff make the retry likely to land after the
    window has actually freed up rather than repeating the same 429 immediately.

    A quota 429 quotes a wait far past what's reasonable to sleep through
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
                raise QuotaExhausted(
                    f"Google AI Studio's rate limit won't clear for about "
                    f"{_format_wait(suggested)} — this is likely a daily quota, not a "
                    "per-minute one. Wait for it to reset before trying again."
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
