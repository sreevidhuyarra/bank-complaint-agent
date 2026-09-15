"""Compatibility shim for Groq's tool-calling models.

Semantic Kernel exposes each plugin function to the model as a fully-qualified
name — `{PluginName}-{function_name}` — and its client-side resolver already
tolerates a bare, unqualified name in the response: `Kernel.get_function`
searches every plugin for a matching function name when no plugin is given
(semantic_kernel/functions/kernel_function_extension.py:276), and
`FunctionCallContent` leaves `plugin_name` unset whenever the incoming name has
no separator in it.

The problem sits one layer below that, on the wire: Groq's `gpt-oss` models
reliably call transfer_to_* handoff functions by their full qualified name, but
just as reliably call this project's own tools (`search_complaints`,
`score_complaints`, ...) by their bare name — dropping the plugin prefix. Groq's
own backend then rejects the whole request with a 400 ("attempted to call tool
'search_complaints' which was not in request.tools") before the response ever
reaches SK, so its tolerant client-side resolver never gets a chance to run.

The fix: duplicate every function-type tool in the outgoing request under its
bare name as well as its qualified name. Whichever one the model calls, Groq's
validator finds a match, and SK's existing fallback resolves it correctly from
there. This is applied via a custom httpx transport rather than a `request`
event hook because event hooks are read-only in httpx — mutating the body
requires intercepting and rebuilding the request.
"""

from __future__ import annotations

import json

import httpx

SEPARATOR = "-"  # semantic_kernel.const.DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR


def _with_bare_name_aliases(tools: list[dict]) -> list[dict]:
    """Add a same-schema alias tool under the bare name for each qualified tool."""
    seen_names = {t.get("function", {}).get("name") for t in tools}
    extra = []
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if tool.get("type") != "function" or SEPARATOR not in name:
            continue
        bare_name = name.rsplit(SEPARATOR, 1)[-1]
        if bare_name in seen_names:
            continue  # two plugins expose the same bare name — ambiguous, skip
        seen_names.add(bare_name)
        extra.append({
            "type": "function",
            "function": {
                "name": bare_name,
                # Kept short: this alias exists only so Groq's validator accepts
                # whichever name the model happens to call, not to be chosen for
                # its description — a full copy would double the tool-schema
                # token cost on every turn against an 8,000 TPM budget.
                "description": f"Alias of {name}. Same arguments, same behavior.",
                "parameters": fn.get("parameters", {}),
            },
        })
    return tools + extra


class _ToolAliasTransport(httpx.AsyncBaseTransport):
    """Wraps another transport, rewriting outgoing chat-completion tool lists."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport):
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.content:
            try:
                body = json.loads(request.content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = None
            if isinstance(body, dict) and body.get("tools"):
                body["tools"] = _with_bare_name_aliases(body["tools"])
                new_body = json.dumps(body).encode("utf-8")
                headers = httpx.Headers(request.headers)
                headers.pop("content-length", None)  # let httpx recompute it
                request = httpx.Request(
                    method=request.method,
                    url=request.url,
                    headers=headers,
                    content=new_body,
                )
        return await self._wrapped.handle_async_request(request)

    async def aclose(self) -> None:
        await self._wrapped.aclose()


def tool_alias_http_client() -> httpx.AsyncClient:
    """An httpx.AsyncClient that patches tool-call name mismatches for Groq."""
    return httpx.AsyncClient(transport=_ToolAliasTransport(httpx.AsyncHTTPTransport()))
