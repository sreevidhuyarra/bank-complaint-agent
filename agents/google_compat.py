"""Compatibility shim for Google AI Studio (Gemini) via Semantic Kernel.

Semantic Kernel 1.44.1 (the latest published release at the time this was
found — there is no newer version to upgrade to for a fix) hardcodes
`role="function"` when replaying a tool's result back to Gemini, in
`GoogleAIChatCompletion._prepare_chat_history_for_request`. Google's live API
rejects that role outright:

    400 ... "Role 'function' is not supported. Please use a valid role:
    SYSTEM, SYSTEM_1, USER, ASSISTANT, DEVELOPER, CONTEXT, USER_CONTEXT,
    MODEL, USER."

confirmed against a real key, not from documentation — this project's own
handoff-mode test hit it directly. "user" is the role Gemini's function-
calling convention actually uses for returning a `functionResponse` Part to
the model (mirroring how a `functionCall` from the model arrives inside an
"model"-role turn), so this subclass overrides just the one broken method
with that one-line fix. Everything else in the connector — request building,
response parsing, streaming — is untouched.
"""

from __future__ import annotations

from google.genai.types import Content
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.connectors.ai.google.google_ai.services.utils import (
    format_assistant_message,
    format_tool_message,
    format_user_message,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole


class PatchedGoogleAIChatCompletion(GoogleAIChatCompletion):
    """GoogleAIChatCompletion with the rejected role="function" fixed to "user".

    Byte-for-byte identical to SK's own `_prepare_chat_history_for_request`
    except the TOOL branch's role string — see module docstring for why.
    """

    def _prepare_chat_history_for_request(
        self,
        chat_history: ChatHistory,
        role_key: str = "role",
        content_key: str = "content",
    ) -> list[Content]:
        chat_request_messages: list[Content] = []
        for message in chat_history.messages:
            if message.role == AuthorRole.SYSTEM:
                continue  # system messages go in system_instruction, not here
            if message.role == AuthorRole.USER:
                chat_request_messages.append(Content(role="user", parts=format_user_message(message)))
            elif message.role == AuthorRole.ASSISTANT:
                chat_request_messages.append(Content(role="model", parts=format_assistant_message(message)))
            elif message.role == AuthorRole.TOOL:
                chat_request_messages.append(Content(role="user", parts=format_tool_message(message)))
        return chat_request_messages
