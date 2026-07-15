from __future__ import annotations

import copy
from typing import Any

from litellm import Message


def add_anthropic_caching(
    messages: list[dict[str, Any] | Message], model_name: str
) -> list[dict[str, Any] | Message]:
    """Add ephemeral cache hints to recent Anthropic messages."""
    if not ("anthropic" in model_name.lower() or "claude" in model_name.lower()):
        return messages

    cached_messages = copy.deepcopy(messages)
    for index, message in enumerate(cached_messages):
        if index < len(cached_messages) - 3:
            continue
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "type" in item:
                        item["cache_control"] = {"type": "ephemeral"}
        elif hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                message.content = [  # type: ignore[assignment]
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "type" in item:
                        item["cache_control"] = {"type": "ephemeral"}

    return cached_messages
