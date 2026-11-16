"""Small local fallback ladder for OpenAI-compatible JSON output modes."""

from __future__ import annotations

from typing import Optional


_MODES: dict[tuple[str, str], str] = {}


def effective_response_format(
    model: str, requested: dict
) -> Optional[dict]:
    mode = _MODES.get((model, str(requested.get("type", ""))), "requested")
    if mode == "requested":
        return requested
    if mode == "object":
        return {"type": "json_object"}
    return None


def note_rejection(
    model: str, requested: dict, error: Exception
) -> bool:
    text = str(error).lower()
    if "response_format" not in text:
        return False
    if not any(
        word in text
        for word in ("unavailable", "not support", "unsupported")
    ):
        return False

    requested_type = str(requested.get("type", ""))
    key = (model, requested_type)
    mode = _MODES.get(key, "requested")
    if mode == "requested":
        next_mode = "object" if requested_type == "json_schema" else "none"
    elif mode == "object":
        next_mode = "none"
    else:
        return False
    _MODES[key] = next_mode
    return True
