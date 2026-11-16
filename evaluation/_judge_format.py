"""Generic response_format compatibility for OpenAI-compatible judge backends.

Judges request either ``json_schema`` (trajectory) or ``json_object``
(per-response).  Some OpenAI-compatible gateways support only the latter, or
neither.  This module remembers, per (model, requested type), which form the
endpoint accepted and lets callers retry without changing any provider-specific
configuration.
"""

from typing import Optional

_MODES: dict[tuple[str, str], str] = {}

_JSON_HINT = (
    "\n\nReturn only a single valid JSON object, with no markdown fences "
    "or commentary."
)


def effective_response_format(model: str, requested: dict) -> Optional[dict]:
    """Return the response_format to send, or None to omit it entirely."""
    key = (model, str(requested.get("type", "")))
    mode = _MODES.get(key, "requested")
    if mode == "requested":
        return requested
    if mode == "object":
        return {"type": "json_object"}
    return None


def ensure_json_prompt(system: str) -> str:
    """json_object mode requires the word "json" somewhere in the prompt."""
    if "json" in system.lower():
        return system
    return system.rstrip() + _JSON_HINT


def note_response_format_rejection(
        model: str, requested: dict, error: Exception) -> bool:
    """Advance the fallback ladder after a response_format rejection.

    Returns True when the caller should retry immediately in the new mode.
    """
    text = str(error).lower()
    if "response_format" not in text:
        return False
    if not any(word in text for word in
               ("unavailable", "not support", "unsupported")):
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
