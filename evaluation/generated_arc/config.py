"""Configuration for narrative-intrinsic character-arc evaluation.

All variables use a ``GENARC_`` prefix so this experiment does not alter the
configuration of ArcANE's existing probes or role-play judges.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
else:
    # Keep this experiment's credentials separate while allowing the project's
    # conventional evaluation/.env to be picked up automatically.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    # The experiment-local file is authoritative over stale variables left in
    # an IDE terminal. An explicit GENARC_JUDGE_API_KEY still takes precedence
    # below because that variable is intentionally not duplicated in the file.
    load_dotenv(
        PROJECT_ROOT / "evaluation" / ".env",
        override=True,
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BACKEND = os.environ.get("GENARC_JUDGE_BACKEND", "openrouter").strip().lower()

_BACKEND_DEFAULTS = {
    "openai": {
        "model": "gpt-5.4-mini",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "price_input": 0.75,
        "price_output": 4.50,
    },
    "openrouter": {
        "model": "deepseek/deepseek-v4-flash",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "price_input": 0.14,
        "price_output": 0.28,
    },
}

if BACKEND not in _BACKEND_DEFAULTS:
    raise ValueError(
        f"unknown GENARC_JUDGE_BACKEND={BACKEND!r}; "
        f"expected one of {sorted(_BACKEND_DEFAULTS)}"
    )

_DEFAULTS = _BACKEND_DEFAULTS[BACKEND]

MODEL = os.environ.get("GENARC_JUDGE_MODEL", _DEFAULTS["model"])
BASE_URL: Optional[str] = (
    os.environ.get("GENARC_JUDGE_BASE_URL") or _DEFAULTS["base_url"]
)
API_KEY = (
    os.environ.get("GENARC_JUDGE_API_KEY")
    or os.environ.get(_DEFAULTS["api_key_env"])
    or "EMPTY"
)

PROVIDER_ORDER = [
    item.strip()
    for item in os.environ.get(
        "GENARC_JUDGE_PROVIDER_ORDER", "DeepSeek,Alibaba"
    ).split(",")
    if item.strip()
]
PROVIDER_ALLOW_FALLBACK = _env_bool(
    "GENARC_JUDGE_PROVIDER_ALLOW_FALLBACK", False
)
DISABLE_REASONING = _env_bool("GENARC_JUDGE_DISABLE_REASONING", True)

PRICE_INPUT = float(
    os.environ.get("GENARC_JUDGE_PRICE_INPUT", str(_DEFAULTS["price_input"]))
)
PRICE_OUTPUT = float(
    os.environ.get("GENARC_JUDGE_PRICE_OUTPUT", str(_DEFAULTS["price_output"]))
)

SCORE_SCALE = int(os.environ.get("GENARC_SCORE_SCALE", "100"))
ARC_PRESENCE_MIN_SCORE = float(
    os.environ.get("GENARC_ARC_PRESENCE_MIN_SCORE", "60")
)
ARC_PRESENCE_MIN_DIMENSION = float(
    os.environ.get("GENARC_ARC_PRESENCE_MIN_DIMENSION", "40")
)
MIN_EVIDENCE_GROUNDING = float(
    os.environ.get("GENARC_MIN_EVIDENCE_GROUNDING", "0.8")
)
MIN_GROUNDED_PHASES = int(os.environ.get("GENARC_MIN_GROUNDED_PHASES", "2"))

MAX_CHARACTERS = int(os.environ.get("GENARC_MAX_CHARACTERS", "5"))
MAX_AXES_PER_CHARACTER = int(
    os.environ.get("GENARC_MAX_AXES_PER_CHARACTER", "4")
)
EXTRACTION_RUNS = int(os.environ.get("GENARC_EXTRACTION_RUNS", "1"))
SEGMENT_CHARS = int(os.environ.get("GENARC_SEGMENT_CHARS", "3500"))
MAX_STORY_CHARS = int(os.environ.get("GENARC_MAX_STORY_CHARS", "120000"))
MAX_EVIDENCE_QUOTE_CHARS = int(
    os.environ.get("GENARC_MAX_EVIDENCE_QUOTE_CHARS", "600")
)
MIN_EVIDENCE_QUOTE_CHARS = int(
    os.environ.get("GENARC_MIN_EVIDENCE_QUOTE_CHARS", "8")
)

CONCURRENCY = int(os.environ.get("GENARC_CONCURRENCY", "50"))
MAX_RETRIES = int(os.environ.get("GENARC_MAX_RETRIES", "3"))
RETRY_BACKOFF_BASE = float(
    os.environ.get("GENARC_RETRY_BACKOFF_BASE", "2")
)
EST_OUTPUT_TOKENS_PER_EXTRACTION = int(
    os.environ.get("GENARC_EST_OUTPUT_TOKENS_EXTRACTION", "1800")
)
EST_OUTPUT_TOKENS_PER_SCORE = int(
    os.environ.get("GENARC_EST_OUTPUT_TOKENS_SCORE", "500")
)

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "GENARC_OUTPUT_ROOT",
        str(PROJECT_ROOT / "results" / "generated_arc"),
    )
)


def make_judge_client():
    """Create an OpenAI-compatible client lazily."""
    import httpx
    from openai import OpenAI

    if BACKEND == "openrouter" and API_KEY != "EMPTY":
        if not API_KEY.startswith("sk-or-"):
            raise ValueError(
                "GENARC_JUDGE_BACKEND=openrouter requires an OpenRouter key "
                "starting with 'sk-or-'. The configured OPENROUTER_API_KEY or "
                "GENARC_JUDGE_API_KEY does not match. For DeepSeek's "
                "OpenAI-compatible endpoint, set "
                "GENARC_JUDGE_BACKEND=openai, "
                "GENARC_JUDGE_BASE_URL=https://api.deepseek.com, and "
                "GENARC_JUDGE_MODEL=deepseek-flash."
            )

    limit = max(CONCURRENCY, 1)
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.Client(
            limits=httpx.Limits(
                max_connections=limit,
                max_keepalive_connections=limit,
            ),
            timeout=httpx.Timeout(180.0, connect=15.0),
        ),
    )
