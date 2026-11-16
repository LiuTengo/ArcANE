"""Load generated stories and validate text-grounded evidence spans."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from . import config
from .models import EvidenceCheck, StoryInput, StorySegment


_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def normalize_text(value: object) -> str:
    """Normalize text only for evidence matching, never for stored quotes."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def safe_story_id(value: str) -> str:
    slug = _SAFE_ID_RE.sub("-", value.strip()).strip(".-")
    return slug[:80] or "story"


def _chunk_paragraph(
    text: str, absolute_start: int, limit: int
) -> Iterable[tuple[str, int, int]]:
    pos = 0
    while pos < len(text):
        remaining = len(text) - pos
        if remaining <= limit:
            piece = text[pos:]
            yield piece, absolute_start + pos, absolute_start + len(text)
            return

        window = text[pos : pos + limit]
        cut = max(
            window.rfind("\n"),
            window.rfind(" "),
            window.rfind("\t"),
        )
        if cut < max(1, limit // 2):
            cut = limit
        piece = text[pos : pos + cut].rstrip()
        if not piece:
            piece = text[pos : pos + limit]
            cut = len(piece)
        yield piece, absolute_start + pos, absolute_start + pos + cut
        pos += cut
        while pos < len(text) and text[pos].isspace():
            pos += 1


def segment_story(
    text: str, *, target_chars: int | None = None
) -> tuple[StorySegment, ...]:
    """Split a story into ordered, auditable paragraph spans.

    Blank-line paragraphs are the primary unit. A very long paragraph is
    split at a nearby whitespace boundary so evidence remains locatable.
    """
    if not text.strip():
        return ()
    limit = target_chars or config.SEGMENT_CHARS
    if limit < 200:
        raise ValueError("segment size must be at least 200 characters")

    paragraphs: list[tuple[str, int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_RE.finditer(text):
        piece = text[cursor : match.start()]
        if piece.strip():
            paragraphs.append((piece.strip(), cursor, match.start()))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        paragraphs.append((tail.strip(), cursor, len(text)))

    segments: list[StorySegment] = []
    for paragraph, start, _end in paragraphs:
        for piece, piece_start, piece_end in _chunk_paragraph(
            paragraph, start, limit
        ):
            if not piece.strip():
                continue
            segments.append(
                StorySegment(
                    span_id=f"P{len(segments) + 1:04d}",
                    text=piece.strip(),
                    start=piece_start,
                    end=piece_end,
                )
            )
    return tuple(segments)


def load_story(
    path: str | Path,
    *,
    story_id: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StoryInput:
    source_path = Path(path).expanduser().resolve()
    text = source_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"story is empty: {source_path}")
    if len(text) > config.MAX_STORY_CHARS:
        raise ValueError(
            f"story has {len(text):,} characters, above "
            f"GENARC_MAX_STORY_CHARS={config.MAX_STORY_CHARS:,}. "
            "This package deliberately refuses silent truncation because it "
            "would invalidate late-arc evidence. Split the story into "
            "narrative books/parts and evaluate each part separately."
        )

    resolved_id = safe_story_id(story_id or source_path.stem)
    resolved_title = title or source_path.stem.replace("_", " ").replace("-", " ")
    segments = segment_story(text)
    return StoryInput(
        story_id=resolved_id,
        title=resolved_title,
        text=text.strip(),
        source_path=source_path,
        metadata=dict(metadata or {}),
        segments=segments,
    )


def load_manifest(path: str | Path) -> list[StoryInput]:
    """Load a JSON array/object or JSONL manifest.

    Required per item: ``story_path`` or ``text``. Optional fields are
    ``story_id``, ``title`` and arbitrary metadata. Relative story paths are
    resolved against the manifest directory.
    """
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if isinstance(parsed, dict):
        parsed = parsed.get("stories", [parsed])
    if not isinstance(parsed, list):
        raise ValueError("manifest must be a JSON array, object, or JSONL")

    stories: list[StoryInput] = []
    seen: set[str] = set()
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {index} is not an object")
        metadata = dict(item.get("metadata") or {})
        for key in ("generator", "condition", "seed", "control", "prompt_id"):
            if key in item:
                metadata[key] = item[key]

        title = item.get("title")
        story_id = item.get("story_id") or f"story-{index + 1:04d}"
        if item.get("story_path"):
            story_path = Path(item["story_path"])
            if not story_path.is_absolute():
                story_path = manifest_path.parent / story_path
            story = load_story(
                story_path,
                story_id=story_id,
                title=title,
                metadata=metadata,
            )
        elif isinstance(item.get("text"), str):
            text = item["text"].strip()
            if not text:
                raise ValueError(f"manifest item {index} has empty text")
            if len(text) > config.MAX_STORY_CHARS:
                raise ValueError(
                    f"manifest item {index} exceeds GENARC_MAX_STORY_CHARS"
                )
            story = StoryInput(
                story_id=safe_story_id(story_id),
                title=title or safe_story_id(story_id),
                text=text,
                source_path=manifest_path,
                metadata=metadata,
                segments=segment_story(text),
            )
        else:
            raise ValueError(
                f"manifest item {index} needs a story_path or inline text"
            )

        if story.story_id in seen:
            raise ValueError(f"duplicate story_id: {story.story_id}")
        seen.add(story.story_id)
        stories.append(story)
    return stories


def render_story_block(story: StoryInput) -> str:
    parts = [
        f"TITLE: {story.title}",
        f"STORY ID: {story.story_id}",
        "SEGMENTS (chronological; quote evidence verbatim from one segment):",
    ]
    for segment in story.segments:
        parts.append(
            f"[{segment.span_id}]\n{segment.text}\n[/{segment.span_id}]"
        )
    return "\n\n".join(parts)


def ground_evidence(
    story: StoryInput,
    *,
    span_ids: object,
    quote: object,
    kind: object = "narrator",
    note: object = "",
) -> EvidenceCheck:
    """Validate that a quote is a verbatim substring of a cited segment."""
    ids = tuple(
        str(item).strip()
        for item in (span_ids if isinstance(span_ids, list) else [span_ids])
        if str(item or "").strip()
    )
    quote_text = str(quote or "").strip()
    kind_text = str(kind or "narrator").strip()[:40]
    note_text = str(note or "").strip()[:600]

    if not ids:
        return EvidenceCheck(
            span_ids=(),
            quote=quote_text,
            kind=kind_text,
            note=note_text,
            error="missing span_ids",
        )
    if len(quote_text) < config.MIN_EVIDENCE_QUOTE_CHARS:
        return EvidenceCheck(
            span_ids=ids,
            quote=quote_text,
            kind=kind_text,
            note=note_text,
            error=(
                "quote is shorter than "
                f"{config.MIN_EVIDENCE_QUOTE_CHARS} characters"
            ),
        )
    if len(quote_text) > config.MAX_EVIDENCE_QUOTE_CHARS:
        return EvidenceCheck(
            span_ids=ids,
            quote=quote_text,
            kind=kind_text,
            note=note_text,
            error=(
                "quote is longer than "
                f"{config.MAX_EVIDENCE_QUOTE_CHARS} characters"
            ),
        )

    by_id = {segment.span_id: segment for segment in story.segments}
    missing = [span_id for span_id in ids if span_id not in by_id]
    if missing:
        return EvidenceCheck(
            span_ids=ids,
            quote=quote_text,
            kind=kind_text,
            note=note_text,
            error=f"unknown span_id(s): {', '.join(missing)}",
        )

    normalized_quote = normalize_text(quote_text)
    for span_id in ids:
        if normalized_quote in normalize_text(by_id[span_id].text):
            return EvidenceCheck(
                span_ids=ids,
                quote=quote_text,
                kind=kind_text,
                note=note_text,
                valid=True,
            )
    return EvidenceCheck(
        span_ids=ids,
        quote=quote_text,
        kind=kind_text,
        note=note_text,
        error="quote does not occur in any cited span",
    )
