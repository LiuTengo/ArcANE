"""Data structures and constants for generated-story arc evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "generated_arc_v1"

# Four dimensions analogous to ArcANE's three phase metrics plus PTF.
DIMENSIONS = (
    "action_progression",
    "reasoning_progression",
    "reasoning_action_entailment",
    "trajectory_coherence",
)

ARC_LABELS = frozenset({"present", "absent", "uncertain"})
CHANGE_TYPES = frozenset(
    {"positive", "negative", "nonlinear", "static", "unclear"}
)


@dataclass(frozen=True)
class StorySegment:
    span_id: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class StoryInput:
    story_id: str
    title: str
    text: str
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    segments: tuple[StorySegment, ...] = ()

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        data = {
            "story_id": self.story_id,
            "title": self.title,
            "source_path": str(self.source_path),
            "metadata": dict(self.metadata),
            "n_segments": len(self.segments),
        }
        if include_text:
            data["text"] = self.text
            data["segments"] = [segment.to_dict() for segment in self.segments]
        return data


@dataclass(frozen=True)
class EvidenceCheck:
    span_ids: tuple[str, ...]
    quote: str
    kind: str = "narrator"
    note: str = ""
    valid: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_ids": list(self.span_ids),
            "quote": self.quote,
            "kind": self.kind,
            "note": self.note,
            "valid": self.valid,
            "error": self.error,
        }
