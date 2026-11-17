"""Adapt ArcANE character-axis JSON into a source-free judge dossier.

ArcANE Stage 1 stores phase descriptions, chapter ranges, and key moments, but
not verbatim quotations. This adapter supports judging the extracted arc's
internal construction without sending the full novel to the judge.
"""

from __future__ import annotations

import json
import re
import unicodedata
import warnings
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_STORY_MATCH_THRESHOLD = 0.82
_STORY_MATCH_MARGIN = 0.05


def _story_key(value: str) -> str:
    """Normalize separators and punctuation for exact and typo matching."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _resolve_story_directory(value: str | Path, directories: list[Path]) -> Path:
    """Resolve an exact story slug, with a guarded fallback for unique typos."""
    requested = str(value).strip()
    requested_key = _story_key(requested)
    exact = [
        path
        for path in directories
        if path.name.casefold() == requested.casefold()
        or _story_key(path.name) == requested_key
    ]
    if len(exact) == 1:
        return exact[0]

    ranked = sorted(
        (
            (
                SequenceMatcher(
                    None, requested_key, _story_key(path.name)
                ).ratio(),
                path,
            )
            for path in directories
        ),
        key=lambda item: (-item[0], item[1].name.casefold()),
    )
    close = [
        (score, path)
        for score, path in ranked
        if score >= _STORY_MATCH_THRESHOLD
    ]

    if len(close) == 1:
        selected = close[0][1]
        warnings.warn(
            f"story {requested!r} was not found exactly; "
            f"resolved to {selected.name!r}",
            UserWarning,
            stacklevel=2,
        )
        return selected
    if (
        len(close) > 1
        and close[0][0] - close[1][0] >= _STORY_MATCH_MARGIN
    ):
        selected = close[0][1]
        warnings.warn(
            f"story {requested!r} was not found exactly; "
            f"resolved to {selected.name!r}",
            UserWarning,
            stacklevel=2,
        )
        return selected

    choices = ", ".join(sorted(path.name for path in directories))
    suggestions = ", ".join(
        f"{path.name!r} ({score:.2f})"
        for score, path in ranked[:3]
        if score > 0
    )
    suggestion_text = (
        f"; close matches: {suggestions}" if suggestions else ""
    )
    raise ValueError(
        f"could not uniquely resolve story {requested!r}"
        f"{suggestion_text}; available stories: {choices}"
    )


def slugify(value: str) -> str:
    slug = _SAFE_SLUG_RE.sub("-", value.strip()).strip(".-")
    return slug[:100] or "story"


def resolve_arcane_story(value: str | Path) -> dict[str, str]:
    """Resolve a story name or axes-file path.

    A story name is matched against ``results/arc_extraction/<story>``.
    """
    raw = Path(value).expanduser()
    if raw.is_file():
        source = raw.resolve()
        story_slug = slugify(
            source.parent.parent.name
            if source.parent.name in {"final", "final_auto", "final_grounded"}
            else source.stem
        )
        return {
            "axes_path": str(source),
            "story_slug": story_slug,
            "title": story_slug.replace("-", " ").replace("_", " "),
        }

    project_root = Path(__file__).resolve().parents[2]
    results_root = project_root / "results" / "arc_extraction"
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"ArcANE extraction root not found: {results_root}"
        )
    dirs = [
        path
        for path in results_root.iterdir()
        if path.is_dir()
    ]
    story_dir = _resolve_story_directory(value, dirs)
    candidates = [
        story_dir / "final" / "all_characters_axes.json",
        story_dir / "final_auto" / "all_characters_auto.json",
        story_dir / "final_grounded" / "all_characters_grounded.json",
    ]
    for path in candidates:
        if path.is_file():
            return {
                "axes_path": str(path),
                "story_slug": slugify(story_dir.name),
                "title": story_dir.name.replace("_", " "),
            }
    raise FileNotFoundError(
        "no all-character ArcANE axes file found for "
        f"{story_dir.name}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def select_characters(
    payload: dict[str, Any],
    selection: str,
    *,
    input_fn=input,
) -> dict[str, Any]:
    """Filter an adapted payload by all, exact names, or interactive selection."""
    characters = payload.get("characters", [])
    if not characters:
        raise ValueError("ArcANE axes payload has no characters")
    choice = (selection or "all").strip()
    if choice.casefold() in {"all", "*"}:
        return payload

    if choice.casefold() in {"select", "selected"}:
        print("Available characters:")
        for index, character in enumerate(characters, start=1):
            print(f"  {index}. {character['name']}")
        raw = input_fn(
            "Select character numbers (comma-separated; empty = cancel): "
        ).strip()
        if not raw:
            raise ValueError("character selection cancelled")
        indices: set[int] = set()
        for token in raw.split(","):
            token = token.strip()
            if "-" in token:
                start, end = token.split("-", 1)
                indices.update(range(int(start), int(end) + 1))
            elif token:
                indices.add(int(token))
        selected = [
            character
            for index, character in enumerate(characters, start=1)
            if index in indices
        ]
        if not selected:
            raise ValueError("no valid character number selected")
        return {**payload, "characters": selected}

    requested = [
        item.strip() for item in choice.split(",") if item.strip()
    ]
    selected = []
    unmatched = []
    for name in requested:
        exact = [
            character
            for character in characters
            if character["name"].casefold() == name.casefold()
        ]
        matches = exact or [
            character
            for character in characters
            if name.casefold() in character["name"].casefold()
        ]
        if len(matches) != 1:
            unmatched.append(name)
        elif matches[0] not in selected:
            selected.append(matches[0])
    if unmatched:
        available = ", ".join(
            character["name"] for character in characters
        )
        raise ValueError(
            f"character selection was missing or ambiguous: {unmatched}; "
            f"available characters: {available}"
        )
    return {**payload, "characters": selected}


def _axes_from_character(
    payload: dict[str, Any],
    fallback_name: str,
    *,
    max_axes: int,
) -> list[dict[str, Any]]:
    character = str(payload.get("character") or fallback_name).strip()
    axes: list[dict[str, Any]] = []
    for axis_type in ("intrapersonal_axes", "relational_axes"):
        raw_axes = payload.get(axis_type)
        if not isinstance(raw_axes, list):
            continue
        for raw_axis in raw_axes:
            if not isinstance(raw_axis, dict):
                continue
            phases = []
            raw_trajectory = raw_axis.get("trajectory")
            if isinstance(raw_trajectory, list):
                for phase_idx, raw_phase in enumerate(raw_trajectory):
                    if not isinstance(raw_phase, dict):
                        continue
                    key_moments = raw_phase.get("key_moments")
                    if not isinstance(key_moments, list):
                        key_moments = []
                    phase = {
                        "phase_idx": phase_idx,
                        "phase_label": str(
                            raw_phase.get("phase") or f"Phase {phase_idx + 1}"
                        ).strip(),
                        "chapter_range": raw_phase.get("chapter_range"),
                        "position": str(
                            raw_phase.get("position_description") or ""
                        ).strip(),
                        "key_moments": [
                            str(item).strip()
                            for item in key_moments
                            if str(item or "").strip()
                        ],
                    }
                    phases.append(phase)
            if len(phases) < 2:
                continue
            axis = {
                "axis_id": str(raw_axis.get("axis_id") or f"axis_{len(axes)+1:02d}"),
                "axis_type": (
                    "relational"
                    if axis_type == "relational_axes"
                    else "intrapersonal"
                ),
                "dimension_label": str(
                    raw_axis.get("dimension_label") or ""
                ).strip(),
                "axis_name": str(raw_axis.get("axis_name") or "").strip(),
                "source_character": str(
                    raw_axis.get("source_character") or character
                ).strip(),
                "target_character": str(
                    raw_axis.get("target_character") or ""
                ).strip(),
                "pole_start": str(raw_axis.get("pole_start") or "").strip(),
                "pole_end": str(raw_axis.get("pole_end") or "").strip(),
                "arc_direction": str(
                    raw_axis.get("arc_direction") or ""
                ).strip(),
                "confidence": str(raw_axis.get("confidence") or "").strip(),
                "evidence_summary": str(
                    raw_axis.get("evidence_summary") or ""
                ).strip(),
                "literary_validation_tag": (
                    (raw_axis.get("literary_validation") or {}).get(
                        "validation_tag"
                    )
                    if isinstance(raw_axis.get("literary_validation"), dict)
                    else None
                ),
                "phases": phases,
                "n_extracted_phases": len(phases),
                "scoreable": True,
            }
            axes.append(axis)
            if len(axes) >= max_axes:
                return axes
    return axes


def load_arcane_axes(
    path: str | Path,
    *,
    max_axes_per_character: int = 4,
    character_names: set[str] | None = None,
) -> dict[str, Any]:
    """Load one character file or an ``all_characters_*.json`` mapping."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ArcANE axes file must contain one JSON object")

    candidates: list[tuple[str, dict[str, Any]]] = []
    if any(
        key in payload
        for key in ("intrapersonal_axes", "relational_axes")
    ):
        candidates.append(
            (str(payload.get("character") or source.stem), payload)
        )
    else:
        for key, value in payload.items():
            if isinstance(value, dict) and any(
                axis_key in value
                for axis_key in ("intrapersonal_axes", "relational_axes")
            ):
                candidates.append((str(key), value))
    if not candidates:
        raise ValueError("no ArcANE intrapersonal_axes or relational_axes found")

    requested = (
        {name.casefold() for name in character_names}
        if character_names
        else None
    )
    characters: list[dict[str, Any]] = []
    for fallback_name, character_payload in candidates:
        character = str(
            character_payload.get("character") or fallback_name
        ).strip()
        if requested and character.casefold() not in requested:
            continue
        axes = _axes_from_character(
            character_payload,
            fallback_name,
            max_axes=max_axes_per_character,
        )
        characters.append(
            {
                "name": character,
                "candidate_axes": axes,
                "n_scoreable_axes": len(axes),
            }
        )
    if not characters:
        raise ValueError("no matching characters with at least two phases")
    return {
        "schema_version": "arcane_stage1_axes_v1",
        "source_path": str(source),
        "verification_scope": "arcane_extraction_only",
        "characters": characters,
    }
