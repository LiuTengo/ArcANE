"""Run extraction, judge generated-story arcs, and aggregate the results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from evaluation.scoring import (
    EvaluationRunError,
    scores_with_average,
    validate_scale,
)

from . import config
from .format_compat import effective_response_format, note_rejection
from .loader import (
    ground_evidence,
    normalize_text,
    render_story_block,
)
from .models import (
    ARC_LABELS,
    CHANGE_TYPES,
    DIMENSIONS,
    SCHEMA_VERSION,
    StoryInput,
)
from .prompts import (
    judge_response_format,
    render_arcane_extraction_judge_messages,
    render_extraction_messages,
    render_inventory_messages,
    render_judge_messages,
)


logger = logging.getLogger(__name__)


def extraction_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "extraction.json"


def scores_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "scores.jsonl"


def summary_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "summary.json"


def report_path(output_root: str | Path) -> Path:
    return Path(output_root) / "report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def _parse_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = "\n".join(
            line
            for line in raw.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("judge returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("judge must return one JSON object")
    return value


def _request_validated_json(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    validator: Callable[[dict[str, Any]], Any],
    temperature: float,
    response_format: Optional[dict[str, str]] = None,
    scale: Optional[int] = None,
) -> tuple[Any, dict[str, Any]]:
    last_error: Optional[Exception] = None
    current_temperature: Optional[float] = temperature
    for attempt in range(config.MAX_RETRIES):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            effective_format = (
                effective_response_format(model, response_format)
                if response_format is not None
                else None
            )
            if effective_format is not None:
                kwargs["response_format"] = effective_format
            if current_temperature is not None:
                kwargs["temperature"] = current_temperature
            if config.BACKEND == "openrouter":
                extra_body: dict[str, Any] = {}
                if config.PROVIDER_ORDER:
                    extra_body["provider"] = {
                        "order": config.PROVIDER_ORDER,
                        "allow_fallbacks": config.PROVIDER_ALLOW_FALLBACK,
                    }
                if config.DISABLE_REASONING:
                    extra_body["reasoning"] = {"enabled": False}
                if extra_body:
                    kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise RuntimeError(f"judge refused: {message.refusal}")
            value = _parse_object(message.content or "")
            result = validator(value)
            usage = response.usage.model_dump() if response.usage else {}
            return result, usage
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if (
                response_format is not None
                and note_rejection(model, response_format, exc)
            ):
                continue
            if (
                current_temperature is not None
                and "temperature" in str(exc).lower()
            ):
                current_temperature = None
                continue
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_BACKOFF_BASE ** (attempt + 1))
    raise RuntimeError(
        f"JSON judge failed after {config.MAX_RETRIES} attempts: {last_error}"
    )


def _text_story_tokens(story: StoryInput) -> int:
    return max(1, len(story.text) // 4)


def estimate_cost(
    story: StoryInput,
    *,
    n_axes: int = 0,
    model: str = config.MODEL,
) -> dict[str, Any]:
    """Heuristic pre-run estimate for one story.

    Inventory and arc extraction each see the story once. Judging repeats the
    story once per candidate axis, which is the dominant cost.
    """
    story_tokens = _text_story_tokens(story)
    inventory_calls = 1
    extraction_calls = max(1, config.MAX_CHARACTERS) * config.EXTRACTION_RUNS
    judge_calls = n_axes
    prompt_overhead = 1200
    input_tokens = (
        (inventory_calls + extraction_calls + judge_calls)
        * (story_tokens + prompt_overhead)
    )
    output_tokens = (
        inventory_calls * 300
        + extraction_calls * config.EST_OUTPUT_TOKENS_PER_EXTRACTION
        + judge_calls * config.EST_OUTPUT_TOKENS_PER_SCORE
    )
    cost = (
        input_tokens / 1_000_000 * config.PRICE_INPUT
        + output_tokens / 1_000_000 * config.PRICE_OUTPUT
    )
    return {
        "model": model,
        "inventory_calls_upper_bound": inventory_calls,
        "arc_extraction_calls_upper_bound": extraction_calls,
        "judge_calls": judge_calls,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 4),
    }


def _normalize_inventory(
    raw: dict[str, Any], story: StoryInput
) -> dict[str, Any]:
    entries = raw.get("characters")
    if not isinstance(entries, list) or not entries:
        raise ValueError("inventory needs a non-empty characters array")

    story_norm = normalize_text(story.text)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    warnings: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        raw_aliases = entry.get("aliases")
        aliases = (
            [
                str(alias).strip()
                for alias in raw_aliases
                if str(alias or "").strip()
            ]
            if isinstance(raw_aliases, list)
            else []
        )
        forms = [name, *aliases]
        if not any(normalize_text(form) in story_norm for form in forms):
            warnings.append(f"dropped character absent from story text: {name!r}")
            continue
        key = normalize_text(name).casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "name": name,
                "aliases": aliases,
                "role": str(entry.get("role") or "").strip()[:500],
                "selection_reason": str(
                    entry.get("selection_reason") or ""
                ).strip()[:1000],
                "first_span": str(entry.get("first_span") or "")[:20],
                "last_span": str(entry.get("last_span") or "")[:20],
            }
        )
        if len(selected) >= config.MAX_CHARACTERS:
            break
    if not selected:
        raise ValueError("inventory contained no character grounded in the story")
    return {"characters": selected, "warnings": warnings}


def _optional_text(value: object, limit: int = 1000) -> str:
    return str(value or "").strip()[:limit]


def normalize_candidate_axes(
    raw: dict[str, Any],
    story: StoryInput,
    character_name: str,
    *,
    run_index: int = 0,
) -> dict[str, Any]:
    """Ground an extractor reply and retain invalid axes as audit records."""
    raw_axes = raw.get("candidate_axes")
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ValueError("arc extraction needs a non-empty candidate_axes array")

    warnings: list[str] = []
    axes: list[dict[str, Any]] = []
    for axis_index, raw_axis in enumerate(raw_axes, start=1):
        if not isinstance(raw_axis, dict):
            continue
        axis_type = (
            "relational"
            if raw_axis.get("axis_type") == "relational"
            else "intrapersonal"
        )
        axis_id = f"r{run_index + 1}_axis_{axis_index:02d}"
        raw_phases = raw_axis.get("phases")
        phases: list[dict[str, Any]] = []
        total_evidence = 0
        valid_evidence = 0
        if isinstance(raw_phases, list):
            for raw_phase in raw_phases:
                if not isinstance(raw_phase, dict):
                    continue
                try:
                    phase_idx = int(raw_phase.get("phase_idx"))
                except (TypeError, ValueError):
                    continue
                raw_evidence = raw_phase.get("evidence")
                if not isinstance(raw_evidence, list):
                    raw_evidence = []
                checked: list[dict[str, Any]] = []
                for evidence in raw_evidence:
                    if not isinstance(evidence, dict):
                        continue
                    total_evidence += 1
                    check = ground_evidence(
                        story,
                        span_ids=evidence.get("span_ids"),
                        quote=evidence.get("quote"),
                        kind=evidence.get("kind"),
                        note=evidence.get("note"),
                    )
                    checked.append(check.to_dict())
                    if check.valid:
                        valid_evidence += 1
                    else:
                        warnings.append(
                            f"{character_name}/{axis_id}: {check.error}"
                        )
                grounded = [item for item in checked if item["valid"]]
                if grounded:
                    phases.append(
                        {
                            "phase_idx": phase_idx,
                            "position": _optional_text(
                                raw_phase.get("position"), 1500
                            ),
                            "evidence": grounded,
                            "invalid_evidence": [
                                item for item in checked if not item["valid"]
                            ],
                        }
                    )

        phases.sort(key=lambda item: item["phase_idx"])
        distinct_phases = len({phase["phase_idx"] for phase in phases})
        scoreable = distinct_phases >= config.MIN_GROUNDED_PHASES
        evidence_rate = (
            round(valid_evidence / total_evidence, 4)
            if total_evidence
            else 0.0
        )
        axis = {
            "axis_id": axis_id,
            "extraction_run": run_index + 1,
            "primary": run_index == 0,
            "axis_type": axis_type,
            "dimension_label": _optional_text(
                raw_axis.get("dimension_label"), 160
            ),
            "axis_name": _optional_text(raw_axis.get("axis_name"), 500),
            "source_character": _optional_text(
                raw_axis.get("source_character") or character_name, 160
            ),
            "target_character": _optional_text(
                raw_axis.get("target_character"), 160
            ),
            "behavioral_contrast": _optional_text(
                raw_axis.get("behavioral_contrast"), 1000
            ),
            "pole_start": _optional_text(raw_axis.get("pole_start"), 1500),
            "pole_end": _optional_text(raw_axis.get("pole_end"), 1500),
            "arc_kind": (
                raw_axis.get("arc_kind")
                if raw_axis.get("arc_kind")
                in {"positive", "negative", "nonlinear", "stable"}
                else "stable"
            ),
            "phases": phases,
            "n_grounded_phases": distinct_phases,
            "evidence_grounding_rate": evidence_rate,
            "scoreable": scoreable,
            "scoreability_error": (
                None
                if scoreable
                else (
                    "needs at least "
                    f"{config.MIN_GROUNDED_PHASES} phases with grounded evidence"
                )
            ),
        }
        axes.append(axis)
        if len(axes) >= config.MAX_AXES_PER_CHARACTER:
            break

    if not axes:
        raise ValueError("no structurally usable candidate axis was returned")
    return {
        "character": character_name,
        "run_index": run_index,
        "candidate_axes": axes,
        "warnings": warnings,
    }


def prepare_extraction(
    story: StoryInput,
    *,
    model: str = config.MODEL,
    client: Any = None,
) -> dict[str, Any]:
    """Run information extraction and return a frozen arc-candidate dossier."""
    client = client or config.make_judge_client()
    block = render_story_block(story)
    usage_in = 0
    usage_out = 0

    inventory, usage = _request_validated_json(
        client,
        model,
        render_inventory_messages(
            block, max_characters=config.MAX_CHARACTERS
        ),
        validator=lambda value: _normalize_inventory(value, story),
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    usage_in += int(usage.get("prompt_tokens", 0) or 0)
    usage_out += int(usage.get("completion_tokens", 0) or 0)

    def extract_one(character: str, run_index: int) -> dict[str, Any]:
        normalized, call_usage = _request_validated_json(
            client,
            model,
            render_extraction_messages(
                block,
                character,
                max_axes=config.MAX_AXES_PER_CHARACTER,
            ),
            validator=lambda value: normalize_candidate_axes(
                value,
                story,
                character,
                run_index=run_index,
            ),
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return {"result": normalized, "usage": call_usage}

    jobs = [
        (entry["name"], run_index)
        for entry in inventory["characters"]
        for run_index in range(config.EXTRACTION_RUNS)
    ]
    workers = max(1, min(config.CONCURRENCY, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(extract_one, character, run_index)
            for character, run_index in jobs
        ]
        extracted = [future.result() for future in futures]

    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    warnings = list(inventory["warnings"])
    for item in extracted:
        result = item["result"]
        by_character[result["character"]].append(result)
        warnings.extend(result["warnings"])
        usage = item["usage"]
        usage_in += int(usage.get("prompt_tokens", 0) or 0)
        usage_out += int(usage.get("completion_tokens", 0) or 0)

    characters = []
    for entry in inventory["characters"]:
        runs = sorted(
            by_character[entry["name"]],
            key=lambda item: item["run_index"],
        )
        axes = [
            axis for run in runs for axis in run["candidate_axes"]
        ]
        characters.append(
            {
                **entry,
                "candidate_axes": axes,
                "n_scoreable_axes": sum(
                    1 for axis in axes if axis["scoreable"]
                ),
                "n_scoreable_primary_axes": sum(
                    1
                    for axis in axes
                    if axis["scoreable"] and axis.get("primary", True)
                ),
            }
        )

    digest = hashlib.sha256(story.text.encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "story": {
            **story.to_dict(),
            "sha256": digest,
        },
        "extraction_config": {
            "model": model,
            "max_characters": config.MAX_CHARACTERS,
            "max_axes_per_character": config.MAX_AXES_PER_CHARACTER,
            "extraction_runs": config.EXTRACTION_RUNS,
            "min_grounded_phases": config.MIN_GROUNDED_PHASES,
        },
        "characters": characters,
        "warnings": warnings,
        "this_run": {
            "input_tokens": usage_in,
            "output_tokens": usage_out,
            "cost_usd": round(
                usage_in / 1_000_000 * config.PRICE_INPUT
                + usage_out / 1_000_000 * config.PRICE_OUTPUT,
                4,
            ),
        },
    }


def write_extraction(
    story: StoryInput,
    output_dir: str | Path,
    extraction: dict[str, Any],
) -> Path:
    path = extraction_path(output_dir)
    _atomic_write(
        path,
        json.dumps(extraction, ensure_ascii=False, indent=2) + "\n",
    )
    return path


def load_extraction(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("extraction file must contain one JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported extraction schema: {value.get('schema_version')!r}"
        )
    return value


def _validate_judge(
    raw: dict[str, Any], scale: int
) -> dict[str, Any]:
    label = str(raw.get("arc_judgment") or "").strip().lower()
    if label not in ARC_LABELS:
        raise ValueError("invalid arc_judgment")
    change_type = str(raw.get("change_type") or "").strip().lower()
    if change_type not in CHANGE_TYPES:
        raise ValueError("invalid change_type")
    scores = scores_with_average(raw, DIMENSIONS, scale)
    if scores is None:
        raise ValueError(
            f"judge scores must be finite values from 1 to {scale}"
        )
    return {
        "arc_judgment": label,
        "change_type": change_type,
        "scores": scores,
        "supporting_evidence": (
            raw.get("supporting_evidence")
            if isinstance(raw.get("supporting_evidence"), list)
            else []
        ),
        "counterevidence": (
            raw.get("counterevidence")
            if isinstance(raw.get("counterevidence"), list)
            else []
        ),
        "rationale": _optional_text(raw.get("rationale"), 3000),
    }


def qualifies_as_arc(
    verdict: dict[str, Any],
    *,
    evidence_grounding_rate: float,
    n_grounded_phases: int,
    scale: int,
) -> bool:
    """Apply the pre-declared operational presence rule."""
    if verdict.get("arc_judgment") != "present":
        return False
    if n_grounded_phases < config.MIN_GROUNDED_PHASES:
        return False
    if evidence_grounding_rate < config.MIN_EVIDENCE_GROUNDING:
        return False
    scores = verdict.get("scores") or {}
    average = float(
        scores.get("average", verdict.get("average") or 0) or 0
    )
    if average < config.ARC_PRESENCE_MIN_SCORE:
        return False
    if any(
        float(scores.get(dim) or 0) < config.ARC_PRESENCE_MIN_DIMENSION
        for dim in DIMENSIONS
    ):
        return False
    return True


def qualifies_extracted_candidate(
    verdict: dict[str, Any],
    *,
    n_phases: int,
    scale: int,
) -> bool:
    """Presence rule for source-free judging of an extracted dossier."""
    if verdict.get("arc_judgment") != "present" or n_phases < 2:
        return False
    scores = verdict.get("scores") or {}
    average = float(
        scores.get("average", verdict.get("average") or 0) or 0
    )
    if average < config.ARC_PRESENCE_MIN_SCORE:
        return False
    return all(
        float(scores.get(dim) or 0) >= config.ARC_PRESENCE_MIN_DIMENSION
        for dim in DIMENSIONS
    )


def judge_axis_view(axis: dict[str, Any]) -> dict[str, Any]:
    """Remove audit-only and ungrounded fields before sending to the judge."""
    phases = []
    for phase in axis.get("phases", []):
        evidence = [
            {
                "span_ids": item.get("span_ids", []),
                "quote": item.get("quote", ""),
                "kind": item.get("kind", "narrator"),
                "note": item.get("note", ""),
            }
            for item in phase.get("evidence", [])
            if item.get("valid")
        ]
        phases.append(
            {
                "phase_idx": phase.get("phase_idx"),
                "position": phase.get("position", ""),
                "evidence": evidence,
            }
        )
    return {
        "axis_type": axis.get("axis_type"),
        "dimension_label": axis.get("dimension_label"),
        "axis_name": axis.get("axis_name"),
        "source_character": axis.get("source_character"),
        "target_character": axis.get("target_character"),
        "behavioral_contrast": axis.get("behavioral_contrast"),
        "pole_start": axis.get("pole_start"),
        "pole_end": axis.get("pole_end"),
        "arc_kind": axis.get("arc_kind"),
        "phases": phases,
    }


def _score_one_axis(
    client: Any,
    model: str,
    scale: int,
    story: StoryInput,
    character: str,
    axis: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = render_judge_messages(
        render_story_block(story),
        character,
        judge_axis_view(axis),
        scale=scale,
    )
    try:
        verdict, usage = _request_validated_json(
            client,
            model,
            messages,
            validator=lambda value: _validate_judge(value, scale),
            temperature=0.0,
            response_format=judge_response_format(),
            scale=scale,
        )
        has_arc = qualifies_as_arc(
            verdict,
            evidence_grounding_rate=float(
                axis.get("evidence_grounding_rate") or 0
            ),
            n_grounded_phases=int(axis.get("n_grounded_phases") or 0),
            scale=scale,
        )
        record = {
            "story_id": story.story_id,
            "character": character,
            "axis_id": axis["axis_id"],
            "extraction_run": axis.get("extraction_run", 1),
            "axis_type": axis.get("axis_type"),
            "dimension_label": axis.get("dimension_label"),
            "scale": scale,
            "n_grounded_phases": axis.get("n_grounded_phases"),
            "evidence_grounding_rate": axis.get(
                "evidence_grounding_rate"
            ),
            "arc_judgment": verdict["arc_judgment"],
            "change_type": verdict["change_type"],
            "has_arc": has_arc,
            "scores": {
                dim: verdict["scores"][dim] for dim in DIMENSIONS
            },
            "average": verdict["scores"]["average"],
            "supporting_evidence": verdict["supporting_evidence"],
            "counterevidence": verdict["counterevidence"],
            "rationale": verdict["rationale"],
            "parse_ok": True,
            "error": None,
        }
        return record, usage
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "generated-arc judge failed [%s/%s/%s]: %s",
            story.story_id,
            character,
            axis.get("axis_id"),
            exc,
        )
        return {
            "story_id": story.story_id,
            "character": character,
            "axis_id": axis.get("axis_id"),
            "extraction_run": axis.get("extraction_run", 1),
            "axis_type": axis.get("axis_type"),
            "dimension_label": axis.get("dimension_label"),
            "scale": scale,
            "n_grounded_phases": axis.get("n_grounded_phases"),
            "evidence_grounding_rate": axis.get(
                "evidence_grounding_rate"
            ),
            "arc_judgment": None,
            "change_type": None,
            "has_arc": False,
            "scores": None,
            "average": None,
            "supporting_evidence": [],
            "counterevidence": [],
            "rationale": "",
            "parse_ok": False,
            "error": str(exc),
        }, {}


def aggregate_scores(
    story: StoryInput,
    extraction: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    model: str,
    scale: int,
) -> dict[str, Any]:
    valid = [
        record
        for record in records
        if record.get("parse_ok")
        and record.get("scores")
        and record.get("scale") == scale
    ]
    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_character[record["character"]].append(record)

    character_rows: dict[str, dict[str, Any]] = {}
    for character in extraction.get("characters", []):
        name = character.get("name")
        rows = by_character.get(name, [])
        if not rows:
            character_rows[name] = {
                "n_scored_axes": 0,
                "n_arc_axes": 0,
                "has_arc": False,
                "best_axis_id": None,
                "best_axis_score": None,
                "status": "no_scoreable_grounded_axis",
            }
            continue
        best = max(rows, key=lambda row: float(row["average"]))
        character_rows[name] = {
            "n_scored_axes": len(rows),
            "n_arc_axes": sum(1 for row in rows if row.get("has_arc")),
            "has_arc": any(bool(row.get("has_arc")) for row in rows),
            "best_axis_id": best["axis_id"],
            "best_axis_score": best["average"],
            "best_axis_dimension": best.get("dimension_label"),
            "status": "evaluated",
        }

    evaluable = [
        row for row in character_rows.values() if row["n_scored_axes"] > 0
    ]
    with_arc = [row for row in evaluable if row["has_arc"]]
    all_axes = valid
    best_scores = [
        float(row["best_axis_score"])
        for row in evaluable
        if row.get("best_axis_score") is not None
    ]
    axis_scores = [float(row["average"]) for row in all_axes]
    dimensions = {
        dim: (
            round(
                sum(float(row["scores"][dim]) for row in all_axes)
                / len(all_axes),
                3,
            )
            if all_axes
            else None
        )
        for dim in DIMENSIONS
    }

    metadata = story.metadata
    candidate_axes = [
        axis
        for character in extraction.get("characters", [])
        for axis in character.get("candidate_axes", [])
    ]
    grounding_rates = [
        float(axis["evidence_grounding_rate"])
        for axis in candidate_axes
        if axis.get("evidence_grounding_rate") is not None
    ]
    invalid_evidence_items = sum(
        len(phase.get("invalid_evidence") or [])
        for axis in candidate_axes
        for phase in axis.get("phases", [])
    ) + sum(
        len(axis.get("invalid_evidence") or [])
        for axis in candidate_axes
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "story_id": story.story_id,
        "title": story.title,
        "generator": metadata.get("generator"),
        "condition": metadata.get("condition"),
        "control": metadata.get("control"),
        "prompt_id": metadata.get("prompt_id"),
        "seed": metadata.get("seed"),
        "judge_model": model,
        "score_scale": scale,
        "presence_rule": {
            "minimum_average": config.ARC_PRESENCE_MIN_SCORE,
            "minimum_dimension": config.ARC_PRESENCE_MIN_DIMENSION,
            "minimum_evidence_grounding": config.MIN_EVIDENCE_GROUNDING,
            "minimum_grounded_phases": config.MIN_GROUNDED_PHASES,
        },
        "story_has_arc": bool(with_arc),
        "n_declared_characters": len(character_rows),
        "n_evaluable_characters": len(evaluable),
        "n_characters_with_arc": len(with_arc),
        "character_arc_rate": (
            round(len(with_arc) / len(evaluable), 4)
            if evaluable
            else None
        ),
        "character_coverage": (
            round(len(evaluable) / len(character_rows), 4)
            if character_rows
            else 0.0
        ),
        "n_axes_scored": len(all_axes),
        "n_candidate_axes": len(candidate_axes),
        "axis_coverage": (
            round(len(all_axes) / len(candidate_axes), 4)
            if candidate_axes
            else 0.0
        ),
        "n_axes_with_arc": sum(
            1 for row in all_axes if row.get("has_arc")
        ),
        "axis_arc_rate": (
            round(
                sum(1 for row in all_axes if row.get("has_arc"))
                / len(all_axes),
                4,
            )
            if all_axes
            else None
        ),
        "mean_axis_score": (
            round(sum(axis_scores) / len(axis_scores), 3)
            if axis_scores
            else None
        ),
        "mean_best_axis_score": (
            round(sum(best_scores) / len(best_scores), 3)
            if best_scores
            else None
        ),
        "mean_candidate_evidence_grounding": (
            round(sum(grounding_rates) / len(grounding_rates), 4)
            if grounding_rates
            else None
        ),
        "n_invalid_evidence_items": invalid_evidence_items,
        "dimension_means": dimensions,
        "by_character": character_rows,
    }


def score_extraction(
    story: StoryInput,
    extraction: dict[str, Any],
    output_dir: str | Path,
    *,
    model: str = config.MODEL,
    client: Any = None,
) -> dict[str, Any]:
    """Judge every grounded candidate axis and publish story-level scores."""
    scale = config.SCORE_SCALE
    validate_scale(scale)
    declared_hash = extraction.get("story", {}).get("sha256")
    actual_hash = hashlib.sha256(story.text.encode("utf-8")).hexdigest()
    if declared_hash and declared_hash != actual_hash:
        raise ValueError(
            "story text changed after extraction; rerun extraction before scoring"
        )

    jobs: list[tuple[str, dict[str, Any]]] = []
    for character in extraction.get("characters", []):
        name = character.get("name")
        for axis in character.get("candidate_axes", []):
            if axis.get("scoreable") and axis.get("primary", True):
                jobs.append((name, axis))

    client = client or config.make_judge_client()
    usage_in = usage_out = 0
    partial_path: Optional[Path] = None
    rec_path = scores_path(output_dir)
    rec_path.parent.mkdir(parents=True, exist_ok=True)

    if jobs:
        workers = max(1, min(config.CONCURRENCY, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _score_one_axis,
                    client,
                    model,
                    scale,
                    story,
                    character,
                    axis,
                )
                for character, axis in jobs
            ]
            results = [future.result() for future in futures]

        records = [record for record, _usage in results]
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=rec_path.parent,
            prefix=rec_path.name + ".",
            suffix=".partial",
            delete=False,
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            partial_path = Path(handle.name)
        for _record, usage in results:
            usage_in += int(usage.get("prompt_tokens", 0) or 0)
            usage_out += int(usage.get("completion_tokens", 0) or 0)

        bad = sum(1 for record in records if not record.get("parse_ok"))
        if bad:
            raise EvaluationRunError(
                f"generated-arc: {bad}/{len(records)} judge calls failed; "
                f"previous outputs unchanged. Attempt records: {partial_path}"
            )
        partial_path.replace(rec_path)
    else:
        records = []
        _atomic_write(rec_path, "")

    summary = aggregate_scores(
        story,
        extraction,
        records,
        model=model,
        scale=scale,
    )
    summary["this_run"] = {
        "axes_scored": len(records),
        "input_tokens": usage_in,
        "output_tokens": usage_out,
        "cost_usd": round(
            usage_in / 1_000_000 * config.PRICE_INPUT
            + usage_out / 1_000_000 * config.PRICE_OUTPUT,
            4,
        ),
    }
    _atomic_write(
        summary_path(output_dir),
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def _score_one_arcane_axis(
    client: Any,
    model: str,
    scale: int,
    story_id: str,
    character: str,
    axis: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = render_arcane_extraction_judge_messages(
        character,
        axis,
        scale=scale,
    )
    try:
        verdict, usage = _request_validated_json(
            client,
            model,
            messages,
            validator=lambda value: _validate_judge(value, scale),
            temperature=0.0,
            response_format=judge_response_format(),
            scale=scale,
        )
        has_arc = qualifies_extracted_candidate(
            verdict,
            n_phases=int(axis.get("n_extracted_phases") or 0),
            scale=scale,
        )
        return {
            "story_id": story_id,
            "character": character,
            "axis_id": axis["axis_id"],
            "extraction_run": 1,
            "axis_type": axis.get("axis_type"),
            "dimension_label": axis.get("dimension_label"),
            "scale": scale,
            "n_extracted_phases": axis.get("n_extracted_phases"),
            "evidence_grounding_rate": None,
            "arc_judgment": verdict["arc_judgment"],
            "change_type": verdict["change_type"],
            "has_arc": has_arc,
            "scores": {
                dim: verdict["scores"][dim] for dim in DIMENSIONS
            },
            "average": verdict["scores"]["average"],
            "supporting_evidence": verdict["supporting_evidence"],
            "counterevidence": verdict["counterevidence"],
            "rationale": verdict["rationale"],
            "verification_scope": "arcane_extraction_only",
            "parse_ok": True,
            "error": None,
        }, usage
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ArcANE extraction-only judge failed [%s/%s/%s]: %s",
            story_id,
            character,
            axis.get("axis_id"),
            exc,
        )
        return {
            "story_id": story_id,
            "character": character,
            "axis_id": axis.get("axis_id"),
            "extraction_run": 1,
            "axis_type": axis.get("axis_type"),
            "dimension_label": axis.get("dimension_label"),
            "scale": scale,
            "n_extracted_phases": axis.get("n_extracted_phases"),
            "evidence_grounding_rate": None,
            "arc_judgment": None,
            "change_type": None,
            "has_arc": False,
            "scores": None,
            "average": None,
            "supporting_evidence": [],
            "counterevidence": [],
            "rationale": "",
            "verification_scope": "arcane_extraction_only",
            "parse_ok": False,
            "error": str(exc),
        }, {}


def score_arcane_axes(
    axes_payload: dict[str, Any],
    output_dir: str | Path,
    *,
    story_id: str,
    title: str,
    model: str = config.MODEL,
    client: Any = None,
) -> dict[str, Any]:
    """Score ArcANE Stage 1 axes without sending the source narrative."""
    scale = config.SCORE_SCALE
    validate_scale(scale)
    jobs = [
        (character["name"], axis)
        for character in axes_payload.get("characters", [])
        for axis in character.get("candidate_axes", [])
        if axis.get("scoreable")
    ]
    if not jobs:
        raise ValueError("no ArcANE axis with at least two phases was found")

    client = client or config.make_judge_client()
    workers = max(1, min(config.CONCURRENCY, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _score_one_arcane_axis,
                client,
                model,
                scale,
                story_id,
                character,
                axis,
            )
            for character, axis in jobs
        ]
        results = [future.result() for future in futures]

    records = [record for record, _usage in results]
    rec_path = scores_path(output_dir)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=rec_path.parent,
        prefix=rec_path.name + ".",
        suffix=".partial",
        delete=False,
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        partial_path = Path(handle.name)
    bad = sum(1 for record in records if not record.get("parse_ok"))
    if bad:
        raise EvaluationRunError(
            f"ArcANE extraction-only judge: {bad}/{len(records)} calls "
            f"failed; previous outputs unchanged. Attempt records: {partial_path}"
        )
    partial_path.replace(rec_path)

    pseudo_story = StoryInput(
        story_id=story_id,
        title=title,
        text="",
        source_path=Path(str(axes_payload.get("source_path") or "")),
        metadata={
            "generator": "arcane_stage1_axes",
            "condition": "extraction_only",
            "control": "not_source_verified",
        },
    )
    summary = aggregate_scores(
        pseudo_story,
        axes_payload,
        records,
        model=model,
        scale=scale,
    )
    usage_in = sum(
        int(usage.get("prompt_tokens", 0) or 0)
        for _record, usage in results
    )
    usage_out = sum(
        int(usage.get("completion_tokens", 0) or 0)
        for _record, usage in results
    )
    summary["verification_scope"] = "arcane_extraction_only"
    summary["source_axes_path"] = axes_payload.get("source_path")
    summary["candidate_has_arc"] = summary["story_has_arc"]
    summary["presence_rule"] = {
        "minimum_average": config.ARC_PRESENCE_MIN_SCORE,
        "minimum_dimension": config.ARC_PRESENCE_MIN_DIMENSION,
        "minimum_phases": 2,
    }
    summary["this_run"] = {
        "axes_scored": len(records),
        "input_tokens": usage_in,
        "output_tokens": usage_out,
        "cost_usd": round(
            usage_in / 1_000_000 * config.PRICE_INPUT
            + usage_out / 1_000_000 * config.PRICE_OUTPUT,
            4,
        ),
    }
    _atomic_write(
        summary_path(output_dir),
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def evaluate_story(
    story: StoryInput,
    output_dir: str | Path,
    *,
    model: str = config.MODEL,
    client: Any = None,
) -> dict[str, Any]:
    """Extract arc candidates, then score them without probes or references."""
    client = client or config.make_judge_client()
    extraction = prepare_extraction(story, model=model, client=client)
    write_extraction(story, output_dir, extraction)
    return score_extraction(
        story,
        extraction,
        output_dir,
        model=model,
        client=client,
    )


def write_report(output_root: str | Path) -> dict[str, Any]:
    """Aggregate completed per-story summaries by generator and condition."""
    root = Path(output_root)
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            summaries.append(value)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        generator = str(summary.get("generator") or "unspecified")
        condition = str(summary.get("condition") or "unspecified")
        grouped[(generator, condition)].append(summary)

    groups = []
    for (generator, condition), stories in sorted(grouped.items()):
        evaluable_characters = sum(
            int(story.get("n_evaluable_characters") or 0)
            for story in stories
        )
        characters_with_arc = sum(
            int(story.get("n_characters_with_arc") or 0)
            for story in stories
        )
        axes_scored = sum(
            int(story.get("n_axes_scored") or 0) for story in stories
        )
        axes_with_arc = sum(
            int(story.get("n_axes_with_arc") or 0) for story in stories
        )
        best_scores = [
            float(story["mean_best_axis_score"])
            for story in stories
            if story.get("mean_best_axis_score") is not None
        ]
        dimension_means = {}
        for dim in DIMENSIONS:
            weighted = 0.0
            weight = 0
            for story in stories:
                value = (story.get("dimension_means") or {}).get(dim)
                count = int(story.get("n_axes_scored") or 0)
                if value is not None and count:
                    weighted += float(value) * count
                    weight += count
            dimension_means[dim] = (
                round(weighted / weight, 3) if weight else None
            )
        groups.append(
            {
                "generator": generator,
                "condition": condition,
                "n_stories": len(stories),
                "n_evaluable_characters": evaluable_characters,
                "n_characters_with_arc": characters_with_arc,
                "character_arc_rate": (
                    round(characters_with_arc / evaluable_characters, 4)
                    if evaluable_characters
                    else None
                ),
                "story_arc_rate": round(
                    sum(
                        1
                        for story in stories
                        if story.get("story_has_arc")
                    )
                    / len(stories),
                    4,
                ),
                "n_axes_scored": axes_scored,
                "n_axes_with_arc": axes_with_arc,
                "axis_arc_rate": (
                    round(axes_with_arc / axes_scored, 4)
                    if axes_scored
                    else None
                ),
                "mean_best_axis_score": (
                    round(sum(best_scores) / len(best_scores), 3)
                    if best_scores
                    else None
                ),
                "dimension_means": dimension_means,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "n_story_summaries": len(summaries),
        "groups": groups,
    }
    _atomic_write(
        report_path(root),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(
        f"GENERATED-ARC  {summary['story_id']} / {summary['title']}   "
        f"(judge: {summary['judge_model']}, "
        f"scale: 1-{summary['score_scale']})"
    )
    print("=" * 78)
    print(
        "characters with arc: "
        f"{summary['n_characters_with_arc']}/"
        f"{summary['n_evaluable_characters']} evaluable "
        f"(coverage {summary['character_coverage']:.1%})"
    )
    print(
        "axes with arc: "
        f"{summary['n_axes_with_arc']}/{summary['n_axes_scored']}   "
        f"mean best-axis score: {summary['mean_best_axis_score']}"
    )
    if summary.get("dimension_means"):
        print("dimension means:")
        for dim, value in summary["dimension_means"].items():
            print(f"  {dim:32} {value}")
    print("=" * 78)
