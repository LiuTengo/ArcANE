"""Command line interface for narrative-intrinsic character-arc evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from evaluation.scoring import EvaluationRunError

from . import config
from .arcane_adapter import (
    load_arcane_axes,
    resolve_arcane_story,
    select_characters,
    slugify,
)
from .loader import load_manifest, load_story
from .runner import (
    estimate_cost,
    evaluate_story,
    load_extraction,
    prepare_extraction,
    print_summary,
    score_extraction,
    score_arcane_axes,
    write_extraction,
    write_report,
)


logger = logging.getLogger(__name__)


def _add_story_inputs(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--story", type=Path, help="one UTF-8 story text file")
    source.add_argument(
        "--manifest",
        type=Path,
        help="JSON array/object or JSONL list of generated stories",
    )
    parser.add_argument("--story-id")
    parser.add_argument("--title")
    parser.add_argument("--generator")
    parser.add_argument("--condition")
    parser.add_argument("--control")
    parser.add_argument("--prompt-id")
    parser.add_argument("--seed")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.DEFAULT_OUTPUT_ROOT,
        help=f"root for per-story outputs (default: {config.DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--model", default=config.MODEL, help=f"LLM model (default: {config.MODEL})"
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=config.MAX_CHARACTERS,
    )
    parser.add_argument(
        "--max-axes",
        type=int,
        default=config.MAX_AXES_PER_CHARACTER,
    )
    parser.add_argument(
        "--extraction-runs",
        type=int,
        default=config.EXTRACTION_RUNS,
        help="independent extraction passes used to reduce missed-axis variance",
    )
    parser.add_argument(
        "--score-scale",
        type=int,
        default=config.SCORE_SCALE,
    )
    parser.add_argument("--limit", type=int, help="limit stories for a smoke test")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="print an upper-bound API estimate and exit",
    )


def _apply_overrides(args: argparse.Namespace) -> None:
    if args.max_characters < 1:
        raise SystemExit("--max-characters must be positive")
    if args.max_axes < 1:
        raise SystemExit("--max-axes must be positive")
    if args.extraction_runs < 1:
        raise SystemExit("--extraction-runs must be positive")
    config.MAX_CHARACTERS = args.max_characters
    config.MAX_AXES_PER_CHARACTER = args.max_axes
    config.EXTRACTION_RUNS = args.extraction_runs
    config.SCORE_SCALE = args.score_scale


def _load_stories(args: argparse.Namespace):
    if args.manifest:
        stories = load_manifest(args.manifest)
    else:
        metadata = {
            key: getattr(args, key)
            for key in ("generator", "condition", "control", "prompt_id", "seed")
            if getattr(args, key) is not None
        }
        stories = [
            load_story(
                args.story,
                story_id=args.story_id,
                title=args.title,
                metadata=metadata,
            )
        ]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        stories = stories[: args.limit]
    if not stories:
        raise SystemExit("no stories loaded")
    return stories


def _warn_missing_key() -> None:
    if config.API_KEY != "EMPTY":
        return
    if config.BACKEND == "openrouter":
        env_name = "OPENROUTER_API_KEY"
    else:
        env_name = "OPENAI_API_KEY"
    logger.warning(
        "no GENARC_JUDGE_API_KEY or %s is set; "
        "the configured endpoint will reject live calls",
        env_name,
    )


def _run(args: argparse.Namespace) -> None:
    _apply_overrides(args)
    stories = _load_stories(args)
    if args.estimate_only:
        upper_axes = (
            args.max_characters
            * args.max_axes
            * args.extraction_runs
        )
        estimates = [
            {
                "story_id": story.story_id,
                **estimate_cost(story, n_axes=upper_axes, model=args.model),
            }
            for story in stories
        ]
        print(json.dumps(estimates, ensure_ascii=False, indent=2))
        return

    _warn_missing_key()
    client = config.make_judge_client()
    for story in stories:
        output_dir = args.output_root / story.story_id
        try:
            summary = evaluate_story(
                story,
                output_dir,
                model=args.model,
                client=client,
            )
        except EvaluationRunError as exc:
            raise SystemExit(str(exc)) from exc
        print_summary(summary)
        print(f"wrote {output_dir}")


def _extract(args: argparse.Namespace) -> None:
    _apply_overrides(args)
    stories = _load_stories(args)
    if args.estimate_only:
        upper_axes = (
            args.max_characters
            * args.max_axes
            * args.extraction_runs
        )
        print(
            json.dumps(
                [
                    {
                        "story_id": story.story_id,
                        **estimate_cost(
                            story, n_axes=upper_axes, model=args.model
                        ),
                    }
                    for story in stories
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    _warn_missing_key()
    client = config.make_judge_client()
    for story in stories:
        output_dir = args.output_root / story.story_id
        extraction = prepare_extraction(
            story, model=args.model, client=client
        )
        path = write_extraction(story, output_dir, extraction)
        scoreable = sum(
            int(character.get("n_scoreable_primary_axes") or 0)
            for character in extraction["characters"]
        )
        print(
            f"{story.story_id}: {len(extraction['characters'])} characters, "
            f"{scoreable} scoreable primary axes -> {path}"
        )


def _score(args: argparse.Namespace) -> None:
    config.SCORE_SCALE = args.score_scale
    story = load_story(
        args.story,
        story_id=args.story_id,
        title=args.title,
    )
    extraction = load_extraction(args.extraction)
    _warn_missing_key()
    client = config.make_judge_client()
    try:
        summary = score_extraction(
            story,
            extraction,
            args.output_dir,
            model=args.model,
            client=client,
        )
    except EvaluationRunError as exc:
        raise SystemExit(str(exc)) from exc
    print_summary(summary)
    print(f"wrote {args.output_dir}")


def _report(args: argparse.Namespace) -> None:
    report = write_report(args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.output_root / 'report.json'}")


def _score_arcane(args: argparse.Namespace) -> None:
    try:
        resolved = resolve_arcane_story(args.story)
        axes_payload = load_arcane_axes(
            resolved["axes_path"],
            max_axes_per_character=config.MAX_AXES_PER_CHARACTER,
        )
        axes_payload = select_characters(axes_payload, args.characters)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    names = [character["name"] for character in axes_payload["characters"]]
    if args.characters.strip().casefold() in {"all", "*"}:
        output_suffix = "all"
    else:
        output_suffix = "selected_" + slugify("-".join(names))
    story_id = f"{resolved['story_slug']}_arcane_{output_suffix}"
    title = resolved["title"]
    output_dir = config.DEFAULT_OUTPUT_ROOT / story_id
    _warn_missing_key()
    client = config.make_judge_client()
    try:
        summary = score_arcane_axes(
            axes_payload,
            output_dir,
            story_id=story_id,
            title=title,
            model=config.MODEL,
            client=client,
        )
    except EvaluationRunError as exc:
        raise SystemExit(str(exc)) from exc
    print_summary(summary)
    print(
        "scope: ArcANE extraction only; source-text faithfulness was not "
        "verified."
    )
    print(f"wrote {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.generated_arc",
        description=(
            "Evaluate whether characters in generated stories have "
            "evidence-grounded arcs. Existing ArcANE evaluation modules are "
            "not modified."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="extract candidate arcs, then judge them"
    )
    _add_story_inputs(run)
    _add_run_options(run)
    run.set_defaults(func=_run)

    extract = subparsers.add_parser(
        "extract", help="freeze character and arc extraction only"
    )
    _add_story_inputs(extract)
    _add_run_options(extract)
    extract.set_defaults(func=_extract)

    score = subparsers.add_parser(
        "score", help="judge a previously frozen extraction"
    )
    score.add_argument("--story", type=Path, required=True)
    score.add_argument("--story-id")
    score.add_argument("--title")
    score.add_argument("--extraction", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--model", default=config.MODEL)
    score.add_argument("--score-scale", type=int, default=config.SCORE_SCALE)
    score.set_defaults(func=_score)

    score_arcane = subparsers.add_parser(
        "score-arcane",
        help=(
            "score an existing ArcANE Stage-1 extraction without source text"
        ),
    )
    score_arcane.add_argument(
        "--story",
        required=True,
        help=(
            "story name under results/arc_extraction, or a path to an "
            "ArcANE axes JSON file"
        ),
    )
    score_arcane.add_argument(
        "--characters",
        default="all",
        help=(
            "all, selected (interactive), or comma-separated character names "
            "(default: all)"
        ),
    )
    score_arcane.set_defaults(func=_score_arcane)

    report = subparsers.add_parser(
        "report", help="aggregate completed story summaries"
    )
    report.add_argument(
        "--output-root",
        type=Path,
        default=config.DEFAULT_OUTPUT_ROOT,
    )
    report.set_defaults(func=_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args.func(args)
