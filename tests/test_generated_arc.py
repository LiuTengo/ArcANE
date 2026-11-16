"""Offline tests for narrative-intrinsic generated-story arc evaluation."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.generated_arc import config
from evaluation.generated_arc import runner
from evaluation.generated_arc.arcane_adapter import load_arcane_axes
from evaluation.generated_arc.loader import (
    ground_evidence,
    load_story,
    segment_story,
)
from evaluation.generated_arc.models import DIMENSIONS, StoryInput
from evaluation.generated_arc.runner import (
    aggregate_scores,
    judge_axis_view,
    normalize_candidate_axes,
    qualifies_extracted_candidate,
    qualifies_as_arc,
    score_extraction,
    score_arcane_axes,
    write_report,
)


STORY_TEXT = """Mira kept every promise, even when each one cost her a night of sleep.

"A promise is a debt," she told Tomas. She refused to leave the wounded traveller alone.

After the bridge collapsed, Mira realised that rigid promises could become another form of cruelty.

"I cannot save everyone by obeying an old rule," she said. She chose to break the oath and stay with the child.

At the end, Mira still valued commitment, but she treated care as the purpose of a promise rather than its wording.
"""


def make_story() -> StoryInput:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", encoding="utf-8", delete=False
    ) as handle:
        handle.write(STORY_TEXT)
        path = Path(handle.name)
    try:
        return load_story(
            path,
            story_id="mira-story",
            title="Mira",
            metadata={"generator": "system-a", "condition": "arc"},
        )
    finally:
        path.unlink(missing_ok=True)


def candidate_payload(quote: str) -> dict:
    return {
        "character": "Mira",
        "candidate_axes": [
            {
                "axis_type": "intrapersonal",
                "dimension_label": "Rigid duty to compassionate care",
                "axis_name": "From rule-bound duty to care-led commitment",
                "source_character": "Mira",
                "target_character": "",
                "behavioral_contrast": (
                    "Obey the wording of a promise or protect the person"
                ),
                "pole_start": "Promise wording is absolute",
                "pole_end": "Care is the purpose of a promise",
                "arc_kind": "positive",
                "phases": [
                    {
                        "phase_idx": 0,
                        "span_ids": ["P0001"],
                        "position": "Rules dominate",
                        "evidence": [
                            {
                                "span_ids": ["P0001"],
                                "quote": quote,
                                "kind": "narrator",
                                "note": "Establishes rigid commitment",
                            }
                        ],
                    },
                    {
                        "phase_idx": 1,
                        "span_ids": ["P0004"],
                        "position": "Care overrides the vow",
                        "evidence": [
                            {
                                "span_ids": ["P0004"],
                                "quote": (
                                    "She chose to break the oath and stay with the child."
                                ),
                                "kind": "action",
                                "note": "Shows the changed decision",
                            }
                        ],
                    },
                ],
            }
        ],
    }


class StoryLoaderTests(unittest.TestCase):
    def test_segmentation_preserves_order_and_ids(self):
        segments = segment_story(STORY_TEXT, target_chars=1000)
        self.assertEqual(
            [segment.span_id for segment in segments],
            [f"P{i:04d}" for i in range(1, len(segments) + 1)],
        )
        self.assertIn("Mira kept every promise", segments[0].text)

    def test_evidence_requires_verbatim_text_in_cited_span(self):
        story = make_story()
        valid = ground_evidence(
            story,
            span_ids=["P0001"],
            quote="Mira kept every promise, even when each one cost her",
        )
        wrong_span = ground_evidence(
            story,
            span_ids=["P0004"],
            quote="Mira kept every promise, even when each one cost her",
        )
        fabricated = ground_evidence(
            story,
            span_ids=["P0001"],
            quote="Mira never cared about promises at all.",
        )
        self.assertTrue(valid.valid)
        self.assertFalse(wrong_span.valid)
        self.assertFalse(fabricated.valid)


class CandidateExtractionTests(unittest.TestCase):
    def test_grounded_two_phase_axis_is_scoreable(self):
        story = make_story()
        candidate = normalize_candidate_axes(
            candidate_payload(
                "Mira kept every promise, even when each one cost her a night of sleep."
            ),
            story,
            "Mira",
        )
        axis = candidate["candidate_axes"][0]
        self.assertTrue(axis["scoreable"])
        self.assertEqual(axis["n_grounded_phases"], 2)
        self.assertEqual(axis["evidence_grounding_rate"], 1.0)

    def test_fabricated_evidence_is_removed_from_scoring_basis(self):
        story = make_story()
        payload = candidate_payload("Mira never cared about promises at all.")
        payload["candidate_axes"][0]["phases"][1]["evidence"] = []
        candidate = normalize_candidate_axes(payload, story, "Mira")
        axis = candidate["candidate_axes"][0]
        self.assertFalse(axis["scoreable"])
        self.assertIsNotNone(axis["scoreability_error"])

        view = judge_axis_view(axis)
        visible_quotes = [
            item["quote"]
            for phase in view["phases"]
            for item in phase["evidence"]
        ]
        self.assertNotIn("Mira never cared about promises at all.", visible_quotes)
        self.assertTrue(
            all("invalid_evidence" not in phase for phase in view["phases"])
        )


class PresenceAndAggregationTests(unittest.TestCase):
    def test_judge_average_is_recomputed_from_dimensions(self):
        verdict = runner._validate_judge(
            {
                "arc_judgment": "present",
                "change_type": "positive",
                "scores": {
                    "action_progression": 40,
                    "reasoning_progression": 50,
                    "reasoning_action_entailment": 60,
                    "trajectory_coherence": 70,
                },
                "average": 99,
            },
            100,
        )
        self.assertEqual(verdict["scores"]["average"], 55)

    def test_presence_rule_requires_label_score_and_evidence(self):
        verdict = {
            "arc_judgment": "present",
            "scores": {
                **{dim: 80 for dim in DIMENSIONS},
                "average": 80,
            },
        }
        self.assertTrue(
            qualifies_as_arc(
                verdict,
                evidence_grounding_rate=1.0,
                n_grounded_phases=3,
                scale=100,
            )
        )
        self.assertTrue(
            qualifies_extracted_candidate(
                verdict,
                n_phases=3,
                scale=100,
            )
        )
        low = {
            **verdict,
            "scores": {
                **verdict["scores"],
                "trajectory_coherence": 20,
            },
        }
        self.assertFalse(
            qualifies_as_arc(
                low,
                evidence_grounding_rate=1.0,
                n_grounded_phases=3,
                scale=100,
            )
        )
        self.assertFalse(
            qualifies_as_arc(
                {**verdict, "arc_judgment": "absent"},
                evidence_grounding_rate=1.0,
                n_grounded_phases=3,
                scale=100,
            )
        )

    def test_aggregation_uses_evaluable_character_denominator(self):
        story = make_story()
        extraction = {
            "characters": [
                {"name": "Mira", "candidate_axes": []},
                {"name": "Tomas", "candidate_axes": []},
                {"name": "Unscored", "candidate_axes": []},
            ]
        }
        records = [
            {
                "character": "Mira",
                "axis_id": "axis-1",
                "dimension_label": "Duty and care",
                "scale": 100,
                "has_arc": True,
                "average": 82,
                "scores": {dim: 82 for dim in DIMENSIONS},
                "parse_ok": True,
            },
            {
                "character": "Mira",
                "axis_id": "axis-2",
                "dimension_label": "Trust",
                "scale": 100,
                "has_arc": False,
                "average": 61,
                "scores": {dim: 61 for dim in DIMENSIONS},
                "parse_ok": True,
            },
            {
                "character": "Tomas",
                "axis_id": "axis-3",
                "dimension_label": "Caution",
                "scale": 100,
                "has_arc": False,
                "average": 40,
                "scores": {dim: 40 for dim in DIMENSIONS},
                "parse_ok": True,
            },
        ]
        summary = aggregate_scores(
            story, extraction, records, model="judge", scale=100
        )
        self.assertTrue(summary["story_has_arc"])
        self.assertEqual(summary["n_characters_with_arc"], 1)
        self.assertEqual(summary["n_evaluable_characters"], 2)
        self.assertEqual(summary["character_arc_rate"], 0.5)
        self.assertEqual(summary["mean_best_axis_score"], 61.0)
        self.assertEqual(
            summary["by_character"]["Unscored"]["status"],
            "no_scoreable_grounded_axis",
        )

    def test_report_groups_completed_summaries(self):
        with tempfile.TemporaryDirectory(prefix="genarc-report-") as temp:
            root = Path(temp)
            for index, generator in enumerate(("a", "a", "b"), start=1):
                directory = root / f"story-{index}"
                directory.mkdir()
                summary = {
                    "generator": generator,
                    "condition": "test",
                    "story_has_arc": index != 2,
                    "n_evaluable_characters": 2,
                    "n_characters_with_arc": 1 if index != 2 else 0,
                    "n_axes_scored": 2,
                    "n_axes_with_arc": 1 if index != 2 else 0,
                    "mean_best_axis_score": 70 if index != 2 else 50,
                }
                (directory / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
            report = write_report(root)
            groups = {
                (row["generator"], row["condition"]): row
                for row in report["groups"]
            }
            self.assertEqual(groups[("a", "test")]["n_stories"], 2)
            self.assertEqual(groups[("a", "test")]["n_characters_with_arc"], 1)
            self.assertEqual(groups[("a", "test")]["character_arc_rate"], 0.25)
            self.assertEqual(groups[("b", "test")]["n_stories"], 1)

    def test_judge_failure_preserves_prior_scoring_outputs(self):
        story = make_story()
        candidate = normalize_candidate_axes(
            candidate_payload(
                "Mira kept every promise, even when each one cost her a night of sleep."
            ),
            story,
            "Mira",
        )
        extraction = {
            "story": {
                "sha256": hashlib.sha256(story.text.encode("utf-8")).hexdigest()
            },
            "characters": [
                {
                    "name": "Mira",
                    "candidate_axes": candidate["candidate_axes"],
                }
            ],
        }
        with tempfile.TemporaryDirectory(prefix="genarc-atomic-") as temp:
            output = Path(temp)
            scores = output / "scores.jsonl"
            summary = output / "summary.json"
            scores.write_text("previous scores\n", encoding="utf-8")
            summary.write_text("previous summary\n", encoding="utf-8")
            error_record = {
                "story_id": story.story_id,
                "character": "Mira",
                "axis_id": "r1_axis_01",
                "scale": 100,
                "parse_ok": False,
                "scores": None,
                "error": "judge unavailable",
            }
            with patch.object(
                runner,
                "_score_one_axis",
                return_value=(error_record, {}),
            ):
                with self.assertRaises(runner.EvaluationRunError):
                    score_extraction(
                        story,
                        extraction,
                        output,
                        model="judge",
                        client=Mock(),
                    )
            self.assertEqual(
                scores.read_text(encoding="utf-8"), "previous scores\n"
            )
            self.assertEqual(
                summary.read_text(encoding="utf-8"), "previous summary\n"
            )
            self.assertTrue(list(output.glob("scores.jsonl.*.partial")))

    def test_no_scoreable_axis_clears_stale_score_file(self):
        story = make_story()
        extraction = {
            "story": {
                "sha256": hashlib.sha256(story.text.encode("utf-8")).hexdigest()
            },
            "characters": [
                {
                    "name": "Mira",
                    "candidate_axes": [
                        {
                            "axis_id": "axis-1",
                            "scoreable": False,
                            "evidence_grounding_rate": 0,
                            "n_grounded_phases": 1,
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory(prefix="genarc-empty-") as temp:
            output = Path(temp)
            (output / "scores.jsonl").write_text("stale\n", encoding="utf-8")
            with patch.object(config, "SCORE_SCALE", 100):
                summary = score_extraction(
                    story,
                    extraction,
                    output,
                    model="judge",
                    client=Mock(),
                )
            self.assertEqual(
                (output / "scores.jsonl").read_text(encoding="utf-8"), ""
            )
            self.assertEqual(summary["n_evaluable_characters"], 0)
            self.assertIsNone(summary["character_arc_rate"])

    def test_existing_arcane_axes_can_be_scored_without_source_text(self):
        payload = {
            "character": "Mira",
            "intrapersonal_axes": [
                {
                    "axis_id": "final_intra_01",
                    "dimension_label": "duty and care",
                    "axis_name": "From rigid duty to compassionate care",
                    "pole_start": "Rules are absolute",
                    "pole_end": "Care guides the rule",
                    "trajectory": [
                        {
                            "phase": "Rule-bound",
                            "chapter_range": [1, 2],
                            "position_description": "Mira follows every rule.",
                            "key_moments": ["She keeps an impossible promise."],
                        },
                        {
                            "phase": "Care-led",
                            "chapter_range": [3, 4],
                            "position_description": "Mira protects the child.",
                            "key_moments": ["She breaks the oath to help."],
                        },
                    ],
                }
            ],
            "relational_axes": [],
        }
        with tempfile.TemporaryDirectory(prefix="genarc-arcane-") as temp:
            root = Path(temp)
            axes_path = root / "mira_final_axes.json"
            axes_path.write_text(json.dumps(payload), encoding="utf-8")
            adapted = load_arcane_axes(axes_path, max_axes_per_character=2)
            self.assertEqual(
                adapted["verification_scope"], "arcane_extraction_only"
            )
            axis = adapted["characters"][0]["candidate_axes"][0]
            record = {
                "story_id": "mira",
                "character": "Mira",
                "axis_id": axis["axis_id"],
                "axis_type": "intrapersonal",
                "dimension_label": axis["dimension_label"],
                "scale": 100,
                "has_arc": True,
                "average": 80,
                "scores": {dim: 80 for dim in DIMENSIONS},
                "parse_ok": True,
                "error": None,
            }
            with patch.object(
                runner,
                "_score_one_arcane_axis",
                return_value=(record, {}),
            ), patch.object(config, "SCORE_SCALE", 100):
                summary = score_arcane_axes(
                    adapted,
                    root / "output",
                    story_id="mira",
                    title="Mira",
                    model="judge",
                    client=Mock(),
                )
            self.assertEqual(
                summary["verification_scope"],
                "arcane_extraction_only",
            )
            self.assertTrue(summary["candidate_has_arc"])
            self.assertIsNone(summary["mean_candidate_evidence_grounding"])
            self.assertTrue((root / "output" / "scores.jsonl").is_file())
            self.assertTrue((root / "output" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
