# Generated-story character-arc evaluation

`evaluation.generated_arc` evaluates the stories emitted by a narrative
generation system. It does not modify ArcANE's existing probe, inference, or
role-play evaluation modules.

ArcANE asks whether an agent response fits a *known reference state* at a
probe phase. Here, there is no probe and no reference response. The unit of
judgment is therefore:

> Does this character undergo a meaningful, text-grounded psychological or
> relational change across chronological phases of this generated story?

The package keeps ArcANE's two useful abstraction levels:

1. phase-level evidence about action, reasoning, and their entailment;
2. one trajectory-level judgment across all phases.

## Pipeline

```text
generated story
  -> paragraph spans with stable P#### IDs
  -> principal-character inventory
  -> per-character candidate axes + quoted evidence
  -> local verbatim-evidence validation
  -> one blind LLM judge per scoreable axis
  -> axis / character / story aggregates
```

Extraction is frozen before scoring. The judge sees the complete generated
story, the candidate axis, and its grounded evidence, but not the generator,
condition, seed, or any reference arc. Only extraction run 1 is scored by
default; additional runs are retained as a recall/sensitivity audit rather
than pooled into the primary result.

## Metrics

All four dimensions are scored from 1 to 100 by default:

| Metric | ArcANE analogue | Question |
| --- | --- | --- |
| `action_progression` | APF | Do observable actions or dialogue shift across phases in the claimed direction? |
| `reasoning_progression` | RPF | Does trigger appraisal, goal, or strategy change across phases, with textual support? |
| `reasoning_action_entailment` | RAE | Does behavior follow plausibly from the narrated state at the same and later phases? |
| `trajectory_coherence` | PTF | Are ordering, direction, turning points, pacing, and regressions coherent across phases? |

`average` is the mean of those four values.

The categorical output `has_arc` is computed from a pre-declared rule rather
than accepting the judge's label alone. By default, an axis qualifies only if:

- the judge says `arc_judgment = present`;
- at least two phases have verbatim grounded evidence;
- evidence-grounding rate is at least 0.8;
- average score is at least 60; and
- every dimension is at least 40.

These thresholds are provisional operational choices, not numbers estimated
by the ArcANE paper. Calibrate them on human labels and report the chosen
threshold before comparing systems.

## Aggregation

The package reports:

- axis level: score, label, `has_arc`;
- character level: whether any candidate axis qualifies, plus the best-axis
  score;
- story level: `character_arc_rate = characters with a qualifying arc /
  evaluable characters`, mean best-axis score, and dimension means;
- batch level: rates and score means grouped by `generator` and `condition`.

A character with no scoreable grounded axis is excluded from the denominator
and reported through `character_coverage`. It is not silently counted as a
negative.

## Usage

One story:

```bash
python -m evaluation.generated_arc run \
  --story path/to/generated_story.txt \
  --story-id sys-a-prompt-001-seed-0 \
  --title "Example story" \
  --generator system_a \
  --condition arc_prompt \
  --prompt-id prompt-001 \
  --seed 0 \
  --output-root results/generated_arc
```

The default backend/model match ArcANE's main text judge
(`deepseek/deepseek-v4-flash` through OpenRouter). Set an independent
credential through `GENARC_JUDGE_API_KEY` or `OPENROUTER_API_KEY`. The model,
backend, score scale, retries, and concurrency can be changed with
`GENARC_*` environment variables in `config.py`.

Batch manifest:

```json
[
  {
    "story_id": "system-a-p001-s0",
    "title": "Example",
    "story_path": "stories/system-a-p001-s0.txt",
    "generator": "system_a",
    "condition": "baseline",
    "prompt_id": "p001",
    "seed": 0,
    "control": "generated"
  }
]
```

Relative paths are resolved against the manifest. JSONL is also accepted.

```bash
python -m evaluation.generated_arc run \
  --manifest experiments/generated_stories.jsonl \
  --output-root results/generated_arc

python -m evaluation.generated_arc report \
  --output-root results/generated_arc
```

Useful split stages:

```bash
# 1. Freeze extraction for manual audit.
python -m evaluation.generated_arc extract \
  --story story.txt --story-id example \
  --output-root results/generated_arc

# 2. Re-score the frozen extraction without repeating extraction.
python -m evaluation.generated_arc score \
  --story story.txt \
  --extraction results/generated_arc/example/extraction.json \
  --output-dir results/generated_arc/example
```

### Score an existing ArcANE Stage 1 extraction

If ArcANE has already produced a `final/<character>_final_axes.json` or an
`all_characters_*.json` file, use the source-free adapter:

```bash
python -m evaluation.generated_arc score-arcane \
  --story Anna_Kareina \
  --characters all
```

For selected characters:

```bash
python -m evaluation.generated_arc score-arcane \
  --story Anna_Kareina \
  --characters "Anna Arkadyevna Karenina,Konstantin Dmitrievitch Levin"
```

`--characters selected` prints a numbered list and asks for character numbers.
The output directory, story title, axes-file location, model, score scale, and
axis limit are resolved automatically. `--story` may also be a direct path to
an ArcANE axes JSON file. Case and separator differences are accepted; a
unique close misspelling is also resolved with a warning, while ambiguous
matches remain an error that lists the available stories.

```text
results/generated_arc/<story>_arcane_all/
results/generated_arc/<story>_arcane_selected_<characters>/
```

The judge receives only the extracted axes, phase descriptions, chapter
ranges, and key moments. It never receives the full novel, so the original
length limit does not apply.

This command has a different verification scope. It scores whether the
extracted candidate itself articulates a clear and coherent arc; it cannot
verify that the candidate is faithful to the source text. Its JSON records
are marked `verification_scope: "arcane_extraction_only"`. Do not report this
mode as source-grounded arc evaluation. A publication-grade source check needs
retrieved phase evidence from the cited chapters, not merely the extraction
summary.

All per-story writes are atomic at the file level. A failed judge run writes
an `.partial` JSONL and preserves the previous canonical `scores.jsonl` and
`summary.json`.

## Output layout

```text
results/generated_arc/
  <story_id>/
    extraction.json
    scores.jsonl
    summary.json
  report.json
```

`extraction.json` includes the story SHA-256. Scoring refuses a story that
changed after extraction.

See [EXPERIMENT.md](EXPERIMENT.md) for the controlled experiment protocol and
validation plan.
