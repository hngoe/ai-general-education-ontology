# A Curriculum-Grounded Benchmark and Ontology for LLM Concept and Relation Extraction from Chinese University AI Syllabi

This repository accompanies the paper **"A Curriculum-Grounded Benchmark and Ontology for LLM Concept and Relation
Extraction from Chinese University AI Syllabi."** It releases the curriculum-grounded benchmark,
the ontology, the prompt templates, and the analysis scripts used in the study.

## Contents

| Path | Description |
|---|---|
| `data/gold_standard/` | The gold-standard benchmark: 131 concepts (with frequency tiering and decision type), 75 main four-type relations, 44 extended items (43 relations + 1 standalone node), the 222-item candidate pool, the screening guideline, and the saturation curve. |
| `data/held_out/` | The held-out test set: 14 author-annotated syllabi gold annotations (`heldout_gold_concepts_annotated.csv`) and metadata. |
| `data/gs_audit/` | Third-party audit records: 30 kept + 20 excluded concept decisions and 20 relation decisions, each with two independent reviewers' decisions and comments; `audit_results.md` summarizes the agreement. |
| `data/metadata.csv` | Corpus source list: source type, institutional tier, audience, source URL, and acquisition date for the main and auxiliary corpora. |
| `ontology/` | The OWL ontology (`ai_general_education.owl`) and an OWL 2 EL-profile-compatible copy, with structural indicators. |
| `experiment/` | Prompt templates (`prompts.py`), the LLM client, normalization/matching, the main extraction driver, and the evaluation scripts. |

## Benchmark summary

- **131 concepts**, tiered into Core / Common / Peripheral (17 / 17 / 97); 127 equivalence-merged slots after
  merging four equivalent pairs.
- **75 main relations** of four types (is-a 37, part-of 18, prerequisite 8, applied-in 12), used for evaluation;
  **44 extended items (43 relations + 1 standalone node)** used only for ontology construction.
- Constructed from a four-layer source portfolio through two-author independent double-blind screening with
  adjudication (percent agreement 85.6%; Cohen's κ = 0.707; Krippendorff's α = 0.708; Gwet's AC1 = 0.716), then
  independently audited by two external reviewers.
- **Ontology**: 147 named classes (6 top-level, 10 intermediate, 131 concept classes), HermiT-consistent
  (no unsatisfiable classes, no warnings), RR = 0.251 (alternative 0.561), IR = 4.66 / 1.30, AvgDepth = 3.44.

## Extraction experiment

- Models: DeepSeek (deepseek-chat), GLM (glm-4-air), Qwen (qwen-plus); temperature = 0.
- Strategies: A (one-shot) and B (decomposed multi-round).
- Matrix: 19 syllabi × 3 models × 2 strategies = 114 combinations, plus a 14-syllabus held-out set and a
  9-document auxiliary corpus.

## Reproducing the experiment

Install dependencies first:

```powershell
pip install -r requirements.txt
```

> **Prerequisite:** the raw syllabus texts (`data/raw/*.txt`) are **not redistributed** in this
> repository (see Data availability below). Obtain them from the corresponding author and place the
> `.txt` files under `data/raw/` before running the commands, otherwise `--doc` will report
> "No document matches".

```powershell
# 1) Pipeline smoke test (mock, no network)
python experiment/run_experiment.py --dry-run --doc 合肥 --strategies A,B --models deepseek,glm,qwen

# 2) Full run (requires API keys; see experiment/README.md)
python experiment/run_experiment.py --doc all

# 3) Evaluation
python experiment/eval_quick.py
```

API keys are read from environment variables or `experiment/.env` and are never committed; see
`experiment/README.md`.

## Data availability

The raw syllabus texts are listed by source URL in `data/metadata.csv` (and in Appendix A of the paper) but are
**not redistributed** in this repository; they are available from the corresponding author upon reasonable request.

## License

See [LICENSE](LICENSE). The ontology, benchmark data, and code are released under the MIT License.
