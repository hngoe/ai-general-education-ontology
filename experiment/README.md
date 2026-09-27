# Extraction Experiment

## Overview

Runs the LLM concept/relation extraction experiment described in the manuscript (§3.3).

- Matrix: **DeepSeek / GLM / Qwen × 19 course syllabi × strategy A (one-shot) / strategy B (three-round decomposed)**;
- temperature = 0, JSON validation with repair retry (up to 3 times).

## Files

| File | Purpose |
|---|---|
| `prompts.py` | Prompt templates for strategies A/B and the JSON validators (no gold-standard content is injected) |
| `config.py` | Paths, model endpoints, limits, mock term list |
| `llm_client.py` | OpenAI-compatible transport (backoff retry) + tolerant JSON parsing + repair-retry orchestration; `--dry-run` mock |
| `normalize.py` | Gold-standard slot / alias / equivalence-group merging and matching (131 concepts → merged slot set) |
| `run_experiment.py` | Main experiment: corpus traversal, A/B orchestration, result writing, manifest, resumable |
| `eval_quick.py` | Document-level concept P/R/F1 + tier stratification + four-type relation P/R/F1 (macro-averaged) |

## API keys (environment variables only)

Keys are read from environment variables or `experiment/.env` and are never committed:

- `DEEPSEEK_API_KEY` (DeepSeek)
- `ZHIPU_API_KEY` or `GLM_API_KEY` (GLM)
- `DASHSCOPE_API_KEY` (Qwen)

Copy `experiment/.env.example` to `experiment/.env` and fill in the keys. Default models:
deepseek-chat / glm-4-air / qwen-plus (changeable in `config.py` `MODELS`). Without keys, run
`--dry-run` to validate the pipeline offline.

## Usage

```powershell
# 1) Pipeline smoke test (mock, no network; 1 syllabus × 2 strategies × 3 models)
python experiment/run_experiment.py --dry-run --doc 合肥 --strategies A,B --models deepseek,glm,qwen

# 2) Validate the JSON repair-retry path (mock, 50% invalid first answers)
python experiment/run_experiment.py --dry-run --doc 合肥 --strategies A --mock-fail-rate 0.5 --force

# 3) Real calibration run (one syllabus)
python experiment/run_experiment.py --doc 合肥 --strategies A,B --models deepseek,glm,qwen

# 4) Quick evaluation (document-level macro P/R/F1)
python experiment/eval_quick.py
python experiment/eval_quick.py --one data/results/deepseek_A_01_合肥工业大学_人工智能基础.json   # single-file FP/FN detail

# 5) Full run (19 syllabi × 3 models × 2 strategies)
python experiment/run_experiment.py --doc all
```

Output: `data/results/{model}_{strategy}_{doc}.json` + `manifest.jsonl` (calls / retries / tokens per combination).

## Evaluation conventions

1. Relation gold = the main relation set `golden_relations.csv` (75 four-type relations). The extended set
   `golden_relations_extended.csv` is **not** used for evaluation, only for OWL construction.
2. Concept gold = all 131 concepts; equivalent/near-synonym groups are merged into slots at the matching layer
   (any hit counts once).
3. Relations use a **document-level denominator**: only gold relations whose two endpoint concepts are both
   anchored in that document are counted; per-document macro average plus a corpus-level union.
4. Tier stratification: Core/Common/Peripheral per-document macro R/F1.
