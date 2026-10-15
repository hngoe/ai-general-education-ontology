# Reviewer Brief — Independent Audit of an AI General-Education Concept Benchmark

> Purpose: independently verify a concept/relation benchmark ("gold standard") built for a study on
> AI general education in Chinese universities. Your task takes about 30–45 minutes. No AI background
> is required beyond familiarity with what a general (non-major) undergraduate AI course teaches.

## What you are auditing

The study team built a benchmark of **131 concepts** and **75 typed relations** meant to represent what
Chinese university AI **general-education** (通识) courses actually teach. The team screened 222
candidate concepts; 131 were **kept**, 91 were **excluded**. We ask you to independently judge a
random sample of **kept**, **excluded**, and **relation** items, so we can report inter-rater agreement.

## Your task (three spreadsheets)

Open the three CSVs in `data/gs_audit/`:

1. `thirdparty_review_concepts.csv` — 30 concepts that were **kept**.
2. `thirdparty_review_excluded_concepts.csv` — 20 concepts that were **excluded**.
3. `thirdparty_review_relations.csv` — 20 relations.

For each row, fill the `decision` column (and optionally `reviewer` + `comment`):

| Spreadsheet | `decision` values |
|---|---|
| concepts (kept) | `keep` / `remove` / `borderline` |
| concepts (excluded) | `keep` / `remove` / `borderline` |
| relations | `keep` / `remove` / `borderline` / `change_type` / `change_direction` |

## The single decision rule

Judge **whether the item deserves to be a knowledge point in a Chinese university AI general-education
course for non-AI-major undergraduates** — i.e., after the course a student should be able to say
*what it is, which family it belongs to, what problem it solves, and one typical example*.

- For a **concept**: `keep` = it deserves its own slot as a teachable knowledge point; `remove` = it is
  a sub-algorithm detail, a product/platform name, a slogan, a background discipline, or an
  industry application scene that is only ever used as an example; `borderline` = genuinely unsure.
- For a **relation**: judge whether the direction and type (is-a / part-of / prerequisite / applied-in)
  are correct; `change_type` / `change_direction` if the edge is real but mislabeled.

Do **not** look up external sources; judge from your own teaching sense. Work independently.

## Return

Fill the `decision` column in the three CSVs and return them. We will compute percent agreement and a
simple kappa between your judgments and the team's original decisions, and report any disagreements.
