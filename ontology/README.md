# Ontology Files

## Artifacts

- `ontology/ai_general_education.owl` — the gold-standard skeleton ontology (RDF/XML, UTF-8):
  - 147 named classes: 6 top-level (T1–T6) + 10 intermediate (gold-standard concept levels L_A–L_K) + 131 gold-standard concept classes (C{id});
  - `rdfs:label` holds the Chinese concept name; `rdfs:comment` records the definition, gold-standard id, frequency tier, and decision type;
  - `rdfs:subClassOf`: concept → intermediate → top level (131 + 10), plus 50 concept-level is-a relations (cycle detection: 0 dropped);
  - `owl:equivalentClass`: 4 pairs (e.g., 智能体 ↔ 智能体(Agent)), from `golden_relations_extended.csv`;
  - object properties (each with a Chinese label and an explicit inverse): hasPart/partOf, hasPrerequisite, appliedIn, basedOn, operatesOn, overlapsWith (soft semantics); assertions are expressed as class-level restrictions (⊑ prop some X).
- `ontology/ai_general_education_elk.owl` — an ELK-profile-compatible copy without inverse properties, for re-inspection in restricted environments.

## Top-level mapping

T1 AI基础与元概念 ← A (general overview); T2 AI核心技术 ← C (technology/paradigm);
T3 AI素养与能力 ← F (literacy and ability); T4 AI伦理与治理 ← G + H (ethics, governance, law);
T5 AI工具与数据资源 ← E + I (tools, data resources); T6 AI应用与前沿方向 ← D + J + K (tasks, applications, frontier).

## Indicators

See `data/results/eval/ontology_metrics.md`:

- RR = 0.251 (0.561 under the alternative setting counting only the 50 concept-level is-a edges as hierarchical);
- IR = 4.66 (parent-class average) / 1.30 (all-class average);
- AvgDepth = 3.44 (leaf classes 3.59, maximum 7);
- Coverage: (a) gold standard → skeleton 131/131 (reference baseline); (b) LLM candidates (corpus union merged into slots): Qwen-B 0.626 (Core .941 / Common .882 / Peripheral .526).

## Consistency

Checked with Protégé 5.6.5 and the HermiT reasoner: no unsatisfiable classes, no warnings (92 ms complete).
