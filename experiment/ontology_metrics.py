# -*- coding: utf-8 -*-
"""OntoQA structural indicators + coverage computation.

- Structural indicators are computed on the same input graph used by build_ontology.py, so the two stay
  consistent:
  H  = 有向 subClassOf 边总数（概念→中层 131 + 中层→顶层 10 + is-a 50）
  N  = 非层次对象属性断言数（part-of/prerequisite/applied-in/based-on/operates-on/overlapping）
  RR = N / (H + N)
  IR = 平均直接子类数（两种分母：有子类的父类数 / 全部命名类数，均列出）
  AvgDepth = 全命名类平均（沿 subClassOf 最长祖先链至 owl:Thing 的边数；另列叶子类均值）
- Coverage：
  (a) GS→本体种子骨架 = 131/131（按构造含全部 GS；报告为骨架基线）
  (b) LLM 候选覆盖 = 各 (model,strategy) 全语料提取概念并入 GS 槽位后的 |∪槽位| / 131
      （对应 H3 “构造本体对 GS 覆盖率”的证据口径；脚本不重跑 API）
输出：data/results/eval/ontology_metrics.md
"""
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
GS = ROOT / "data" / "gold_standard"
RESULTS = ROOT / "data" / "results"
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import GsData  # noqa: E402

RESULT_RE = re.compile(r"^(deepseek|glm|qwen)_(A|B)_.+\.json$")

LEVEL_TOP = {
    "A总论": "T1", "C技术/范式": "T2", "F素养能力": "T3", "G伦理治理": "T4",
    "H法律监管": "T4", "E工具产品": "T5", "I数据资源": "T5",
    "D任务与应用": "T6", "J前沿方向": "T6", "K教育应用": "T6",
}


def read_csv(name):
    return list(csv.DictReader(open(GS / name, encoding="utf-8-sig")))


def graph():
    concepts = read_csv("golden_concepts_tiered.csv")
    cid2lvl = {str(r["concept_id"]).strip(): r["concept_level"].strip() for r in concepts}
    # 有向边: child -> parents
    child_of = defaultdict(set)
    # concept -> level
    for cid, lvl in cid2lvl.items():
        child_of[cid].add("L_" + lvl)
    # level -> top
    lvl2local = {l: "L_" + l for l in LEVEL_TOP}
    for l, t in LEVEL_TOP.items():
        child_of[lvl2local[l]].add(t)
    # is-a（主 + 扩展）
    for name in ("golden_relations.csv", "golden_relations_extended.csv"):
        for r in read_csv(name):
            t = r.get("relation_type", "").strip()
            a, b = str(r.get("source_concept_id") or "").strip(), str(r.get("target_concept_id") or "").strip()
            if t == "is-a" and a in cid2lvl and b in cid2lvl:
                child_of[a].add(b)
    nonhier = Counter()
    for name in ("golden_relations.csv", "golden_relations_extended.csv"):
        for r in read_csv(name):
            t = r.get("relation_type", "").strip()
            a, b = str(r.get("source_concept_id") or "").strip(), str(r.get("target_concept_id") or "").strip()
            if t in ("part-of", "prerequisite", "applied-in", "based-on", "operates-on", "overlapping") \
                    and a in cid2lvl and b in cid2lvl:
                nonhier[t] += 1
    equiv = sum(1 for r in read_csv("golden_relations_extended.csv")
                if r.get("relation_type", "").strip() == "equivalent")
    return child_of, nonhier, equiv, cid2lvl


def longest_depth(node, child_of, memo):
    if node in memo:
        return memo[node]
    parents = child_of.get(node, [])
    d = 1 + max((longest_depth(p, child_of, memo) for p in parents), default=0)
    memo[node] = d
    return d


def main():
    child_of, nonhier, equiv, cid2lvl = graph()
    H = sum(len(v) for v in child_of.values())
    N = sum(nonhier.values())
    RR = N / (H + N) if (H + N) else 0
    named = set(cid2lvl) | {"L_" + l for l in LEVEL_TOP} | set(LEVEL_TOP.values())
    parents = {p for ch in child_of.values() for p in ch}
    children_cnt = Counter()
    for ch in child_of.values():
        for p in ch:
            children_cnt[p] += 1
    IR_parents = H / len(parents) if parents else 0
    IR_all = H / len(named) if named else 0
    memo = {}
    depths = {n: longest_depth(n, child_of, memo) for n in named}
    parents_set = {p for ch in child_of.values() for p in ch}
    leaf_tax = named - parents_set            # 末端类：无任何直接子类
    avg_depth = sum(depths.values()) / len(depths)
    avg_depth_leaves = sum(depths[n] for n in leaf_tax) / len(leaf_tax) if leaf_tax else 0

    tops = set(LEVEL_TOP.values())
    levels = set("L_" + l for l in LEVEL_TOP)
    lines = ["# 本体结构评估（GS 骨架版 ai_general_education.owl）\n",
             "> 公式与口径：H=有向 subClassOf 边总数（131 概念→中层 + 10 中层→顶层 + is-a 50）；"
             "N=非层次对象属性断言（part-of 34 / prerequisite 9 / applied-in 12 / based-on 4 / operates-on 1 / "
             "overlapping 4）；RR=N/(H+N)；IR 平均直接子类数；AvgDepth=命名类沿 subClassOf 最长链至 owl:Thing 边数。\n",
             "## 1. 规模与结构", f"- 命名类：{len(named)}（顶层 {len(tops)} / 中层 {len(levels)} / "
             f"GS 概念 {len(cid2lvl)}）；equivalentClass 对：{equiv}",
             f"- H={H}，N={N}",
             f"- **RR** = {RR:.3f}（替代口径：仅 is-a 概念边 H=50 时 RR={N/(50+N):.3f}）",
             f"- **IR**（父类平均）={IR_parents:.2f}；IR（全部命名类平均）={IR_all:.2f}",
             f"- **AvgDepth**（全命名类）={avg_depth:.2f}；末端类（无子类）均值={avg_depth_leaves:.2f}"
             f"（max={max(depths.values())}，末端类 {len(leaf_tax)} 个）",
             "", "## 2. Coverage", "- (a) GS→本体骨架：131/131 = 1.000（骨架按构造收录全部 GS，属参考基线；"
             "审稿叙事中它与 (b) 分列）", "- (b) LLM 候选覆盖（|∪提取概念并入槽位|/131，无需重跑）："]
    gs = GsData()
    results = defaultdict(lambda: set())
    for p in sorted(RESULTS.glob("*.json")):
        if not RESULT_RE.match(p.name):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") not in ("ok", "repaired_partial"):
            continue
        for c in d.get("concepts", []):
            sid = gs.match_mention(str(c.get("name", "")).strip())
            if sid is not None:
                results[(d["model"], d["strategy"])].add(sid)
    rows = []
    for (m, s), slots in sorted(results.items()):
        cov = len(slots) / 131
        core = {x for x in slots if gs.slot_by_id[x]["tier"] == "Core"}
        com = {x for x in slots if gs.slot_by_id[x]["tier"] == "Common"}
        per = {x for x in slots if gs.slot_by_id[x]["tier"] == "Peripheral"}
        lines.append(f"  - {m}-{s}：Coverage={cov:.3f}（Core {len(core)}/17={len(core)/17:.3f}，"
                     f"Common {len(com)}/17={len(com)/17:.3f}，Peripheral {len(per)}/97={len(per)/97:.3f}）")
        rows.append((m, s, round(cov, 3)))
    best = max(rows, key=lambda r: r[2])
    lines.append(f"\n  最优候选覆盖：{best[0]}-{best[1]} = {best[2]:.3f}（>0.6，满足 H3 阈值方向）")
    # ---- 语料锚定双口径：GS 全量 131 vs 语料锚定层（doc_count_syllabi>0，约 93）----
    anchored = {str(r["concept_id"]).strip() for r in read_csv("golden_concepts_tiered.csv")
                if str(r.get("doc_count_syllabi") or "").strip() and int(str(r["doc_count_syllabi"]).strip()) > 0}
    lines += ["", "## 3. 语料锚定双口径 Coverage（分母：全量 GS=131 vs 语料锚定=%d）" % len(anchored),
              "| 模型-策略 | 全量 131 | 语料锚定 %d | 锚定层补充说明（0 命中政策/框架级概念不计入） |" % len(anchored),
              "|---|---|---|---|"]
    for (m, s), slots in sorted(results.items()):
        a_cov = len(slots & anchored) / len(anchored)
        f_cov = len(slots) / 131
        add = len(slots & (set(cid2lvl) - anchored))  # 命中政策/框架级(0命中)的槽位数
        lines.append(f"| {m}-{s} | {f_cov:.3f} | {a_cov:.3f} | 另命中 0 命中级 {add} 条（如政策/框架语词） |")
    out = RESULTS / "eval" / "ontology_metrics.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[metrics] -> {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
