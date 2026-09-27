# -*- coding: utf-8 -*-
"""Generate the OWL skeleton from the gold standard (golden_concepts_tiered + golden_relations + extended).

Design:
- Three class layers:
  T1..T6 top-level domain classes (Chinese labels) ← GS concept_level groups (10 layers, intermediate classes)
  ← GS 131 concept classes (C{id}, label = concept name).
  Top-level mapping (adjustable in Protégé):
    T1 AI基础与元概念 ← A总论
    T2 AI核心技术 ← C技术/范式
    T3 AI素养与能力 ← F素养能力
    T4 AI伦理治理与法律 ← G伦理治理 + H法律监管
    T5 AI工具与数据资源 ← E工具产品 + I数据资源
    T6 AI任务应用与前沿 ← D任务与应用 + J前沿方向 + K教育应用
- is-a（主 37 + 扩展 is-a 13）→ rdfs:subClassOf（含环检测，成环边跳过并记录）；
- equivalent（扩展 4 对）→ owl:equivalentClass；
- part-of/prerequisite/applied-in（主+扩展）与 based-on/operates-on/overlapsWith → 对象属性断言；
  overlapsWith 为软语义（注释说明，不进入一致性关键推理叙事）；
- 注解：rdfs:label(zh)、rdfs:comment=定义、:gsId、:frequencyTier、:decisionType。

输出：ontology/ai_general_education.owl（RDF/XML）。
"""
import csv
import html
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GS = ROOT / "data" / "gold_standard"
OUT_DIR = ROOT / "ontology"

# ELK 兼容副本：去掉 owl:inverseOf（OWL 2 EL 不支持逆属性，Protégé 市场装不了 HermiT 时用）
NO_INVERSE = "--no-inverse" in sys.argv
OUT = OUT_DIR / ("ai_general_education_elk.owl" if NO_INVERSE else "ai_general_education.owl")

BASE = "http://example.org/ai-general-education/"

LEVEL_TOP = {
    "A总论": "T1", "C技术/范式": "T2", "F素养能力": "T3",
    "G伦理治理": "T4", "H法律监管": "T4",
    "E工具产品": "T5", "I数据资源": "T5",
    "D任务与应用": "T6", "J前沿方向": "T6", "K教育应用": "T6",
}
TOP_LABEL = {
    "T1": "AI基础与元概念", "T2": "AI核心技术", "T3": "AI素养与能力",
    "T4": "AI伦理与治理", "T5": "AI工具与数据资源", "T6": "AI应用与前沿方向",
}
LEVEL_LABEL = {"A总论": "总论与元概念", "C技术/范式": "技术/范式", "F素养能力": "素养能力",
               "G伦理治理": "伦理治理", "H法律监管": "法律监管", "E工具产品": "工具产品",
               "I数据资源": "数据资源", "D任务与应用": "任务与应用", "J前沿方向": "前沿方向",
               "K教育应用": "教育应用"}
LEVEL_LOCAL = {"A总论": "L_A", "C技术/范式": "L_C", "F素养能力": "L_F", "G伦理治理": "L_G",
               "H法律监管": "L_H", "E工具产品": "L_E", "I数据资源": "L_I", "D任务与应用": "L_D",
               "J前沿方向": "L_J", "K教育应用": "L_K"}


def esc(s):
    return html.escape(str(s), quote=True)


def read_csv(name):
    return list(csv.DictReader(open(GS / name, encoding="utf-8-sig")))


def main():
    concepts = read_csv("golden_concepts_tiered.csv")
    main_rel = read_csv("golden_relations.csv")
    ext_rel = read_csv("golden_relations_extended.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    concept = {}
    for r in concepts:
        cid = str(r["concept_id"]).strip()
        concept[cid] = {"name": r["concept_name"].strip(), "level": r["concept_level"].strip(),
                        "def": (r.get("definition") or "").strip(),
                        "tier": (r.get("frequency_tier") or "Peripheral").strip(),
                        "decision": (r.get("decision_type") or "").strip()}

    # ---------------- 收集断言 ----------------
    sub_of = []      # (child, parent) 概念级 is-a
    equiv = []       # (a, b)
    props = defaultdict(list)  # prop -> list[(a,b)]

    def _prop(p, a, b):
        if a and b and a != b:
            props[p].append((a, b))

    def _is_a(a, b):
        if a and b and a != b:
            sub_of.append((a, b))

    for r in main_rel:
        a, b = str(r["source_concept_id"]).strip(), str(r["target_concept_id"]).strip()
        t = r["relation_type"].strip()
        if t == "is-a":
            _is_a(a, b)
        elif t in ("part-of", "prerequisite", "applied-in"):
            _prop(t, a, b)
    for r in ext_rel:
        a = str(r.get("source_concept_id") or "").strip()
        b = str(r.get("target_concept_id") or "").strip()
        t = (r.get("relation_type") or "").strip()
        if t == "is-a":
            _is_a(a, b)
        elif t == "equivalent":
            equiv.append((a, b))
        elif t in ("part-of", "prerequisite", "applied-in", "based-on", "operates-on", "overlapping"):
            _prop(t, a, b)
        elif t == "standalone(独立节点)":
            _prop("standalone", a, "")  # 仅记录
        # 其它类型忽略

    # ---------------- 环检测（is-a 图） ----------------
    adj = defaultdict(list)
    for a, b in sub_of:
        if a in concept and b in concept:
            adj[a].append(b)

    def has_cycle_through(a, b):
        # 新增 a->b 是否成环：存在 b ->* a 路径
        seen = set()
        stack = [b]
        while stack:
            x = stack.pop()
            if x == a:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj.get(x, []))
        return False

    keep = []
    dropped = []
    for a, b in sub_of:
        if a not in concept or b not in concept:
            dropped.append((a, b, "端点不在 GS"))
            continue
        if has_cycle_through(a, b):
            dropped.append((a, b, "成环"))
            continue
        keep.append((a, b))
        adj[a].append(b)

    # ---------------- 生成 RDF/XML ----------------
    L = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
             'xmlns:owl="http://www.w3.org/2002/07/owl#" '
             'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
             'xmlns:xsd="http://www.w3.org/2001/XMLSchema#">')
    L.append('  <owl:Ontology rdf:about="' + BASE + '">')
    L.append('    <rdfs:comment xml:lang="zh">高校人工智能通识教育领域本体（金标准骨架版）。'
             '由 golden_concepts_tiered / golden_relations / golden_relations_extended 程序化生成，'
             '供 Protégé 精修与 Pellet/HermiT 校验。</rdfs:comment>')
    L.append('  </owl:Ontology>')

    def cls(local, label, comment=None, extra=""):
        s = f'  <owl:Class rdf:about="{BASE}{local}">\n'
        s += f'    <rdfs:label xml:lang="zh">{esc(label)}</rdfs:label>\n'
        if comment:
            s += f'    <rdfs:comment xml:lang="zh">{esc(comment)}</rdfs:comment>\n'
        s += extra
        s += '  </owl:Class>'
        return s

    # 顶层类
    for t in sorted(TOP_LABEL):
        L.append(cls(t, TOP_LABEL[t], "领域顶层类（由 GS concept_level 分组映射，可在 Protégé 内调整）。"))
    # 中层 level 类 + 挂到顶层
    for lvl in LEVEL_LOCAL:
        L.append(cls(LEVEL_LOCAL[lvl], LEVEL_LABEL[lvl],
                     f"GS concept_level 中层类（{lvl}）。",
                     f'    <rdfs:subClassOf rdf:resource="{BASE}{LEVEL_TOP[lvl]}"/>\n'))
    # 概念类：挂 level + 注解
    for cid in sorted(concept, key=int):
        c = concept[cid]
        comment = (c["def"] or "（无定义）") + f"｜GS {cid}｜分层 {c['tier']}｜{c['decision']}"
        L.append(cls(f"C{cid}", c["name"], comment,
                     f'    <rdfs:subClassOf rdf:resource="{BASE}{LEVEL_LOCAL[c["level"]]}"/>\n'))
    # is-a
    for a, b in keep:
        L.append(f'  <owl:Class rdf:about="{BASE}C{a}">')
        L.append(f'    <rdfs:subClassOf rdf:resource="{BASE}C{b}"/>')
        L.append('  </owl:Class>')
    # equivalent
    for a, b in equiv:
        if a in concept and b in concept:
            L.append(f'  <owl:Class rdf:about="{BASE}C{a}">')
            L.append(f'    <owl:equivalentClass rdf:resource="{BASE}C{b}"/>')
            L.append('  </owl:Class>')

    # ---------------- 对象属性 ----------------
    PROP = {
        "part-of": ("hasPart", "有组成部分（A hasPart B：A 的组成部分含 B）", "partOf"),
        "prerequisite": ("hasPrerequisite", "先修关系（A hasPrerequisite B：A 以 B 为先修）", "prerequisiteOf"),
        "applied-in": ("appliedIn", "应用关系（A appliedIn B：A 应用于 B）", "applicationOf"),
        "based-on": ("basedOn", "奠基/基于（扩展层，来源为教材/理论结构）", "foundationOf"),
        "operates-on": ("operatesOn", "操作对象（扩展层）", "operatedOnBy"),
        "overlapping": ("overlapsWith", "语义交叠（扩展层软语义，不做一致性关键断言）", None),
    }
    for key, (p_local, p_cn, inv) in PROP.items():
        # 主属性
        L.append(f'  <owl:ObjectProperty rdf:about="{BASE}{p_local}">')
        L.append(f'    <rdfs:label xml:lang="zh">{esc(p_cn)}</rdfs:label>')
        if inv and not NO_INVERSE:
            L.append(f'    <owl:inverseOf rdf:resource="{BASE}{inv}"/>')
        L.append('  </owl:ObjectProperty>')
        # 反向属性（独立并列元素；ELK 副本不输出逆属性）
        if inv and not NO_INVERSE:
            L.append(f'  <owl:ObjectProperty rdf:about="{BASE}{inv}">')
            L.append(f'    <rdfs:label xml:lang="zh">{esc("逆：" + p_cn)}</rdfs:label>')
            L.append('  </owl:ObjectProperty>')
    # 类级断言：以 restriction 表达（A ⊑ hasPart some B 等）
    for key, (p_local, _, _) in PROP.items():
        for a, b in props.get(key, []):
            if a not in concept or not b or b not in concept:
                continue
            L.append(f'  <owl:Class rdf:about="{BASE}C{a}">')
            L.append('    <rdfs:subClassOf>')
            L.append('      <owl:Restriction>')
            L.append(f'        <owl:onProperty rdf:resource="{BASE}{p_local}"/>')
            L.append('        <owl:someValuesFrom>')
            L.append(f'          <owl:Class rdf:about="{BASE}C{b}"/>')
            L.append('        </owl:someValuesFrom>')
            L.append('      </owl:Restriction>')
            L.append('    </rdfs:subClassOf>')
            L.append('  </owl:Class>')

    L.append('</rdf:RDF>')
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------- 结构自检报告 ----------------
    print(f"[owl] 写入 {OUT}")
    print(f"  概念类: {len(concept)}  中层类: {len(LEVEL_LOCAL)}  顶层类: {len(TOP_LABEL)}")
    print(f"  is-a 断言: {len(keep)} (丢弃 {len(dropped)})   dropped={dropped[:6]}")
    print(f"  equivalent: {len(equiv)}")
    total_rel = sum(len(v) for k, v in props.items() if k != "standalone")
    print(f"  非层次对象属性断言: {total_rel}   by-type: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(props.items()) if k != "standalone"))


if __name__ == "__main__":
    main()
