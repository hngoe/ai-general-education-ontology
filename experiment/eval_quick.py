# -*- coding: utf-8 -*-
"""Evaluation: document-level P/R/F1 over run_experiment.py outputs.

Concept evaluation (document-level):
- gold = 该大纲文本中锚定的 GS 槽位（alias/子串扫描）；
- 预测 = 结果 concepts 名称经 match_mention 归槽位（去重）；
- 文档级 P/R/F1 → (model,strategy) 组内对 19 份文档宏平均；另报全语料 union 口径。
- 分层：按槽位 frequency_tier（Core/Common/Peripheral）逐文档宏平均 recall/F1。

关系评测（文档级，四型分列）：
- gold = golden_relations.csv 中 source/target 槽位都在该文档锚定的关系（方向敏感）；
- 预测 = 结果 relations 的 source/target 概念名归槽位后的 (src_slot,tgt_slot,type)；
- 每型 P/R/F1 → 宏平均；某文档某型无可判定 gold 则该文档不参与该型平均。

输出：打印 + data/results/eval/quick_report.md + quick_metrics.csv。

用法：
  python experiment/eval_quick.py                 # 汇总全部结果文件
  python experiment/eval_quick.py --doc 合肥      # 只看该文档的 (model,strategy)
  python experiment/eval_quick.py --one results/deepseek_A_*.json   # 单文件含 FP/FN 明细
"""
import argparse
import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config
from normalize import GsData, normalize_name

RESULT_RE = re.compile(r"^(deepseek|glm|qwen)_(A|B)_.+\.json$")
TIERS = ["Core", "Common", "Peripheral"]
REL_TYPES = ["is-a", "part-of", "prerequisite", "applied-in"]


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def macro(rows):
    """rows: list[(doc, P, R, F1)] 或 dict list"""
    if not rows:
        return None
    n = len(rows)
    return (sum(r[1] for r in rows) / n, sum(r[2] for r in rows) / n,
            sum(r[3] for r in rows) / n)


def doc_concept_metrics(gs, result, doc_text):
    gold = gs.anchor_slots(doc_text)
    matched, unmatched, seen = set(), [], set()
    for c in result.get("concepts", []):
        name = c.get("name", "").strip()
        sid = gs.match_mention(name)
        if sid is not None:
            matched.add(sid)
        else:
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                unmatched.append(name)   # 未命中任何 GS 槽位 → 按标准口径计入 FP
    tp = matched & gold
    fp_wrong = matched - gold            # 命中错误槽位
    fn = gold - matched
    p = len(tp) / (len(matched) + len(unmatched)) if (matched or unmatched) else 0.0
    r = len(tp) / len(gold) if gold else 0.0
    # 分层（precision 仅按匹配槽位计；未匹配项无法归层，分层以 recall 为主）
    tier = {}
    for t in TIERS:
        g_t = {s for s in gold if gs.slot_by_id[s]["tier"] == t}
        p_t = {s for s in matched if gs.slot_by_id[s]["tier"] == t}
        tp_t = p_t & g_t
        pt = len(tp_t) / len(p_t) if p_t else 0.0
        rt = len(tp_t) / len(g_t) if g_t else 0.0
        tier[t] = (len(g_t), len(p_t), len(tp_t), pt, rt)
    return {"gold": gold, "matched": matched, "unmatched": unmatched,
            "tp": tp, "fp_wrong": fp_wrong, "fn": fn,
            "P": p, "R": r, "F1": f1(p, r), "tier": tier}


def doc_rel_metrics(gs, result, doc_text):
    anchored = gs.anchor_slots(doc_text)
    # gold per type
    gold_by = defaultdict(list)   # type -> list[(src,tgt)]
    for r in gs.main_relations:
        ss = gs.concept_id2slot.get(r["source_concept_id"])
        ts = gs.concept_id2slot.get(r["target_concept_id"])
        if ss is None or ts is None:
            continue
        if ss in anchored and ts in anchored:
            gold_by[r["relation_type"]].append((ss, ts))
    # predicted per type
    by_id = {c.get("id"): c.get("name", "") for c in result.get("concepts", [])}
    pred_by = defaultdict(list)
    for rel in result.get("relations", []):
        rt = rel.get("type")
        if rt not in REL_TYPES:
            continue
        ss = gs.match_mention(by_id.get(rel.get("source", ""), ""))
        ts = gs.match_mention(by_id.get(rel.get("target", ""), ""))
        if ss is None or ts is None or ss == ts:
            continue  # 未命中 GS 槽位 或 同槽自指，不计
        pred_by[rt].append((ss, ts))
    out = {}
    for rt in REL_TYPES:
        g, p = set(gold_by[rt]), set(pred_by[rt])
        if not g:
            out[rt] = None   # 该文档无可判定金标准
            continue
        tp = p & g
        fp = p - g
        fn = g - p
        pr = len(tp) / len(p) if p else 0.0
        rc = len(tp) / len(g) if g else 0.0
        out[rt] = (len(g), len(p), len(tp), pr, rc, f1(pr, rc))
    return out


def load_results(base=None):
    base = Path(base) if base else config.RESULTS_DIR
    out = {}
    for p in sorted(base.glob("*.json")):
        if not RESULT_RE.match(p.name):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") in ("error", "invalid_after_retry"):
            continue
        out[(d["model"], d["strategy"], d["doc_id"])] = d
    return out


def fmt_num(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="", help="只看含该子串的文档")
    ap.add_argument("--one", default="", help="单结果文件路径：输出该文件 FP/FN 明细")
    ap.add_argument("--dir", default="", help="结果目录（默认 data/results；可比对归档如 calib_v01）")
    ap.add_argument("--models", default="", help="只看这些模型，逗号分隔（如 deepseek,glm）")
    ap.add_argument("--heldout", action="store_true",
                    help="held-out 独立集评测：gold 读 data/held_out/heldout_gold_concepts.csv（人工标注），结果读 data/results_heldout")
    args = ap.parse_args()
    gs = GsData()
    if args.heldout:
        run_heldout(args, gs)
        return
    base = Path(args.dir) if args.dir else config.RESULTS_DIR
    only_models = {m.strip() for m in args.models.split(",") if m.strip()}

    if args.one:
        p = Path(args.one)
        d = json.loads(p.read_text(encoding="utf-8"))
        text = (config.RAW_DIR / f"{d['doc_id']}.txt").read_text(encoding="utf-8-sig")
        cm = doc_concept_metrics(gs, d, text)
        rm = doc_rel_metrics(gs, d, text)
        print(f"== 单文件 {p.name} status={d.get('status')}")
        print(f"  概念: gold={len(cm['gold'])} 匹配槽位={len(cm['matched'])} "
              f"未匹配={len(cm['unmatched'])} P={cm['P']:.3f} R={cm['R']:.3f} F1={cm['F1']:.3f}")
        wrong = [(gs.slot_by_id[s]['canonical'], s) for s in sorted(cm['fp_wrong'], key=int)]
        print("  命中错误槽位:", wrong[:15])
        print("  未匹配(计入FP):", cm['unmatched'][:25])
        fn_names = [gs.slot_by_id[s]['canonical'] for s in sorted(cm['fn'], key=int)]
        print("  FN(漏提):", fn_names[:25])
        for rt in REL_TYPES:
            v = rm[rt]
            if v is None:
                print(f"  关系 {rt}: 该文档无可判定金标准")
            else:
                g, pr, tp, prec, rec, f = v
                print(f"  关系 {rt}: gold={g} pred={pr} TP={tp} P={prec:.3f} R={rec:.3f} F1={f:.3f}")
        return

    results = load_results(base)
    if not results:
        sys.exit("[eval] No evaluable results (data/results/*.json with status ok/repaired_partial)."
                 "Run python experiment/run_experiment.py --dry-run ... or a real run first.")
    groups = sorted({(m, s) for (m, s, _) in results if not only_models or m in only_models})
    report_lines = ["# Quick evaluation (document-level macro average)\n"]
    table = []
    for (model, strategy) in groups:
        rows = [(m, s, doc) for (m, s, doc) in results if (m, s) == (model, strategy)]
        if args.doc:
            rows = [x for x in rows if args.doc in x[2]]
        if not rows:
            continue
        c_pairs, c_all = [], []
        rel_acc = {rt: [] for rt in REL_TYPES}
        tier_acc = {t: [] for t in TIERS}
        for (_, _, doc) in rows:
            text = (config.RAW_DIR / f"{doc}.txt").read_text(encoding="utf-8-sig")
            cm = doc_concept_metrics(gs, results[(model, strategy, doc)], text)
            if cm["gold"]:
                c_pairs.append((doc, cm["P"], cm["R"], cm["F1"]))
            # union 口径（未匹配项以名称键计入，永远非 TP）
            c_all.append({"gold": cm["gold"], "matched": cm["matched"],
                          "unmatched": cm["unmatched"]})
            for t in TIERS:
                g_t, p_t, tp_t, pt, rt = cm["tier"][t]
                if g_t:
                    tier_acc[t].append((doc, pt, rt, f1(pt, rt)))
            rm = doc_rel_metrics(gs, results[(model, strategy, doc)], text)
            for rt in REL_TYPES:
                if rm[rt] is not None:
                    g, pr, tp, prec, rec, f = rm[rt]
                    rel_acc[rt].append((doc, prec, rec, f))
        mc = macro(c_pairs)
        # union 概念
        ug = set().union(*(x["gold"] for x in c_all))
        up_m = set().union(*(x["matched"] for x in c_all))
        up_u = set()
        for x in c_all:
            up_u |= {"u|" + n.lower().strip() for n in x["unmatched"]}
        up = up_m | up_u
        utp = up_m & ug
        ur = len(utp) / len(ug) if ug else 0.0
        uprec = len(utp) / len(up) if up else 0.0
        print(f"\n== {model} / 策略{strategy}  (文档数={len(c_pairs)})")
        print(f"  概念(文档级宏平均): P={fmt_num(mc[0])} R={fmt_num(mc[1])} F1={fmt_num(mc[2])}  | union: P={uprec:.3f} R={ur:.3f}")
        tier_line = "  "
        for t in TIERS:
            m = macro(tier_acc[t])
            tier_line += f"{t}: R={fmt_num(m[1] if m else None)} F1={fmt_num(m[2] if m else None)}  "
        print(tier_line)
        for rt in REL_TYPES:
            m = macro(rel_acc[rt])
            if m:
                print(f"  关系 {rt:<12}: P={m[0]:.3f} R={m[1]:.3f} F1={m[2]:.3f}  (参与文档 {len(rel_acc[rt])})")
        # 记录
        report_lines.append(f"## {model} / 策略{strategy}\n")
        report_lines.append(f"- 概念（文档级宏平均）：P={fmt_num(mc[0])} R={fmt_num(mc[1])} F1={fmt_num(mc[2])}；"
                            f"union 口径：P={uprec:.3f} R={ur:.3f}")
        report_lines.append("- 分层 R/F1："
                            + "  ".join(f"{t} R={fmt_num(macro(tier_acc[t])[1] if tier_acc[t] else None)}"
                                        f" F1={fmt_num(macro(tier_acc[t])[2] if tier_acc[t] else None)}"
                                        for t in TIERS))
        for rt in REL_TYPES:
            m = macro(rel_acc[rt])
            report_lines.append(f"- 关系 {rt}："
                                + ("P={:.3f} R={:.3f} F1={:.3f}（参与文档 {}）".format(*m, len(rel_acc[rt]))
                                   if m else "无可判定文档"))
        row = {"model": model, "strategy": strategy,
               "concept_P": None if not mc else round(mc[0], 3),
               "concept_R": None if not mc else round(mc[1], 3),
               "concept_F1": None if not mc else round(mc[2], 3),
               "union_R": round(ur, 3)}
        for t in TIERS:
            row[f"{t}_R"] = round(macro(tier_acc[t])[1], 3) if tier_acc[t] else None
            row[f"{t}_F1"] = round(macro(tier_acc[t])[2], 3) if tier_acc[t] else None
        table.append(row)

    evdir = base / "eval"
    evdir.mkdir(parents=True, exist_ok=True)
    (evdir / "quick_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if table:
        with open(evdir / "quick_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    print(f"\n[eval] 报告: {evdir / 'quick_report.md'}")


# ---------------------------------------------------------------- held-out 独立集评测

def load_heldout_gold(gs, heldout_dir):
    """读人工文档级标注 heldout_gold_concepts.csv：
    present=y 的概念 → 该文档 gold 槽位集合；is_new_concept=y → 开放集新概念名列表。
    与 anchor_slots 不同，这里 gold 来自人工逐文档判断，不是子串匹配。
    """
    gold_csv = Path(heldout_dir) / "heldout_gold_concepts.csv"
    gold_by = defaultdict(set)
    new_by = defaultdict(list)
    if not gold_csv.exists():
        return gold_by, new_by
    yes = ("y", "yes", "1", "true")
    for r in csv.DictReader(open(gold_csv, encoding="utf-8-sig")):
        doc = (r.get("doc_id") or "").strip()
        if not doc:
            continue
        if (r.get("present") or "").strip().lower() in yes:
            cid = (r.get("concept_id") or "").strip()
            slot = gs.concept_id2slot.get(cid) if cid else None
            if slot is not None:
                gold_by[doc].add(slot)
        if (r.get("is_new_concept") or "").strip().lower() in yes:
            nn = (r.get("new_concept_name") or "").strip()
            if nn:
                new_by[doc].append(nn)
    return gold_by, new_by


def heldout_doc_metrics(gs, result, gold_slots, new_names):
    matched, unmatched, seen = set(), [], set()
    for c in result.get("concepts", []):
        name = (c.get("name") or "").strip()
        sid = gs.match_mention(name)
        if sid is not None:
            matched.add(sid)
        else:
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                unmatched.append(name)
    tp = matched & gold_slots
    fp = matched - gold_slots
    fn = gold_slots - matched
    p = len(tp) / (len(matched) + len(unmatched)) if (matched or unmatched) else 0.0
    r = len(tp) / len(gold_slots) if gold_slots else 0.0
    tier = {}
    for t in TIERS:
        g_t = {s for s in gold_slots if gs.slot_by_id[s]["tier"] == t}
        tp_t = matched & g_t
        tier[t] = (len(g_t), len(tp_t), (len(tp_t) / len(g_t)) if g_t else 0.0)
    # 开放集新概念召回：非 GS 命中（unmatched）中，规范化后与新概念名一致/互含者
    norm_new = [normalize_name(n) for n in new_names]
    norm_new = [x for x in norm_new if x]
    hit_idx = set()
    for u in unmatched:
        uk = normalize_name(u)
        if not uk:
            continue
        for i, nk in enumerate(norm_new):
            if i in hit_idx:
                continue
            if uk == nk or (len(nk) >= 2 and (nk in uk or uk in nk)):
                hit_idx.add(i)
                break
    open_rec = (len(hit_idx) / len(norm_new)) if norm_new else None
    return {"gold": gold_slots, "matched": matched, "unmatched": unmatched,
            "P": p, "R": r, "F1": f1(p, r), "tier": tier,
            "open_rec": open_rec, "n_new": len(norm_new)}


def run_heldout(args, gs):
    heldout_dir = Path(config.HELDOUT_DIR)
    results_dir = Path(args.dir) if args.dir else Path(config.HELDOUT_RESULTS_DIR)
    gold_by, new_by = load_heldout_gold(gs, heldout_dir)
    if not gold_by:
        sys.exit("[eval] heldout_gold_concepts.csv 为空或不存在：data/held_out/heldout_gold_concepts.csv")
    results = load_results(results_dir)
    if not results:
        sys.exit(f"[eval] No held-out results under {results_dir}/*.json. Run:"
                 f"python experiment/run_experiment.py --heldout --models deepseek,glm,qwen")
    groups = sorted({(m, s) for (m, s, _) in results})
    lines = ["# Held-out 独立集评测（人工文档级 gold）\n",
             "> gold 来自 `data/held_out/heldout_gold_concepts.csv`（人工标注 present 概念），"
             "**不再**使用 `anchor_slots` 子串匹配。旧 19 份同源结果为 in-corpus reference。\n"]
    table = []
    for (model, strategy) in groups:
        rows = [(m, s, doc) for (m, s, doc) in results if (m, s) == (model, strategy)]
        c_pairs, tier_acc = [], {t: [] for t in TIERS}
        open_recs, missing_gold = [], []
        for (_, _, doc) in rows:
            if doc not in gold_by:
                missing_gold.append(doc)
                continue
            hm = heldout_doc_metrics(gs, results[(model, strategy, doc)],
                                     gold_by[doc], new_by.get(doc, []))
            if hm["gold"]:
                c_pairs.append((doc, hm["P"], hm["R"], hm["F1"]))
            for t in TIERS:
                g_t, tp_t, rt = hm["tier"][t]
                if g_t:
                    tier_acc[t].append((doc, rt))
            if hm["open_rec"] is not None:
                open_recs.append(hm["open_rec"])
        mc = macro(c_pairs)
        print(f"\n== {model} / 策略{strategy}  (held-out 文档数={len(c_pairs)})")
        if missing_gold:
            print(f"   ⚠ 无人工标注 gold，跳过: {missing_gold}")
        print(f"  概念(文档级宏平均): P={fmt_num(mc[0])} R={fmt_num(mc[1])} F1={fmt_num(mc[2])}")
        tl = "  "
        for t in TIERS:
            rt = (sum(x[1] for x in tier_acc[t]) / len(tier_acc[t])) if tier_acc[t] else None
            tl += f"{t}: R={fmt_num(rt)}  "
        print(tl)
        if open_recs:
            orr = sum(open_recs) / len(open_recs)
            print(f"  开放集新概念召回: {orr:.3f} (参与文档 {len(open_recs)})")
        lines.append(f"## {model} / 策略{strategy}\n")
        lines.append(f"- 概念（文档级宏平均）：P={fmt_num(mc[0])} R={fmt_num(mc[1])} F1={fmt_num(mc[2])}")
        lines.append("- 分层 R：" + "  ".join(
            f"{t} {fmt_num(sum(x[1] for x in tier_acc[t]) / len(tier_acc[t]) if tier_acc[t] else None)}"
            for t in TIERS))
        if open_recs:
            lines.append(f"- 开放集新概念召回：{sum(open_recs) / len(open_recs):.3f}（参与文档 {len(open_recs)}）")
        row = {"model": model, "strategy": strategy,
               "concept_P": None if not mc else round(mc[0], 3),
               "concept_R": None if not mc else round(mc[1], 3),
               "concept_F1": None if not mc else round(mc[2], 3)}
        for t in TIERS:
            row[f"{t}_R"] = (round(sum(x[1] for x in tier_acc[t]) / len(tier_acc[t]), 3)
                             if tier_acc[t] else None)
        if open_recs:
            row["open_set_recall"] = round(sum(open_recs) / len(open_recs), 3)
        table.append(row)
    evdir = results_dir / "eval"
    evdir.mkdir(parents=True, exist_ok=True)
    (evdir / "heldout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if table:
        with open(evdir / "heldout_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    print(f"\n[eval] held-out 报告: {evdir / 'heldout_report.md'}")


if __name__ == "__main__":
    main()
