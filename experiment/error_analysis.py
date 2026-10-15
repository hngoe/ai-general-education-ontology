# -*- coding: utf-8 -*-
"""Error analysis: Top-N missed (FN) / spurious (FP) concepts per (model, strategy) across 19 syllabi.

- FN: a GS slot in the document's gold anchor set that was not extracted → ranked by number of documents missed.
- FP（标准口径）：预测概念名 ① 未命中任何 GS 槽位，或 ② 命中的槽位不在该文档锚定集
  （即提取了该文档未出现/本不该算的 GS 概念）；跨文档按出现次数聚合。

输出：data/results/eval/error_topN.md（按 model_strategy 分节）。
用法：python experiment/error_analysis.py [--top 10] [--models deepseek,glm,qwen]
"""
import argparse
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config
from normalize import GsData

RESULT_RE = re.compile(r"^(deepseek|glm|qwen)_(A|B)_.+\.json$")


def load_ok_results(base):
    out = {}
    for p in sorted(base.glob("*.json")):
        if not RESULT_RE.match(p.name):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") not in ("ok", "repaired_partial"):
            continue
        out[(d["model"], d["strategy"], d["doc_id"])] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--models", default="", help="逗号分隔；缺省全部")
    ap.add_argument("--dir", default="")
    args = ap.parse_args()
    only = {m.strip() for m in args.models.split(",") if m.strip()}
    base = Path(args.dir) if args.dir else config.RESULTS_DIR
    gs = GsData()
    results = load_ok_results(base)
    groups = sorted({(m, s) for (m, s, _) in results if not only or m in only})

    lines = ["# Top-N 漏提 / 误提 错误分析（跨文档聚合）\n"]
    for (model, strategy) in groups:
        keys = [(m, s, doc) for (m, s, doc) in results if (m, s) == (model, strategy)]
        fn_cnt = Counter()      # slot_id -> 漏提文档数
        fn_tier = {}
        fp_cnt = Counter()      # 预测名 -> 文档出现次数（FP）
        n_doc = len(keys)
        for (_, _, doc) in keys:
            text = (config.RAW_DIR / f"{doc}.txt").read_text(encoding="utf-8-sig")
            res = results[(model, strategy, doc)]
            gold = gs.anchor_slots(text)
            matched, seen = set(), set()
            for c in res.get("concepts", []):
                name = str(c.get("name", "")).strip()
                sid = gs.match_mention(name)
                if sid is not None:
                    matched.add(sid)
                    if sid not in gold:
                        fp_cnt[(name, sid)] += 1
                else:
                    if name and name.lower() not in seen:
                        seen.add(name.lower())
                        fp_cnt[(name, None)] += 1
            for sid in gold - matched:
                fn_cnt[sid] += 1
                fn_tier[sid] = gs.slot_by_id[sid]["tier"]
        # 汇总输出
        lines.append(f"\n## {model} / 策略{strategy}（{n_doc} 份）\n")
        lines.append(f"### Top-{args.top} 漏提（FN）")
        lines.append("| GS概念 | 分层 | 漏提文档数 |")
        lines.append("|---|---|---|")
        for sid, cnt in fn_cnt.most_common(args.top):
            s = gs.slot_by_id[sid]
            lines.append(f"| {s['canonical']} | {s['tier']} | {cnt} |")
        lines.append(f"\n### Top-{args.top} 误提（FP）")
        lines.append("| 提取概念 | 命中的GS槽位 | 文档出现次数 |")
        lines.append("|---|---|---|")
        for (name, sid), cnt in fp_cnt.most_common(args.top):
            slot = gs.slot_by_id[sid]["canonical"] if sid else "（未命中GS）"
            lines.append(f"| {name} | {slot} | {cnt} |")

    out_path = base / "eval" / "error_topN.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[error] 报告 -> {out_path}")
    print("\n".join(lines[:80]))


if __name__ == "__main__":
    main()
