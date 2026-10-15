# -*- coding: utf-8 -*-
"""Statistical analysis: bootstrap 95% CI + paired-difference CI + effect sizes + per-family Holm correction.

Comparison families:
  F1 概念 A vs B（3 个模型）
  F2 is-a F1 A vs B（3 个模型）
  F3 Core vs Peripheral（6 组合）
对每族内的原始 p（配对 t 检验）做 Holm-Bonferroni 校正。
输出：data/results/eval/statistics_ci.md
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

import config
import eval_quick  # 导入即完成 stdout UTF-8 包装，勿再包装
from normalize import GsData

RESULT_RE = re.compile(r"^(deepseek|glm|qwen)_(A|B)_.+\.json$")
MODELS = ["deepseek", "glm", "qwen"]
RNG = np.random.default_rng(42)
BOOT = 20000


def ci_mean(x, alpha=0.05):
    x = np.asarray(x, float)
    means = np.array([np.mean(RNG.choice(x, size=len(x), replace=True)) for _ in range(BOOT)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def ci_paired_diff(x, y, alpha=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    ds = np.array([np.mean(RNG.choice(d, size=len(d), replace=True)) for _ in range(BOOT)])
    lo, hi = np.percentile(ds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def holm(ps):
    """Holm-Bonferroni 校正，返回校正后 p 列表（保持原顺序）。"""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    k = len(ps)
    prev = 0.0
    for rank, i in enumerate(order):
        v = max((k - rank) * ps[i], prev)
        adj[i] = v
        prev = v
    return adj


def main():
    gs = GsData()
    results = {}
    for p in sorted(config.RESULTS_DIR.glob("*.json")):
        if not RESULT_RE.match(p.name):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") not in ("ok", "repaired_partial"):
            continue
        results[(d["model"], d["strategy"], d["doc_id"])] = d
    docs = sorted({k[2] for k in results})
    per = {}
    for (m, s, doc), d in results.items():
        text = (config.RAW_DIR / f"{doc}.txt").read_text(encoding="utf-8-sig")
        cm = eval_quick.doc_concept_metrics(gs, d, text)
        rm = eval_quick.doc_rel_metrics(gs, d, text)
        per[(m, s, doc)] = {"cf1": cm["F1"], "tier": cm["tier"], "is_a": rm["is-a"]}

    def get_tier_f1(p, t):
        if not p:
            return None
        g_t, pt_t, tp_t, prec, rec = p["tier"][t]
        return eval_quick.f1(prec, rec) if g_t else None

    L = ["# 统计补强：bootstrap 95%CI / 配对差 / 效应量 / Holm 校正\n",
         f"> bootstrap n={BOOT}（种子 42）；配对单元=同一文档（n≤19）；双侧；"
         "按族做 Holm 校正：F1 概念 A vs B（3 模型）、F2 is-a A vs B（3 模型）、F3 Core vs Peripheral（6 组合）。\n"]

    def row(name, m_a, m_b, x, y):
        lo, hi = ci_mean(x)
        lo_d, hi_d = ci_paired_diff(x, y)
        t, p = stats.ttest_rel(x, y)
        d = np.asarray(x, float) - np.asarray(y, float)
        dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
        w, pw = (np.nan, np.nan)
        try:
            w, pw = stats.wilcoxon(x, y)
        except ValueError:
            pass
        L.append(f"- **{name}**：均值 {np.mean(x):.3f}（A） vs {np.mean(y):.3f}（B）"
                 f"；mean_A 95%CI=[{lo:.3f},{hi:.3f}]；配对差 Δ95%CI=[{lo_d:+.3f},{hi_d:+.3f}]；"
                 f"t={t:.2f} p_t={p:.4f}；Wilcoxon p={pw:.4f}；dz={dz:.2f}")

    # 收集原始 p 用于 Holm
    fam = {}
    ab_cf = []
    ab_isa = []
    coreper = []
    for m in MODELS:
        common = [doc for doc in docs if (m, "A", doc) in per and (m, "B", doc) in per]
        x = [per[(m, "A", d)]["cf1"] for d in common]
        y = [per[(m, "B", d)]["cf1"] for d in common]
        if len(x) > 1:
            t, p = stats.ttest_rel(x, y)
            ab_cf.append((m, p))
    for m in MODELS:
        xs, ys = [], []
        for d in docs:
            a, b = per.get((m, "A", d)), per.get((m, "B", d))
            if not a or not b or a["is_a"] is None or b["is_a"] is None:
                continue
            xs.append(a["is_a"][5]); ys.append(b["is_a"][5])
        if len(xs) > 1:
            t, p = stats.ttest_rel(xs, ys)
            ab_isa.append((m, p))
    for m in MODELS:
        for s in ("A", "B"):
            xs, ys = [], []
            for d in docs:
                p = per.get((m, s, d))
                if not p:
                    continue
                c, q = get_tier_f1(p, "Core"), get_tier_f1(p, "Peripheral")
                if c is None or q is None:
                    continue
                xs.append(c); ys.append(q)
            if len(xs) > 1:
                t, p = stats.ttest_rel(xs, ys)
                coreper.append(((m, s), p))
    holm_map = {}
    for key, ps in (("F1概念", ab_cf), ("F2is-a", ab_isa), ("F3CorePer", coreper)):
        if ps:
            vals = [p for _, p in ps]
            adj = holm(vals)
            for (idx, p_) in enumerate(ps):
                holm_map[(key, p_[0] if not isinstance(p_[0], tuple) else p_[0])] = adj[idx]

    # 输出主体表
    L.append("\n## F1 概念 A vs B（每模型）")
    for m in MODELS:
        common = [doc for doc in docs if (m, "A", doc) in per and (m, "B", doc) in per]
        x = [per[(m, "A", d)]["cf1"] for d in common]
        y = [per[(m, "B", d)]["cf1"] for d in common]
        if len(x) > 1:
            lo, hi = ci_mean(x); lo_d, hi_d = ci_paired_diff(x, y)
            t, p = stats.ttest_rel(x, y)
            d = np.asarray(x) - np.asarray(y)
            dz = d.mean() / d.std(ddof=1)
            w, pw = np.nan, np.nan
            try:
                w, pw = stats.wilcoxon(x, y)
            except ValueError:
                pass
            ph = holm_map.get(("F1概念", m), np.nan)
            L.append(f"- {m}：A={np.mean(x):.3f} B={np.mean(y):.3f}（Δ95%CI=[{lo_d:+.3f},{hi_d:+.3f}]），"
                     f"p_t={p:.4f}（Holm p={ph:.4f}），Wilcoxon p={pw:.4f}，dz={dz:.2f}，n={len(x)}")

    L.append("\n## F2 is-a F1 A vs B")
    for m in MODELS:
        xs, ys = [], []
        for d in docs:
            a, b = per.get((m, "A", d)), per.get((m, "B", d))
            if not a or not b or a["is_a"] is None or b["is_a"] is None:
                continue
            xs.append(a["is_a"][5]); ys.append(b["is_a"][5])
        if len(xs) > 1:
            lo_d, hi_d = ci_paired_diff(xs, ys)
            t, p = stats.ttest_rel(xs, ys)
            d = np.asarray(xs) - np.asarray(ys)
            dz = d.mean() / d.std(ddof=1)
            w, pw = np.nan, np.nan
            try:
                w, pw = stats.wilcoxon(xs, ys)
            except ValueError:
                pass
            ph = holm_map.get(("F2is-a", m), np.nan)
            L.append(f"- {m}：A={np.mean(xs):.3f} B={np.mean(ys):.3f}（Δ95%CI=[{lo_d:+.3f},{hi_d:+.3f}]），"
                     f"p_t={p:.4f}（Holm p={ph:.4f}），Wilcoxon p={pw:.4f}，dz={dz:.2f}，n={len(xs)}")

    L.append("\n## F3 Core vs Peripheral（同策略内）")
    for m in MODELS:
        for s in ("A", "B"):
            xs, ys = [], []
            for d in docs:
                p = per.get((m, s, d))
                if not p:
                    continue
                c, q = get_tier_f1(p, "Core"), get_tier_f1(p, "Peripheral")
                if c is None or q is None:
                    continue
                xs.append(c); ys.append(q)
            if len(xs) > 1:
                lo, hi = ci_mean(xs); lo_d, hi_d = ci_paired_diff(xs, ys)
                t, p = stats.ttest_rel(xs, ys)
                d = np.asarray(xs) - np.asarray(ys)
                dz = d.mean() / d.std(ddof=1)
                w, pw = np.nan, np.nan
                try:
                    w, pw = stats.wilcoxon(xs, ys)
                except ValueError:
                    pass
                ph = holm_map.get(("F3CorePer", (m, s)), np.nan)
                L.append(f"- {m}-{s}：Core={np.mean(xs):.3f} Per={np.mean(ys):.3f}（Δ95%CI=[{lo_d:+.3f},{hi_d:+.3f}]），"
                         f"p_t={p:.4f}（Holm p={ph:.4f}），Wilcoxon p={pw:.4f}，dz={dz:.2f}，n={len(xs)}")

    L.append("\n> 解读建议：F1 中 Qwen A>B 在 Holm 校正后仍显著则强结论；F3 全组合显著。"
             "报告时给 Δ95%CI 与 dz，p 注明未校正/Holm 两列，避免多重比较质疑。")
    out = config.RESULTS_DIR / "eval" / "statistics_ci.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ci] -> {out}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
