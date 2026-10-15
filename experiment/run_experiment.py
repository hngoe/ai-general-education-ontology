# -*- coding: utf-8 -*-
"""Main extraction experiment driver.

Matrix: model (DeepSeek/GLM/Qwen) × strategy (A/B) × corpus (19 syllabi; single / substring / all).
- temperature = 0; JSON parse + validation + repair retry (default ≤ 3, configurable via --max-repair).
- Results written to data/results/{model}_{strategy}_{doc}.json; manifest appended to data/results/manifest.jsonl.
- Resumable: existing results are skipped by default; --force overwrites.
- --dry-run uses a mock (no network); --mock-fail-rate forces invalid output to test the retry path.

示例：
  python experiment/run_experiment.py --dry-run --doc 合肥  --strategies A,B --models deepseek,glm,qwen
  python experiment/run_experiment.py --doc all   --strategies A,B
  python experiment/run_experiment.py --doc 01_浙江大学_人工智能基础A --strategy A --models qwen --force
"""
import argparse
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config
import llm_client
from llm_client import LLMClient, run_extraction

SYL_PREFIX = re.compile(r"^(01|02|03)_.*\.txt$")
# 辅语料排除：UNESCO 教师向框架仅用于对齐（角色过滤），不跑抽取
AUX_EXCLUDE = {"UNESCO_教师人工智能能力框架_中文版"}


def list_docs():
    docs = sorted(
        p.stem for p in config.RAW_DIR.glob("*.txt")
        if SYL_PREFIX.match(p.name)
    )
    return docs


def list_aux():
    """辅语料 = data/raw 顶层 txt 中非大纲、非档案的文件（政策/文献 9 份，另报用）。"""
    docs = sorted(
        p.stem for p in config.RAW_DIR.glob("*.txt")
        if not SYL_PREFIX.match(p.name)
        and not p.name.startswith("_")
        and p.stem not in AUX_EXCLUDE
    )
    return docs


def list_heldout():
    """held-out 独立验证集 = data/held_out 顶层 txt（不含 _ 开头档案）。"""
    docs = sorted(
        p.stem for p in config.HELDOUT_DIR.glob("*.txt")
        if not p.name.startswith("_")
    )
    return docs


def resolve_doc(arg):
    docs = list_docs()
    if arg in ("all", "*"):
        return docs
    hits = [d for d in docs if arg in d]
    if not hits:
        sys.exit(f"[run] No document matches {arg!r}. Options: {docs}")
    if len(hits) > 1:
        sys.exit(f"[run] {arg!r} matches multiple documents: {hits}; use the full name or a longer substring.")
    return hits


def load_doc_text(doc_id, doc_dir=None):
    d = doc_dir or config.RAW_DIR
    return (d / f"{doc_id}.txt").read_text(encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description="LLM 抽取实验（策略 A/B × 三模型 × 大纲语料）")
    ap.add_argument("--doc", default="all", help="大纲：all | 全名 | 唯一子串（如 合肥）")
    ap.add_argument("--aux", action="store_true", help="跑辅语料（政策/文献 9 份，与大纲分开报告）")
    ap.add_argument("--heldout", action="store_true", help="跑 held-out 独立集（data/held_out，结果写 data/results_heldout）")
    ap.add_argument("--out-dir", default="", help="结果输出目录（默认 data/results；辅料建议 data/results_aux）")
    ap.add_argument("--models", default="", help="逗号分隔，如 deepseek,glm,qwen；缺省=环境变量有 key 的模型")
    ap.add_argument("--strategies", default="A,B", help="逗号分隔，如 A 或 B 或 A,B")
    ap.add_argument("--dry-run", action="store_true", help="mock 模式，不联网")
    ap.add_argument("--mock-fail-rate", type=float, default=0.0, help="mock 首次输出非法概率(0~1)")
    ap.add_argument("--force", action="store_true", help="覆盖已存在结果")
    ap.add_argument("--max-repair", type=int, default=config.MAX_REPAIR)
    ap.add_argument("--sleep", type=float, default=config.SLEEP_BETWEEN)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or config.available_models()
    strategies = [s.strip().upper() for s in args.strategies.split(",") if s.strip()]
    if args.heldout:
        docs = list_heldout()
        RD = (Path(args.out_dir) if args.out_dir else config.HELDOUT_RESULTS_DIR)
        text_dir = config.HELDOUT_DIR
        if not docs:
            sys.exit("[run] No extractable txt files under data/held_out/; add held-out syllabi first.")
    else:
        docs = list_aux() if args.aux else resolve_doc(args.doc)
        RD = (Path(args.out_dir) if args.out_dir else config.RESULTS_DIR)
        text_dir = config.RAW_DIR
    import prompts as _prompts
    prompt_version = getattr(_prompts, "PROMPT_VERSION", "?")
    if not models:
        sys.exit("[run] No --models specified and no API key in environment;"
                 "set DEEPSEEK_API_KEY / ZHIPU_API_KEY / DASHSCOPE_API_KEY, or add --dry-run.")
    unknown = [m for m in models if m not in config.MODELS]
    if unknown:
        sys.exit(f"[run] 未知模型 {unknown}，可选：{list(config.MODELS)}")

    RD.mkdir(parents=True, exist_ok=True)
    manifest = RD / "manifest.jsonl"
    total_calls = 0
    summary = {"n_combo": 0, "ok": 0, "partial": 0, "invalid": 0, "error": 0,
               "prompt_tokens": 0, "completion_tokens": 0}

    combos = [(d, m, s) for d in docs for m in models for s in strategies]
    corpus_tag = "held-out" if args.heldout else ("辅语料" if args.aux else "大纲")
    print(f"[run] 组合数={len(combos)} ({corpus_tag} {len(docs)} 份, models={models}, strategies={strategies})")
    print(f"[run] dry_run={args.dry_run}  sleep={args.sleep}s  max_repair={args.max_repair}")

    clients = {}
    t0 = time.time()
    for combo_idx, (doc, model_key, strategy) in enumerate(combos, 1):
        base = f"{model_key}_{strategy}_{doc}"
        out = RD / f"{base}.json"
        err_out = RD / f"{base}.error.json"
        if out.exists() and not args.force:
            try:
                _prev = json.loads(out.read_text(encoding="utf-8"))
            except Exception:
                _prev = {}
            # 仅当上次状态为 ok / repaired_partial 才算完成（invalid_after_retry 需重跑）
            if _prev.get("status") in ("ok", "repaired_partial"):
                print(f"[run] {combo_idx}/{len(combos)} 跳过（已完成）: {out.name}")
                continue
        try:
            text = load_doc_text(doc, text_dir)
        except Exception as e:
            print(f"[run] {combo_idx}/{len(combos)} {doc}: 读取失败 {e}")
            continue
        cli = clients.get((model_key, args.dry_run))
        if cli is None:
            cli = LLMClient(model_key, dry_run=args.dry_run, mock_fail_rate=args.mock_fail_rate)
            clients[(model_key, args.dry_run)] = cli
        rec = {"model": model_key, "strategy": strategy, "doc_id": doc,
               "doc_chars": len(text), "temperature": config.TEMPERATURE,
               "prompt_version": prompt_version}
        try:
            if not args.dry_run and combo_idx > 1:
                time.sleep(args.sleep)
            res = run_extraction(cli, strategy, text, max_repair=args.max_repair)
            rec.update(res)
            status = res["status"]
            total_calls += 1
            if status == "ok":
                summary["ok"] += 1
            elif status == "repaired_partial":
                summary["partial"] += 1
            else:
                summary["invalid"] += 1
            summary["prompt_tokens"] += res["usage"].get("prompt_tokens", 0)
            summary["completion_tokens"] += res["usage"].get("completion_tokens", 0)
            rec["ts"] = datetime.now(timezone.utc).isoformat()
            rec["status"] = status
            rec["repairs"] = res["repairs"]
            rec["transport_retries"] = res["transport_retries"]
            rec["n_concepts"] = len(res["concepts"])
            rec["n_relations"] = len(res["relations"])
            rec.pop("usage", None)  # usage 单独存
            with open(out, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "path": out.name, "usage": res["usage"]},
                                   ensure_ascii=False) + "\n")
            print(f"[run] {combo_idx}/{len(combos)} OK  {out.name}  "
                  f"status={status} concepts={len(res['concepts'])} "
                  f"relations={len(res['relations'])} repairs={res['repairs']} "
                  f"tok={res['usage'].get('prompt_tokens', 0)}+{res['usage'].get('completion_tokens', 0)}")
        except Exception as e:
            summary["error"] += 1
            rec["status"] = "error"
            rec["error"] = str(e)[:500]
            with open(err_out, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            print(f"[run] {combo_idx}/{len(combos)} ERROR {doc} [{model_key}/{strategy}]: {e}")

    elapsed = time.time() - t0
    print("\n[run] 汇总: " + json.dumps(summary, ensure_ascii=False))
    print(f"[run] 耗时 {elapsed:.0f}s；结果目录: {RD}")


if __name__ == "__main__":
    main()
