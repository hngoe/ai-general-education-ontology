# -*- coding: utf-8 -*-
"""LLM client: OpenAI-compatible chat/completions transport + JSON parse / validate / repair-retry orchestration.

- Real calls: requests POST, bearer auth, temperature = 0; HTTP/network errors retry with BACKOFF.
- dry_run (mock): no network; returns valid JSON built from generic terms matched in the syllabus text;
  with mock_fail_rate the first output can be forced invalid to exercise the repair path.
- Repair retry: on parse/validation failure the previous output and error reason are fed back as a new
  user message, up to MAX_REPAIR times; if it still fails, keep the object as partial when parseable
  (drop invalid relations), otherwise mark invalid.
"""
import json
import re
import time

import config

try:
    import requests
except ImportError:  # 仅 dry-run 也需要？requests 已装；此处防御
    requests = None


# ---------------------------------------------------------------- JSON 解析

def parse_json(content: str):
    """宽容解析：容忍 ```json 围栏 / 前后缀文字 / 首尾空白。失败抛 ValueError。"""
    if not content:
        raise ValueError("空输出")
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip().lstrip("\ufeff")
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if 0 <= a < b:
            return json.loads(s[a:b + 1])
        raise ValueError("找不到合法 JSON 对象")


# ---------------------------------------------------------------- 传输层

class LLMClient:
    def __init__(self, model_key: str, dry_run: bool = False, mock_fail_rate: float = 0.0):
        self.key = model_key
        self.cfg = config.MODELS[model_key]
        self.dry_run = dry_run
        self.mock_fail_rate = mock_fail_rate
        self.transport_retries = 0  # 累计（跨调用），供报告

    # ---- mock ----
    def _mock_complete(self, messages):
        # 依任务文本推断应返回的输出形状（仅管线冒烟，非真实抽取）
        text = messages[-1]["content"]
        terms = [t for t in config.MOCK_TERMS if t in text][:12]
        concepts = [{"id": f"C{i + 1:02d}", "name": t, "definition": "（mock 定义）"}
                    for i, t in enumerate(terms)]
        if not concepts:
            concepts = [{"id": "C01", "name": "人工智能", "definition": "（mock 定义）"}]
        if "【非层次关系】" in text:            # 策略 B R3
            obj = {"relations": []}
        elif "【层次关系】" in text:            # 策略 B R2
            obj = {"relations": []}
        elif "四类关系" in text or '"relations"' in text:  # 策略 A
            obj = {"concepts": concepts, "relations": []}
        else:                                  # 策略 B R1
            obj = {"concepts": concepts}
        content = json.dumps(obj, ensure_ascii=False, indent=1)
        if self.mock_fail_rate > 0:
            import random
            if random.random() < self.mock_fail_rate:
                content = '这不是 JSON：```\n' + content[:40] + '...'  # 故意非法
        usage = {
            "prompt_tokens": sum(len(str(m.get("content", ""))) for m in messages),
            "completion_tokens": len(content),
        }
        return content, usage

    # ---- 真实调用 ----
    def _real_complete(self, messages):
        if requests is None:
            raise RuntimeError("requests is not installed; use --dry-run or install requests first")
        key = config.api_key(self.key)
        if not key:
            raise RuntimeError(
                f"Missing API key for {self.cfg['model']}: set environment variable {self.cfg['env']}"
                f"{self.cfg.get('env2', '')} and retry, or use --dry-run.")
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "temperature": config.TEMPERATURE,
            "max_tokens": self.cfg["max_output"],
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        last_err = None
        for attempt in range(config.TRANSPORT_RETRY + 1):
            try:
                r = requests.post(self.cfg["base_url"], json=payload, headers=headers,
                                  timeout=config.TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    self.transport_retries += attempt
                    return content, usage
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            except requests.exceptions.RequestException as e:
                last_err = f"网络异常: {e}"
            if attempt < config.TRANSPORT_RETRY:
                time.sleep(config.BACKOFF[min(attempt, len(config.BACKOFF) - 1)])
        raise RuntimeError(f"传输失败（已重试 {config.TRANSPORT_RETRY} 次）: {last_err}")

    def complete(self, messages):
        if self.dry_run:
            return self._mock_complete(messages)
        return self._real_complete(messages)


# ---------------------------------------------------------------- 抽取编排

def run_extraction(client: LLMClient, strategy: str, doc_text: str, max_repair: int):
    """对一个文档执行策略 A 或 B；返回结构化结果。

    返回 dict：{status, concepts, relations, rounds, repairs, transport_retries,
                raw_rounds, usage:{prompt_tokens, completion_tokens}}
    status: ok | repaired_partial | invalid_after_retry
    """
    import prompts
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    total_repairs = 0

    def _tally(u):
        usage["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        usage["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)

    def _loop(messages, validate):
        """单轮消息序列：解析+校验+修复，返回 (obj, repairs, raw_last, status)"""
        nonlocal total_repairs
        last_content = ""
        for attempt in range(1 + max_repair):
            content, u = client.complete(messages)
            _tally(u)
            last_content = content
            try:
                obj = parse_json(content)
            except ValueError as e:
                if attempt < max_repair:
                    hint = ("（若上次输出因过长被截断，请压缩概念数量、精简 definition 后重试）"
                            if attempt == 0 else "")
                    messages += [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": f"你上次的输出不是合法 JSON（{e}）。"
                                                    "请只输出符合要求的 JSON，不要任何解释或代码块标记。" + hint},
                    ]
                    total_repairs += 1
                    continue
                return None, total_repairs, last_content, "invalid_after_retry"
            errs = validate(obj)
            if not errs:
                return obj, total_repairs, last_content, "ok"
            if attempt < max_repair:
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"你上次输出 JSON 合法但不符合约束，问题："
                                                f"{'；'.join(errs[:4])}。请修正后只输出符合约束的 JSON。"},
                ]
                total_repairs += 1
                continue
            # 修复用尽：可解析则保留，剔除非法 relations
            return obj, total_repairs, last_content, "repaired_partial"

    def _finalize_rel(obj, id_set, allowed):
        if obj is None:
            return []
        rels = obj.get("relations")
        if not isinstance(rels, list):
            return []
        good = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            if r.get("type") in allowed and r.get("source") in id_set and r.get("target") in id_set:
                good.append(r)
        return good

    rounds = []
    raw_rounds = []
    if strategy == "A":
        obj, repairs, raw, status = _loop(
            prompts.prompt_A(doc_text, config.MAX_CONCEPTS["A"]), prompts.validate_A)
        raw_rounds.append(raw)
        if obj is None:
            return {"status": "invalid_after_retry", "concepts": [], "relations": [],
                    "rounds": rounds, "repairs": repairs, "raw_rounds": raw_rounds,
                    "transport_retries": client.transport_retries, "usage": usage}
        concepts = prompts.dedupe_concepts(obj.get("concepts", []))
        ids = {c["id"] for c in concepts}
        rels = _finalize_rel(obj, ids, prompts.REL_TYPES)
        rounds.append({"round": "A", "status": status, "n_concepts": len(concepts),
                       "n_relations": len(rels)})
        return {"status": status, "concepts": concepts, "relations": rels,
                "rounds": rounds, "repairs": repairs, "raw_rounds": raw_rounds,
                "transport_retries": client.transport_retries, "usage": usage}

    # ---- 策略 B：三轮 ----
    # R1 概念
    obj1, repairs, raw1, status1 = _loop(
        prompts.prompt_B1(doc_text, config.MAX_CONCEPTS["B"]), prompts.validate_B1)
    raw_rounds.append(raw1)
    if obj1 is None:
        return {"status": "invalid_after_retry", "concepts": [], "relations": [],
                "rounds": rounds, "repairs": repairs, "raw_rounds": raw_rounds,
                "transport_retries": client.transport_retries, "usage": usage}
    concepts = prompts.dedupe_concepts(obj1.get("concepts", []))
    ids = {c["id"] for c in concepts}
    rounds.append({"round": "R1_concepts", "status": status1, "n_concepts": len(concepts)})
    rels_all = []

    # R2 层次关系
    from functools import partial as _partial
    concepts_obj = {"concepts": concepts}
    obj2, repairs, raw2, status2 = _loop(
        prompts.prompt_B2(doc_text, concepts_obj),
        _partial(prompts.validate_B2, concepts_obj=concepts_obj))
    raw_rounds.append(raw2)
    rels2 = _finalize_rel(obj2, ids, {"is-a", "part-of"}) if obj2 is not None else []
    rounds.append({"round": "R2_hier", "status": status2, "n_relations": len(rels2)})
    rels_all.extend(rels2)

    # R3 非层次关系
    obj3, repairs, raw3, status3 = _loop(
        prompts.prompt_B3(doc_text, concepts_obj, {"relations": rels2}),
        _partial(prompts.validate_B3, concepts_obj=concepts_obj))
    raw_rounds.append(raw3)
    rels3 = _finalize_rel(obj3, ids, {"prerequisite", "applied-in"}) if obj3 is not None else []
    rounds.append({"round": "R3_nonhier", "status": status3, "n_relations": len(rels3)})
    rels_all.extend(rels3)

    worst = "ok"
    if any(r["status"] == "invalid_after_retry" for r in rounds):
        worst = "invalid_after_retry"
    elif any(r["status"] == "repaired_partial" for r in rounds):
        worst = "repaired_partial"

    return {"status": worst, "concepts": concepts, "relations": rels_all,
            "rounds": rounds, "repairs": repairs, "raw_rounds": raw_rounds,
            "transport_retries": client.transport_retries, "usage": usage}
