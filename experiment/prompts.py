# -*- coding: utf-8 -*-
"""Prompt templates and JSON validators for the LLM concept/relation extraction experiment.

Strategy A: one-shot — a single call outputs both concepts and relations (four types).
Strategy B: decomposed multi-round — R1 concepts, R2 is-a/part-of, R3 prerequisite/applied-in.

Design notes:
- Chinese-language task; prompts instruct the model to rely only on the syllabus text, and no
  gold-standard content is injected (leakage prevention).
- No reliance on vendor-specific response_format=json_object; strict instructions + parse + repair retry.
- The output schema is given as a code block; the parser tolerates ```json fences and leading prose.
"""

# ---------------------------------------------------------------- 共享指令

SYS_TEXT = """你是一位知识工程与本体论专家，正在协助构建“中国高校人工智能（AI）通识教育”领域本体。
你的任务：仅依据下方提供的【课程大纲文本】，抽取与 AI 通识教育直接相关的核心概念与概念间关系。

抽取纪律：
1. 概念应是文本中实际出现、或由文本内容明确支撑、可作为独立知识点讲授的单元；宁缺毋滥。
2. 忽略无关内容：课程安排/学时分配/考核方式/教学平台与导航语/教材版本信息/人名与机构名等。
3. 概念名称用中文教学常用叫法；同义不同名只保留一个最常用规范名，不要制造重复条目。
4. 排除过细条目（如单个函数名、某次作业名）与过泛条目（如“数学基础”“学习”“能力”）——除非文本明确把它列为 AI 通识知识点。
5. 只输出指定 JSON，不要输出任何解释、前后缀或 Markdown 代码块标记。"""

REL_TAXONOMY = """关系类型定义（只允许这 4 类）：
- is-a：A 是 B 的一种/子类。例：深度学习 is-a 机器学习。
- part-of：A 是 B 的组成部分。例：数据标注 part-of 监督学习流程。
- prerequisite：A 是 B 的先修/前提知识。例：线性代数 prerequisite 机器学习。
- applied-in：A 应用于 B。例：计算机视觉 applied-in 人脸识别。"""

REL_TYPES = ("is-a", "part-of", "prerequisite", "applied-in")

PROMPT_VERSION = "v0.2"   # version recorded with the result files

# 概念粒度与去噪（A 与 B1 共用）
CONCEPT_CLEAN = """概念去噪与粒度：
- 具体产品/模型品牌名（如 DeepSeek、豆包、Kimi、CivilGPT）若仅作为教学举例出现，不作为独立概念；
- 同一知识点的紧密细分变体（如同一模糊推理算法的 三I/泛三I/泛五I 等实现）若文本未分别单列为独立讲授单元，只收一个代表性条目；
- 只收可作为独立讲授单元的课程级知识点/术语。"""

# 关系纪律（A 与 B 各轮共用）
REL_DISCIPLINE = """关系纪律：
- 同一条语义只输出一次：同一对概念只保留一条“类型最恰当、证据最充分”的关系；
- evidence 必须是【原文逐字】短引（≤40 字）；无法逐字引用时才写 推断：依据简述；
- 宁缺毋滥：无法从原文确定类型或找到支撑的关系不要输出；全篇只收录确有语义支撑的核心关系。"""

def _cap(max_concepts: int) -> str:
    return (
        f"概念总数请控制在 {max_concepts} 条以内，优先保留最重要的核心概念；"
        "若文本概念确实少于上限，按实际数量输出即可。"
    )

# ---------------------------------------------------------------- 策略 A

def prompt_A(doc_text: str, max_concepts: int = 80):
    """one-shot：概念 + 四型关系一次输出。返回 messages 列表。"""
    user = f"""{REL_TAXONOMY}

{CONCEPT_CLEAN}

【任务】从课程大纲文本中抽取全部核心概念，并在概念之间识别有文本证据支撑的四类关系。{_cap(max_concepts)}全篇关系建议控制在 10~50 条量级，只收核心关系。

严格按如下 JSON 结构输出（除 JSON 外不要任何文字）：
```json
{{
  "concepts": [
    {{"id": "C01", "name": "概念名", "definition": "一句话定义（≤40字）"}}
  ],
  "relations": [
    {{"type": "is-a", "source": "C01", "target": "C02", "evidence": "原文短引（≤40字）或 推断：原因"}}
  ]
}}
```
规则：
- concept.id 唯一且形如 C01、C02……；
- relation 的 source/target 必须引用 concepts 中已存在的 id；
- relation.type 只能是 is-a / part-of / prerequisite / applied-in 之一；
- {REL_DISCIPLINE}
- 没有把握的关系不要编造。

【课程大纲文本】
{doc_text}"""
    return [{"role": "system", "content": SYS_TEXT},
            {"role": "user", "content": user}]


def validate_A(d) -> list:
    """返回错误信息列表；空列表=通过。"""
    errs = []
    if not isinstance(d, dict):
        return ["顶层不是 JSON 对象"]
    concepts = d.get("concepts")
    relations = d.get("relations")
    if not isinstance(concepts, list) or not isinstance(relations, list):
        return ["缺少 concepts 或 relations 数组"]
    ids, names = set(), set()
    for i, c in enumerate(concepts):
        if not isinstance(c, dict):
            errs.append(f"concepts[{i}] 不是对象"); continue
        cid = str(c.get("id", "")).strip()
        name = str(c.get("name", "")).strip()
        if not cid:
            errs.append(f"concepts[{i}] 缺 id")
        elif cid in ids:
            errs.append(f"concepts[{i}] id 重复: {cid}")
        else:
            ids.add(cid)
        if not name:
            errs.append(f"concepts[{i}] 缺 name")
        elif name in names:
            errs.append(f"concepts[{i}] name 重复: {name}")
        else:
            names.add(name)
    for j, r in enumerate(relations):
        if not isinstance(r, dict):
            errs.append(f"relations[{j}] 不是对象"); continue
        rt = str(r.get("type", "")).strip()
        s, t = str(r.get("source", "")).strip(), str(r.get("target", "")).strip()
        if rt not in REL_TYPES:
            errs.append(f"relations[{j}] type 非法: {rt}")
        if s not in ids:
            errs.append(f"relations[{j}] source 引用不存在的 id: {s}")
        if t not in ids:
            errs.append(f"relations[{j}] target 引用不存在的 id: {t}")
        if s and t and s == t:
            errs.append(f"relations[{j}] source==target 自环: {s}")
    return errs


# ---------------------------------------------------------------- 策略 B

def prompt_B1(doc_text: str, max_concepts: int = 60):
    """R1：只抽概念。返回 messages。"""
    user = f"""{CONCEPT_CLEAN}

【任务】从课程大纲文本中抽取全部核心概念。{_cap(max_concepts)}

严格按如下 JSON 结构输出（除 JSON 外不要任何文字）：
```json
{{
  "concepts": [
    {{"id": "C01", "name": "概念名", "definition": "一句话定义（≤40字）"}}
  ]
}}
```
规则：id 唯一且形如 C01……；name 不重复；definition 一句话即可（≤40 字）。

【课程大纲文本】
{doc_text}"""
    return [{"role": "system", "content": SYS_TEXT},
            {"role": "user", "content": user}]


def validate_B1(d) -> list:
    errs = []
    if not isinstance(d, dict):
        return ["顶层不是 JSON 对象"]
    concepts = d.get("concepts")
    if not isinstance(concepts, list):
        return ["缺少 concepts 数组"]
    ids, names = set(), set()
    for i, c in enumerate(concepts):
        if not isinstance(c, dict):
            errs.append(f"concepts[{i}] 不是对象"); continue
        cid = str(c.get("id", "")).strip()
        name = str(c.get("name", "")).strip()
        if not cid:
            errs.append(f"concepts[{i}] 缺 id")
        elif cid in ids:
            errs.append(f"concepts[{i}] id 重复: {cid}")
        else:
            ids.add(cid)
        if not name:
            errs.append(f"concepts[{i}] 缺 name")
        elif name in names:
            errs.append(f"concepts[{i}] name 重复: {name}")
        else:
            names.add(name)
    return errs


def _rels_user(doc_text, concept_json, r2_rel_json, types_allowed, task_desc):
    parts = [f"{task_desc}。请只基于大纲文本判断，不要编造。",
             "第一轮已抽取的概念如下（id 即本轮可引用的全部概念）：",
             concept_json]
    if r2_rel_json:
        parts += ["第二轮已识别的层次关系如下，本轮不得重复输出这些关系：", r2_rel_json]
    parts += [f"关系类型只允许：{types_allowed}。",
              "严格按如下 JSON 结构输出（除 JSON 外不要任何文字）：",
              "```json\n{\n  \"relations\": [\n    {\"type\": \""
              + types_allowed.split("/")[0]
              + "\", \"source\": \"C01\", \"target\": \"C02\", "
                "\"evidence\": \"原文短引（≤40字）或 推断：原因\"}\n  ]\n}\n```",
              f"规则：type 只能是 {types_allowed} 之一；source/target 必须来自上面概念列表的 id；",
              REL_DISCIPLINE,
              "建议本回合只输出确有支撑的核心关系（通常 5~40 条），宁缺毋滥。",
              "【课程大纲文本】", doc_text]
    return "\n\n".join(parts)


def prompt_B2(doc_text: str, concepts_obj) -> list:
    """R2：层次关系 is-a / part-of。concepts_obj 为 R1 校验通过的对象。"""
    import json as _json
    concept_json = _json.dumps(concepts_obj, ensure_ascii=False, indent=1)
    user = _rels_user(doc_text, concept_json, "", "is-a/part-of",
                      "【任务】在已抽取的概念之间识别【层次关系】is-a 与 part-of")
    return [{"role": "system", "content": SYS_TEXT},
            {"role": "user", "content": user}]


def prompt_B3(doc_text: str, concepts_obj, r2_rel_obj) -> list:
    """R3：非层次关系 prerequisite / applied-in。"""
    import json as _json
    concept_json = _json.dumps(concepts_obj, ensure_ascii=False, indent=1)
    r2_json = _json.dumps(r2_rel_obj, ensure_ascii=False, indent=1)
    user = _rels_user(doc_text, concept_json, r2_json, "prerequisite/applied-in",
                      "【任务】在已抽取的概念之间识别【非层次关系】prerequisite 与 applied-in")
    return [{"role": "system", "content": SYS_TEXT},
            {"role": "user", "content": user}]


def _validate_rels(d, id_set, allowed_types) -> list:
    errs = []
    if not isinstance(d, dict):
        return ["顶层不是 JSON 对象"]
    rels = d.get("relations")
    if not isinstance(rels, list):
        return ["缺少 relations 数组"]
    for j, r in enumerate(rels):
        if not isinstance(r, dict):
            errs.append(f"relations[{j}] 不是对象"); continue
        rt = str(r.get("type", "")).strip()
        s, t = str(r.get("source", "")).strip(), str(r.get("target", "")).strip()
        if rt not in allowed_types:
            errs.append(f"relations[{j}] type 非法（应为 {allowed_types}）: {rt}")
        if s not in id_set:
            errs.append(f"relations[{j}] source 不在概念列表: {s}")
        if t not in id_set:
            errs.append(f"relations[{j}] target 不在概念列表: {t}")
        if s and t and s == t:
            errs.append(f"relations[{j}] source==target 自环")
    return errs


def validate_B2(d, concepts_obj) -> list:
    ids = {str(c["id"]) for c in concepts_obj.get("concepts", [])}
    return _validate_rels(d, ids, {"is-a", "part-of"})


def validate_B3(d, concepts_obj) -> list:
    ids = {str(c["id"]) for c in concepts_obj.get("concepts", [])}
    return _validate_rels(d, ids, {"prerequisite", "applied-in"})


def dedupe_concepts(concepts):
    """最终兜底：按 name 去重（保序），供修复重试仍失败时使用。"""
    seen, out = set(), []
    for c in concepts:
        name = str(c.get("name", "")).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(c)
    return out
