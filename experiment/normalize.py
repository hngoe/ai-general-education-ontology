# -*- coding: utf-8 -*-
"""Gold-standard slot / alias / equivalence-group normalization and matching.

Key points:
- GS = golden_concepts_tiered.csv (131 concepts with frequency_tier / concept_level / doc_count_syllabi).
- Equivalent pairs (relation_type == equivalent in golden_relations_extended.csv) are merged into a single
  slot: a hit on any member name counts the slot once (prevents double counting of 自然语言处理 ↔ 自然语言处理(NLP)).
- Each slot stores: canonical name (no-parenthesis, shortest, smallest id preferred) + aliases
  (member names, de-parenthesized variants, English abbreviations, manual aliases).
- Matching priority: normalized exact / alias hit > bidirectional containment (longest covering surface)
  > difflib similarity fallback (threshold adjustable).
- Document anchoring (the gold concept set of a document) uses the same surface scan.

Manual aliases record only clearly safe synonymous forms.
"""
import csv
import re
from difflib import SequenceMatcher

from pathlib import Path

GS_DIR = Path(__file__).resolve().parents[1] / "data" / "gold_standard"

# 人工别名：canonical_name -> [表面写法…]（canonical_name 须存在于 GS，载入时校验）
MANUAL_ALIASES = {
    "大语言模型": ["大模型", "LLM", "语言大模型", "大型语言模型",
                 # matching-table convention: mainstream LLM brand/product names are merged into large language model
                 "DeepSeek", "Kimi", "豆包", "ChatGPT", "文心一言", "通义千问",
                 "GLM", "大语言模型（LLM）"],
    "提示工程": ["提示词工程", "Prompt工程", "prompt工程", "提示工程（Prompt）"],
    "神经网络": ["人工神经网络", "神经网络模型"],
    "计算机视觉": ["机器视觉", "机器视觉技术"],
    "生成式人工智能": ["生成式AI", "生成式人工智能技术"],
    "人工智能": ["AI"],
    "机器学习": ["机器学习技术", "机器学习算法"],
    "深度学习": ["深度神经网络学习"],
    "语音识别": ["语音识别技术"],
    "图像识别": ["图像识别技术"],
    "模式识别": ["模式识别技术"],
    "知识图谱": ["知识图谱技术"],
    "智能体": ["AI智能体", "Agent", "智能体（agent）"],
    "人工智能伦理": ["AI伦理", "人工智能伦理问题"],
    "机器学习算法": [],
}
# 全局归一化替换（匹配前做）：统一括号为半角并删除空格、常见分隔符
_PAREN_FULL = re.compile(r"[（(][^（）()]*[）)]")
_WS = re.compile(r"[\s\u3000·.．,，、;；:：/／\-—_]")


def normalize_name(s: str) -> str:
    s = _PAREN_FULL.sub("", s)          # 先摘掉括号内容（保留主干）
    s = _WS.sub("", s)
    s = s.lower()
    return s


def _strip_paren(s: str) -> str:
    return _PAREN_FULL.sub("", s).strip()


# ---------------------------------------------------------------- 载入 GS

class GsData:
    def __init__(self, merge_equiv=True, fuzzy_ratio=0.88, min_surface_len=2):
        self.rows = self._load_rows()           # 原始 131 行
        self.merge_equiv = merge_equiv
        self.fuzzy_ratio = fuzzy_ratio
        self.min_surface_len = min_surface_len
        self.slots = self._build_slots()        # list[dict]
        self.surface2slot = self._build_surface_map()  # norm surface -> slot_id
        self.slot_by_id = {s["slot_id"]: s for s in self.slots}
        # concept_id -> slot_id（equivalent 组内成员归并到同一槽位）
        self.concept_id2slot = {}
        for s in self.slots:
            for mid in s["member_ids"]:
                self.concept_id2slot[mid] = s["slot_id"]
        # 主关系（75 条 4 型）载入备用
        self.main_relations = self._load_main_relations()

    @staticmethod
    def _load_rows():
        rows = list(csv.DictReader(
            open(GS_DIR / "golden_concepts_tiered.csv", encoding="utf-8-sig")))
        for r in rows:
            r["concept_id"] = str(r["concept_id"]).strip()
            r["concept_name"] = r["concept_name"].strip()
            r["frequency_tier"] = r.get("frequency_tier", "").strip() or "Peripheral"
        return rows

    def _equiv_groups(self):
        """读扩展关系里的 equivalent 行 → 组（id 集合）。"""
        groups = []
        p = GS_DIR / "golden_relations_extended.csv"
        if not p.exists():
            return groups
        seen = set()
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r.get("relation_type", "").strip() != "equivalent":
                continue
            a, b = str(r.get("source_concept_id", "")).strip(), str(r.get("target_concept_id", "")).strip()
            if not a or not b:
                continue
            ga = next((g for g in groups if a in g), None)
            gb = next((g for g in groups if b in g), None)
            if ga is None and gb is None:
                groups.append({a, b})
            elif ga is not None and gb is not None:
                groups.remove(gb); ga |= gb
            elif ga is not None:
                ga.add(b)
            else:
                gb.add(a)
        _ = seen
        return groups

    def _build_slots(self):
        if not self.merge_equiv:
            groups = [{r["concept_id"]} for r in self.rows]
        else:
            groups = self._equiv_groups()
            covered = set().union(*groups) if groups else set()
            groups += [{r["concept_id"]} for r in self.rows
                       if r["concept_id"] not in covered]
        slots = []
        for g in groups:
            members = [r for r in self.rows if r["concept_id"] in g]
            # canonical：无括号 → 最短 → 最小 id
            bare = [m for m in members if "（" not in m["concept_name"] and "(" not in m["concept_name"]]
            pool = bare or members
            canon = sorted(pool, key=lambda m: (len(m["concept_name"]), int(m["concept_id"])))[0]
            aliases = set()
            for m in members:
                aliases.add(m["concept_name"])
                v = _strip_paren(m["concept_name"])
                if v:
                    aliases.add(v)
            for extra in MANUAL_ALIASES.get(canon["concept_name"], []):
                if extra:
                    aliases.add(extra)
            tier = canon["frequency_tier"] or "Peripheral"
            level = canon["concept_level"]
            slots.append({
                "slot_id": canon["concept_id"],
                "canonical": canon["concept_name"],
                "member_ids": sorted(g, key=int),
                "tier": tier, "level": level,
                "aliases": {a for a in aliases if len(a) >= self.min_surface_len},
            })
        slots.sort(key=lambda s: int(s["slot_id"]))
        return slots

    def _build_surface_map(self):
        m = {}
        warn = []
        for s in self.slots:
            for a in s["aliases"]:
                k = normalize_name(a)
                if not k:
                    continue
                if k in m and m[k] != s["slot_id"]:
                    warn.append((a, m[k], s["slot_id"]))
                    continue
                m[k] = s["slot_id"]
        if warn:
            print(f"[normalize] 表面冲突（首胜保留）: {warn[:8]}")
        return m

    def _load_main_relations(self):
        p = GS_DIR / "golden_relations.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        for r in rows:
            for c in ("source_concept_id", "target_concept_id"):
                r[c] = str(r[c]).strip()
            r["relation_type"] = r["relation_type"].strip()
        return rows

    # ---------------------------------------------------------------- 匹配

    def match_mention(self, mention: str):
        """单条抽取概念名 → slot_id；返回 None 表示未命中任何 GS 槽位。

        规则（防反包含误匹配，如 数据 误挂 通用数据保护条例）：
        1) 规范化精确/别名命中；
        2) 抽取名包含某槽位表面串（如 大模型工具 含 大模型），取“最长表面”槽位；
        3) difflib 相似度兜底（对规范化名）。
        不再执行“槽位表面包含抽取名”的反向匹配。
        """
        mention = (mention or "").strip()
        if not mention:
            return None
        k = normalize_name(mention)
        if k in self.surface2slot:
            return self.surface2slot[k]
        # 抽取名包含槽位表面 → 取最长表面对应的槽位（最具体优先）
        best, best_len = None, -1
        for s in self.slots:
            for a in s["aliases"]:
                if len(a) < self.min_surface_len:
                    continue
                ak = normalize_name(a)
                if not ak:
                    continue
                if ak and ak in k and len(ak) > best_len:
                    best, best_len = s["slot_id"], len(ak)
        if best is not None:
            return best
        # difflib 兜底（对规范化名按字符比较）
        ck = normalize_name(mention)
        best, best_r = None, self.fuzzy_ratio
        for s in self.slots:
            ak = normalize_name(s["canonical"])
            if not ak:
                continue
            r = SequenceMatcher(None, ck, ak).ratio()
            if r >= best_r:
                best, best_r = s["slot_id"], r
        return best

    def anchor_slots(self, doc_text: str):
        """该文档文本中出现的 GS 槽位集合（文档级 gold 概念集）。"""
        if not doc_text:
            return set()
        found = set()
        for s in self.slots:
            for a in s["aliases"]:
                if len(a) < self.min_surface_len:
                    continue
                if a in doc_text:
                    found.add(s["slot_id"])
                    break
        return found
