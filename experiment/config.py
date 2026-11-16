# -*- coding: utf-8 -*-
"""Experiment configuration: paths, models, and runtime parameters.

API keys are read from environment variables only and are never committed:
  DeepSeek : DEEPSEEK_API_KEY
  GLM      : ZHIPU_API_KEY or GLM_API_KEY
  Qwen     : DASHSCOPE_API_KEY
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
GS_DIR = ROOT / "data" / "gold_standard"
RESULTS_DIR = ROOT / "data" / "results"
HELDOUT_DIR = ROOT / "data" / "held_out"
HELDOUT_RESULTS_DIR = ROOT / "data" / "results_heldout"
EXPERIMENT_DIR = ROOT / "experiment"

# ---------------------------------------------------------------- .env 加载
# 支持 experiment/.env（KEY=VALUE 逐行，忽略 # 注释与空行）。
# 仅当环境变量未设置时填充——环境变量优先，避免覆盖。
import os as _os
_ENV_FILE = EXPERIMENT_DIR / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _v:
            _os.environ.setdefault(_k, _v)
del _os

TEMPERATURE = 0.0
MAX_REPAIR = 3          # JSON 校验失败后的修复重试次数（另加首次调用，共 1+MAX_REPAIR 次）
TRANSPORT_RETRY = 3     # HTTP/网络失败的退避重试次数
TIMEOUT = 180           # single-request timeout (seconds)
SLEEP_BETWEEN = 0.5     # interval between consecutive requests (seconds)
BACKOFF = (2, 4, 8)     # 传输重试退避秒数

# OpenAI 兼容 chat/completions 端点；max_output 为各家安全上限（保守取值，首次真跑后可调）
MODELS = {
    "deepseek": {
        "label": "DeepSeek", "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "env": "DEEPSEEK_API_KEY", "max_output": 8192,
    },
    "glm": {
        "label": "GLM", "model": "glm-4-air",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "env": "ZHIPU_API_KEY", "env2": "GLM_API_KEY", "max_output": 8192,
    },
    "qwen": {
        "label": "Qwen", "model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "env": "DASHSCOPE_API_KEY", "max_output": 8192,
    },
}

# 概念数量上限（策略 A 输出多、文档长，上限放宽；策略 B R1 控制输出长度）
MAX_CONCEPTS = {"A": 80, "B": 60}

# Generic AI terms used in mock mode only (pipeline smoke test; not gold standard, not used for evaluation)
MOCK_TERMS = [
    "人工智能", "机器学习", "深度学习", "神经网络", "卷积神经网络", "大语言模型",
    "自然语言处理", "计算机视觉", "语音识别", "图像识别", "人脸识别", "强化学习",
    "智能体", "机器人", "知识图谱", "专家系统", "数据挖掘", "聚类", "监督学习",
    "无监督学习", "提示工程", "生成式人工智能", "Transformer", "推荐系统",
    "算法", "大数据", "数据标注", "自动驾驶", "机器学习算法", "知识表示",
    "人工智能伦理", "数据隐私", "算法偏见", "人工智能+X", "人机交互",
]


def api_key(model_key: str) -> str:
    cfg = MODELS.get(model_key)
    if not cfg:
        return ""
    v = os.environ.get(cfg["env"], "") or ""
    if cfg.get("env2"):
        v = v or os.environ.get(cfg["env2"], "") or ""
    return v.strip()


def available_models():
    """返回环境变量中已有密钥的模型 key 列表（保持 MODELS 定义顺序）。"""
    return [k for k in MODELS if api_key(k)]
