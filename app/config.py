"""配置管理：读 .env / 环境变量，集中成一个 Settings 对象。

为什么要单独一个 config.py？
- 密钥不能写死在代码里（会被 git 提交出去）；
- 所有可调参数（chunk 大小、召回条数、模型名）集中在一处，改参数只需动一个地方。

这里没有用 pydantic-settings，而是手写了一个 20 行的 .env 读取器，
为的是减少依赖、保持零门槛启动。生产项目里换成 pydantic-settings 更稳妥。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 项目根目录（app/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 读取：KEY=VALUE，# 开头的行忽略，已存在的环境变量优先。"""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def _str(key: str, default: str = "") -> str:
    return os.getenv(key) or default


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # ---- 大模型（OpenAI 兼容接口）----
    llm_api_key: str = _str("LLM_API_KEY")
    llm_base_url: str = _str("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    llm_model: str = _str("LLM_MODEL", "glm-4-flash")
    temperature: float = _float("LLM_TEMPERATURE", 0.3)

    # ---- 向量化（可以和大模型用同一家服务商）----
    embedding_api_key: str = _str("EMBEDDING_API_KEY")
    embedding_base_url: str = _str("EMBEDDING_BASE_URL")
    embedding_model: str = _str("EMBEDDING_MODEL", "embedding-3")
    embedding_batch: int = _int("EMBEDDING_BATCH", 16)

    # ---- 检索参数 ----
    chunk_size: int = _int("CHUNK_SIZE", 500)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 80)
    top_k: int = _int("TOP_K", 4)
    # 混合检索里向量得分的权重，其余给关键词得分
    hybrid_alpha: float = _float("HYBRID_ALPHA", 0.6)
    # 最高相似度低于这个值就拒答，用来压幻觉
    score_threshold: float = _float("SCORE_THRESHOLD", 0.30)

    # ---- 会话 ----
    max_turns: int = _int("MAX_TURNS", 8)
    redis_url: str = _str("REDIS_URL")
    session_ttl: int = _int("SESSION_TTL", 3600)

    # ---- 其它 ----
    index_path: str = _str("INDEX_PATH", str(ROOT / "index.json"))
    data_dir: str = _str("DATA_DIR", str(ROOT / "data"))
    max_agent_rounds: int = _int("MAX_AGENT_ROUNDS", 5)

    def embedding_client_kwargs(self) -> dict:
        """向量化服务的连接参数；没单独配就复用大模型的。"""
        return {
            "api_key": self.embedding_api_key or self.llm_api_key,
            "base_url": self.embedding_base_url or self.llm_base_url,
        }


settings = Settings()
