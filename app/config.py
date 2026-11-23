"""全局配置模块。

从环境变量（及 `.env` 文件）加载模型接口、存储路径与 RAG/Agent 默认超参数。
全部模型调用基于通用 OpenAI 兼容协议，支持零代码修改切换平台。
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录（app/config.py 的上一级的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用全局配置类。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 1. 对话大模型配置（OpenAI 兼容） ---
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_FALLBACK_MODEL: str = ""
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048

    # --- 2. 向量模型配置（OpenAI 兼容） ---
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # --- 3. 重排模型配置（可选开关，默认关闭走本地 RRF） ---
    RERANK_ENABLED: bool = False
    RERANK_API_KEY: str = ""
    RERANK_BASE_URL: str = ""
    RERANK_MODEL: str = ""

    # --- 4. 数据与存储路径 ---
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "nebulalink_kb"
    SQLITE_DB_PATH: str = "data/business.db"
    DOCS_DIR: str = "data/docs"

    # --- 5. RAG 检索默认超参数 ---
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 60
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_MAX_RETRIES: int = 5
    EMBEDDING_BATCH_INTERVAL: float = 0.3
    VECTOR_TOP_K: int = 10
    BM25_TOP_K: int = 10
    FINAL_TOP_K: int = 3
    RRF_K: int = 60
    # 相关性过滤阈值（cosine 相似度范围 0~1，已针对 BAAI/bge-m3 与中文场景校准，防止无关问题强行召回）
    VECTOR_SCORE_THRESHOLD: float = 0.56
    BM25_SCORE_THRESHOLD: float = 5.0
    VECTOR_FLOOR_FOR_BM25: float = 0.45
    RERANK_SCORE_THRESHOLD: float = 0.35

    # --- 6. Agent 运行超参数 ---
    AGENT_MAX_STEPS: int = 5
    AGENT_HISTORY_WINDOW: int = 6
    LLM_MAX_RETRIES: int = 5
    MAX_TOOL_ITERATIONS: int = 5

    @property
    def chroma_dir_path(self) -> Path:
        """返回解析后的 Chroma 持久化绝对路径。"""
        p = Path(self.CHROMA_PERSIST_DIR)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def sqlite_db_file(self) -> Path:
        """返回解析后的 SQLite 数据库绝对路径。"""
        p = Path(self.SQLITE_DB_PATH)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def docs_dir_path(self) -> Path:
        """返回解析后的业务知识库文档目录绝对路径。"""
        p = Path(self.DOCS_DIR)
        return p if p.is_absolute() else BASE_DIR / p


@lru_cache
def get_settings() -> Settings:
    """获取缓存后的全局配置实例。"""
    return Settings()
