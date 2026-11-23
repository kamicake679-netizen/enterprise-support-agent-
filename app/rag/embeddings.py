"""通用 OpenAI 兼容 Embedding 客户端与限流重试封装。

功能：
- 统一从 `app.config.get_settings()` 读取 `EMBEDDING_*` 配置；
- 支持批量请求向量化，控制单批大小与批间间隔，降低免费 API 触发 429 概率；
- 内置 429 限流、5xx 服务端异常及网络超时指数退避重试机制。
"""

from __future__ import annotations

import logging
import random
import time
from typing import Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """带分批与 429 自动退避重试的 OpenAI 兼容向量客户端。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        api_key = self.settings.EMBEDDING_API_KEY.strip()
        if not api_key:
            raise ValueError(
                "EMBEDDING_API_KEY 未配置，请先在根目录 .env 文件中填写有效的向量模型 API Key。"
            )
        self.model = self.settings.EMBEDDING_MODEL
        self.batch_size = max(1, self.settings.EMBEDDING_BATCH_SIZE)
        self.max_retries = max(1, self.settings.EMBEDDING_MAX_RETRIES)
        self.batch_interval = max(0.0, self.settings.EMBEDDING_BATCH_INTERVAL)
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.settings.EMBEDDING_BASE_URL,
        )

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        """尝试从 HTTP 响应头中解析 Retry-After 秒数。"""
        response = getattr(exc, "response", None)
        if response is None:
            return None
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if not retry_after:
            return None
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            return None

    def _request_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """对单个批次发起 Embedding 请求，遇到 429 或瞬时错误自动指数退避重试。"""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # 若兼容接口均返回了有效 index，则按 index 排序；否则保留接口原始顺序（如部分 Gemini 批量响应）
                if all(getattr(item, "index", None) is not None for item in resp.data):
                    sorted_data = sorted(resp.data, key=lambda item: item.index)
                else:
                    sorted_data = list(resp.data)
                return [list(item.embedding) for item in sorted_data]
            except RateLimitError as exc:
                last_error = exc
                retry_after = self._extract_retry_after(exc)
                base_wait = retry_after if retry_after is not None else min(60.0, 2.0 ** attempt)
                wait_seconds = base_wait + random.uniform(0.2, 1.0)
                logger.warning(
                    "Embedding 接口触发 429 限流（第 %d/%d 次），%.2f 秒后重试...",
                    attempt,
                    self.max_retries,
                    wait_seconds,
                )
                if attempt < self.max_retries:
                    time.sleep(wait_seconds)
            except (APITimeoutError, APIConnectionError) as exc:
                last_error = exc
                wait_seconds = min(30.0, 1.5 ** attempt) + random.uniform(0.1, 0.6)
                logger.warning(
                    "Embedding 网络/超时异常（第 %d/%d 次）：%s，%.2f 秒后重试...",
                    attempt,
                    self.max_retries,
                    exc,
                    wait_seconds,
                )
                if attempt < self.max_retries:
                    time.sleep(wait_seconds)
            except APIStatusError as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", 0)
                if status_code == 429 or status_code >= 500:
                    retry_after = self._extract_retry_after(exc)
                    base_wait = retry_after if retry_after is not None else min(45.0, 2.0 ** attempt)
                    wait_seconds = base_wait + random.uniform(0.2, 0.8)
                    logger.warning(
                        "Embedding 服务返回状态码 %s（第 %d/%d 次），%.2f 秒后重试...",
                        status_code,
                        attempt,
                        self.max_retries,
                        wait_seconds,
                    )
                    if attempt < self.max_retries:
                        time.sleep(wait_seconds)
                else:
                    raise

        raise RuntimeError(
            f"Embedding 请求连续重试 {self.max_retries} 次后仍失败: {last_error}"
        ) from last_error

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """批量生成文本向量，支持自动分批与限流间隔控制。"""
        if not texts:
            return []

        cleaned = [t.strip() if t and t.strip() else " " for t in texts]
        total = len(cleaned)
        all_embeddings: list[list[float]] = []

        for start in range(0, total, self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            batch_embeddings = self._request_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            # 还有下一批时做短暂间隔，保护免费 API 速率配额
            if start + self.batch_size < total and self.batch_interval > 0:
                time.sleep(self.batch_interval)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """生成单条查询语句的向量。"""
        vectors = self.embed_texts([query])
        return vectors[0]
