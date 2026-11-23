"""混合检索与 RRF 融合精排模块。

能力特性：
1. 双路召回：ChromaDB 向量检索（Top 10） + `jieba` 增强分词 `BM25` 关键词检索（Top 10）；
2. RRF（Reciprocal Rank Fusion）融合去重；
3. 可选 Rerank 开关（`RERANK_ENABLED`）：默认关闭直接走本地免费 RRF 取 Top 3，开启后自动调用外部 Rerank API 重排；
4. 相关性阈值过滤：过滤低相似度/低置信度片段，防止知识库外无关问题被强行召回；
5. 提供 `vector_only_search(query, top_k=3)` 方法作为后续评测对比的 Baseline 基线。
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Any

import chromadb
import httpx
import jieba
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.rag.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.WARNING)

# 常见中文/标点停用词：避免无关问句因“的/怎么/什么/如何”误触 BM25
_STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "呢", "吗", "啊", "吧", "呀", "哦", "嘛", "着", "过",
        "是", "在", "有", "和", "与", "或", "及", "就", "都", "也", "还",
        "被", "把", "对", "给", "从", "向", "到", "为", "以", "按", "于",
        "我", "你", "您", "他", "她", "它", "我们", "你们", "他们", "自己",
        "这", "那", "这个", "那个", "这些", "那些", "某", "每", "各",
        "什么", "怎么", "怎样", "如何", "为什么", "为何", "哪", "哪个", "哪里", "哪些",
        "多少", "几", "是否", "能否", "可以", "请问", "帮我", "帮忙", "一下", "一下子",
        "一个", "一种", "一些", "如果", "但是", "因为", "所以", "而且", "并且", "或者",
        "非常", "比较", "已经", "现在", "今天", "明天", "昨天", "大家", "问题", "情况",
        "需要", "想要", "想问", "知道", "告诉", "一下", "一下吧", "谢谢", "感谢",
    }
)

# 精确匹配产品型号、百分比、数值条款与专业缩写
_SPECIAL_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 连字符型号/代码：如 NL-Pro2026、NL-Air3、NL-Studio、USB-C、3.5mm
    re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)+"),
    # 带加号专有名词：如 NebulaCare+
    re.compile(r"[A-Za-z0-9]+\+"),
    # 百分比与带单位数值：如 15%、10%、-52dB、-48dB、168小时、30天、12个月、24bit、96kHz
    re.compile(
        r"[+-]?\d+(?:\.\d+)?(?:%|dB|db|DB|mm|g|h|mAh|kbps|kHz|bit|V|A|小时|天|个月|年|元|只|个|次)",
    ),
    # 人民币金额：如 ¥1,299、¥399、¥149
    re.compile(r"[¥￥]\d+(?:,\d+)*(?:\.\d+)?"),
    # 英文缩写/型号片段：如 IP55、IPX4、LDAC、LHDC、AAC、ANC、ENC、P0、P1、P2、RMA、SN
    re.compile(r"[A-Za-z][A-Za-z0-9_]*"),
)

_PURE_PUNCT_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    """面向中英混合与型号/数字条款的 BM25 增强分词器。

    特点：
    - 完整保留 `NL-Pro2026`、`15%`、`-52dB`、`¥399`、`IP55` 等精确词元；
    - 结合 `jieba.lcut_for_search` 切分中文细粒度语义词；
    - 过滤纯标点与高频无意义停用词，减少无关问句的噪音命中。
    """
    if not text or not text.strip():
        return []

    tokens: list[str] = []

    # 1. 优先提取型号、百分比、金额、专业缩写等精确复合词元
    for pattern in _SPECIAL_TOKEN_PATTERNS:
        for match in pattern.findall(text):
            norm = match.lower().replace(",", "").strip()
            if not norm:
                continue
            tokens.append(norm)
            # 对带负号/货币符号的数值补充去符号版本（如 -52db -> 52db, ¥399 -> 399）
            stripped = norm.lstrip("+-¥￥")
            if stripped and stripped != norm:
                tokens.append(stripped)

    # 2. jieba 搜索引擎模式中文分词
    for raw_tok in jieba.lcut_for_search(text):
        tok = raw_tok.strip().lower()
        if not tok:
            continue
        if _PURE_PUNCT_PATTERN.match(tok):
            continue
        if tok in _STOPWORDS:
            continue
        # 过滤单个无意义英文字母，但保留单个有意义汉字或纯数字
        if len(tok) == 1 and tok.isascii() and tok.isalpha():
            continue
        tokens.append(tok)

    return tokens


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度，结果截断至 [0.0, 1.0]。"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return max(0.0, min(1.0, sim))


class HybridRetriever:
    """星云智联混合检索器（Chroma 向量检索 + BM25 + RRF 融合 + 可选 Rerank）。"""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or EmbeddingClient(settings=self.settings)

        self._chroma_client: Any = None
        self._collection: Any = None
        self._chunk_ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._embeddings: list[list[float]] = []
        self._id_to_index: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None

        self.reload()

    def reload(self) -> None:
        """从本地持久化 ChromaDB 加载集合并重建内存 BM25 索引。"""
        persist_dir = self.settings.chroma_dir_path
        collection_name = self.settings.CHROMA_COLLECTION_NAME

        if not persist_dir.exists():
            raise RuntimeError(
                f"ChromaDB 持久化目录不存在 ({persist_dir})，请先运行 `python -m app.rag.ingest` 完成知识库入库。"
            )

        self._chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        existing_names = [
            col.name if hasattr(col, "name") else str(col)
            for col in self._chroma_client.list_collections()
        ]
        if collection_name not in existing_names:
            raise RuntimeError(
                f"ChromaDB 中未找到集合 `{collection_name}`，请先运行 `python -m app.rag.ingest` 完成知识库入库。"
            )

        self._collection = self._chroma_client.get_collection(name=collection_name)
        data = self._collection.get(include=["documents", "metadatas", "embeddings"])

        self._chunk_ids = list(data.get("ids") or [])
        self._documents = list(data.get("documents") or [])
        self._metadatas = [dict(m or {}) for m in (data.get("metadatas") or [])]
        raw_embeddings = data.get("embeddings")
        if raw_embeddings is not None and len(raw_embeddings) > 0:
            self._embeddings = [list(map(float, emb)) for emb in raw_embeddings]
        else:
            self._embeddings = []

        if not self._chunk_ids:
            raise RuntimeError(
                f"集合 `{collection_name}` 为空，请先运行 `python -m app.rag.ingest` 入库知识文档。"
            )

        self._id_to_index = {cid: idx for idx, cid in enumerate(self._chunk_ids)}

        # 构建 BM25 索引（为空词元文档保底占位，防止 rank_bm25 出现除零）
        tokenized_corpus = [
            tokenize_for_bm25(doc) or ["_empty_"] for doc in self._documents
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(
            "HybridRetriever 已就绪：集合 `%s` 共加载 %d 条切片。",
            collection_name,
            len(self._chunk_ids),
        )

    @property
    def total_chunks(self) -> int:
        """返回知识库当前加载的切片总数。"""
        return len(self._chunk_ids)

    def _vector_recall(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """执行 ChromaDB 向量召回，返回按相似度降序排列的候选列表。"""
        n_results = min(max(1, top_k), len(self._chunk_ids))
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists), start=1
        ):
            # Chroma 使用 cosine 距离：distance = 1 - cosine_similarity
            vec_score = max(0.0, min(1.0, 1.0 - float(dist)))
            meta_dict = dict(meta or {})
            results.append(
                {
                    "chunk_id": cid,
                    "text": doc,
                    "source": str(meta_dict.get("source", "")),
                    "section": str(meta_dict.get("section", "")),
                    "vector_score": vec_score,
                    "vector_rank": rank,
                }
            )
        return results

    def _bm25_recall(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """执行 jieba + BM25 关键词召回，仅保留得分严格大于 0 的切片。"""
        if self._bm25 is None:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        scored_indices = [
            (idx, float(score))
            for idx, score in enumerate(scores)
            if float(score) > 0.0
        ]
        scored_indices.sort(key=lambda item: item[1], reverse=True)

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(scored_indices[: max(1, top_k)], start=1):
            meta_dict = self._metadatas[idx]
            results.append(
                {
                    "chunk_id": self._chunk_ids[idx],
                    "text": self._documents[idx],
                    "source": str(meta_dict.get("source", "")),
                    "section": str(meta_dict.get("section", "")),
                    "bm25_score": score,
                    "bm25_rank": rank,
                }
            )
        return results

    def _call_external_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]] | None:
        """当 `RERANK_ENABLED=true` 时，调用外部 Rerank API（兼容 Cohere/Jina/SiliconFlow）。"""
        base_url = self.settings.RERANK_BASE_URL.strip()
        api_key = self.settings.RERANK_API_KEY.strip()
        model = self.settings.RERANK_MODEL.strip()

        if not (base_url and api_key and model):
            logger.warning(
                "RERANK_ENABLED=true 但 RERANK_BASE_URL/API_KEY/MODEL 未完整配置，自动降级为本地 RRF 排序。"
            )
            return None

        endpoint = (
            base_url
            if base_url.rstrip("/").endswith("/rerank")
            else f"{base_url.rstrip('/')}/rerank"
        )
        payload = {
            "model": model,
            "query": query,
            "documents": [item["text"] for item in candidates],
            "top_n": min(top_n, len(candidates)),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("外部 Rerank 接口调用失败（将自动降级为本地 RRF）：%s", exc)
            return None

        raw_results = data.get("results") or data.get("data") or []
        if not raw_results:
            logger.warning("外部 Rerank 返回空结果，降级为本地 RRF。")
            return None

        reranked: list[dict[str, Any]] = []
        for item in raw_results:
            idx = item.get("index")
            if idx is None or not (0 <= idx < len(candidates)):
                continue
            cand = dict(candidates[idx])
            rerank_score = float(
                item.get("relevance_score", item.get("score", 0.0))
            )
            cand["rerank_score"] = rerank_score
            reranked.append(cand)

        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked

    def _compute_real_vector_score(
        self,
        chunk_id: str,
        query_embedding: list[float],
    ) -> float:
        """为仅由 BM25 召回的切片补算真实的余弦相似度。"""
        idx = self._id_to_index.get(chunk_id)
        if idx is None or idx >= len(self._embeddings):
            return 0.0
        return _cosine_similarity(query_embedding, self._embeddings[idx])

    def _evaluate_confidence(
        self,
        candidate: dict[str, Any],
        use_rerank: bool,
        vector_threshold: float,
        bm25_threshold: float,
        vector_floor_for_bm25: float,
        rerank_threshold: float,
    ) -> tuple[bool, str]:
        """评估单条候选的相关性是否过阈值，返回 `(is_relevant, confidence_level)`。"""
        vec_score = float(candidate.get("vector_score", 0.0))
        bm25_score = float(candidate.get("bm25_score", 0.0))

        if use_rerank and "rerank_score" in candidate:
            r_score = float(candidate["rerank_score"])
            if r_score < rerank_threshold:
                return False, "low"
            level = "high" if r_score >= rerank_threshold + 0.25 else "medium"
            return True, level

        vec_pass = vec_score >= vector_threshold
        bm25_pass = bm25_score >= bm25_threshold and vec_score >= vector_floor_for_bm25

        if not (vec_pass or bm25_pass):
            return False, "low"

        if (vec_score >= vector_threshold + 0.06) or (vec_pass and bm25_score >= bm25_threshold):
            return True, "high"
        return True, "medium"

    def search(
        self,
        query: str,
        top_k: int | None = None,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_k: int | None = None,
        vector_threshold: float | None = None,
        bm25_threshold: float | None = None,
        vector_floor_for_bm25: float | None = None,
        rerank_threshold: float | None = None,
        return_low_confidence: bool = False,
    ) -> list[dict[str, Any]]:
        """执行混合检索（向量 Top 10 + BM25 Top 10 -> RRF 融合 -> 可选 Rerank -> 阈值过滤）。

        Args:
            query: 用户查询文本。
            top_k: 最终返回条数，默认读取 `settings.FINAL_TOP_K`（3）。
            vector_top_k: 向量路召回条数，默认 10。
            bm25_top_k: BM25 路召回条数，默认 10。
            rrf_k: RRF 平滑常数，默认 60。
            vector_threshold: 向量余弦相似度门槛。
            bm25_threshold: BM25 强关键词命中门槛。
            vector_floor_for_bm25: BM25 强命中时的最低向量语义保底分，防止无关句偶然撞词。
            rerank_threshold: 开启外部 Rerank 时的重排分数门槛。
            return_low_confidence: 为 False（默认）时丢弃低置信度片段（无关问题返回空列表）；
                为 True 时保留结果并标注 `low_confidence=True`。

        Returns:
            结构化检索结果列表，每项包含 `text`、`source`、`section`、`score` 等字段。
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        final_k = top_k if top_k is not None else self.settings.FINAL_TOP_K
        v_k = vector_top_k if vector_top_k is not None else self.settings.VECTOR_TOP_K
        b_k = bm25_top_k if bm25_top_k is not None else self.settings.BM25_TOP_K
        k_rrf = rrf_k if rrf_k is not None else self.settings.RRF_K

        v_thresh = (
            vector_threshold
            if vector_threshold is not None
            else self.settings.VECTOR_SCORE_THRESHOLD
        )
        b_thresh = (
            bm25_threshold
            if bm25_threshold is not None
            else self.settings.BM25_SCORE_THRESHOLD
        )
        v_floor = (
            vector_floor_for_bm25
            if vector_floor_for_bm25 is not None
            else self.settings.VECTOR_FLOOR_FOR_BM25
        )
        r_thresh = (
            rerank_threshold
            if rerank_threshold is not None
            else self.settings.RERANK_SCORE_THRESHOLD
        )

        # 1. 双路召回
        query_embedding = self.embedder.embed_query(cleaned_query)
        vector_hits = self._vector_recall(query_embedding, top_k=v_k)
        bm25_hits = self._bm25_recall(cleaned_query, top_k=b_k)

        # 2. RRF 融合去重
        merged: dict[str, dict[str, Any]] = {}

        for item in vector_hits:
            cid = item["chunk_id"]
            merged[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "source": item["source"],
                "section": item["section"],
                "vector_score": item["vector_score"],
                "vector_rank": item["vector_rank"],
                "bm25_score": 0.0,
                "bm25_rank": None,
                "rrf_score": 1.0 / (k_rrf + item["vector_rank"]),
            }

        for item in bm25_hits:
            cid = item["chunk_id"]
            bm25_rrf = 1.0 / (k_rrf + item["bm25_rank"])
            if cid in merged:
                merged[cid]["bm25_score"] = item["bm25_score"]
                merged[cid]["bm25_rank"] = item["bm25_rank"]
                merged[cid]["rrf_score"] += bm25_rrf
            else:
                real_vec_score = self._compute_real_vector_score(cid, query_embedding)
                merged[cid] = {
                    "chunk_id": cid,
                    "text": item["text"],
                    "source": item["source"],
                    "section": item["section"],
                    "vector_score": real_vec_score,
                    "vector_rank": None,
                    "bm25_score": item["bm25_score"],
                    "bm25_rank": item["bm25_rank"],
                    "rrf_score": bm25_rrf,
                }

        fused_candidates = sorted(
            merged.values(),
            key=lambda x: (x["rrf_score"], x["vector_score"], x["bm25_score"]),
            reverse=True,
        )

        # 3. 可选外部 Rerank（默认关闭，直接走本地免费 RRF）
        used_rerank = False
        ordered_candidates = fused_candidates
        if self.settings.RERANK_ENABLED and fused_candidates:
            reranked = self._call_external_rerank(
                query=cleaned_query,
                candidates=fused_candidates[: max(final_k * 3, 10)],
                top_n=max(final_k * 2, final_k),
            )
            if reranked is not None:
                ordered_candidates = reranked
                used_rerank = True

        # 4. 相似度 / 分数阈值过滤与结构化封装
        output: list[dict[str, Any]] = []
        mode = "hybrid_rerank" if used_rerank else "hybrid_rrf"

        for cand in ordered_candidates:
            is_relevant, confidence = self._evaluate_confidence(
                candidate=cand,
                use_rerank=used_rerank,
                vector_threshold=v_thresh,
                bm25_threshold=b_thresh,
                vector_floor_for_bm25=v_floor,
                rerank_threshold=r_thresh,
            )
            if not is_relevant and not return_low_confidence:
                continue

            primary_score = (
                cand["rerank_score"] if used_rerank else cand["rrf_score"]
            )
            output.append(
                {
                    "text": cand["text"],
                    "source": cand["source"],
                    "section": cand["section"],
                    "score": round(float(primary_score), 6),
                    "chunk_id": cand["chunk_id"],
                    "vector_score": round(float(cand["vector_score"]), 4),
                    "bm25_score": round(float(cand["bm25_score"]), 4),
                    "rrf_score": round(float(cand["rrf_score"]), 6),
                    "confidence": confidence,
                    "low_confidence": not is_relevant,
                    "retrieval_mode": mode,
                }
            )
            if len(output) >= final_k:
                break

        return output

    def vector_only_search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float | None = None,
        return_low_confidence: bool = False,
    ) -> list[dict[str, Any]]:
        """纯向量检索 Baseline（仅走 ChromaDB，用于后续 Step 5 评测对比）。

        Args:
            query: 用户查询文本。
            top_k: 返回条数，默认 3。
            score_threshold: 相似度阈值；默认读取 `settings.VECTOR_SCORE_THRESHOLD`，
                传入 `0.0` 可禁用阈值过滤。
            return_low_confidence: 是否保留低于阈值的低置信度片段。

        Returns:
            结构化检索结果列表，每项包含 `text`、`source`、`section`、`score` 等字段。
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        threshold = (
            score_threshold
            if score_threshold is not None
            else self.settings.VECTOR_SCORE_THRESHOLD
        )

        query_embedding = self.embedder.embed_query(cleaned_query)
        hits = self._vector_recall(query_embedding, top_k=max(1, top_k))

        results: list[dict[str, Any]] = []
        for hit in hits:
            vec_score = float(hit["vector_score"])
            is_relevant = vec_score >= threshold
            if not is_relevant and not return_low_confidence:
                continue

            if not is_relevant:
                confidence = "low"
            elif vec_score >= threshold + 0.06:
                confidence = "high"
            else:
                confidence = "medium"

            results.append(
                {
                    "text": hit["text"],
                    "source": hit["source"],
                    "section": hit["section"],
                    "score": round(vec_score, 4),
                    "chunk_id": hit["chunk_id"],
                    "vector_score": round(vec_score, 4),
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                    "confidence": confidence,
                    "low_confidence": not is_relevant,
                    "retrieval_mode": "vector_only",
                }
            )
        return results


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """获取进程内复用的混合检索器单例。"""
    return HybridRetriever()


def hybrid_search(query: str, top_k: int = 3, **kwargs: Any) -> list[dict[str, Any]]:
    """便捷函数：使用全局检索器执行混合检索。"""
    return get_retriever().search(query=query, top_k=top_k, **kwargs)


def vector_only_search(
    query: str,
    top_k: int = 3,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """便捷函数：使用全局检索器执行纯向量检索（Baseline）。"""
    return get_retriever().vector_only_search(query=query, top_k=top_k, **kwargs)
