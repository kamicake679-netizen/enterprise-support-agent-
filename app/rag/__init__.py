"""星云智联 RAG 混合检索与溯源模块导出入口。

使用模块级 `__getattr__` 按需延迟导出，避免执行 `python -m app.rag.ingest`
时触发 `runpy` 的提前导入警告。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DocumentChunk",
    "EmbeddingClient",
    "HybridRetriever",
    "build_chunks_from_docs",
    "get_retriever",
    "hybrid_search",
    "ingest_documents",
    "tokenize_for_bm25",
    "vector_only_search",
]


def __getattr__(name: str) -> Any:
    if name == "EmbeddingClient":
        from app.rag.embeddings import EmbeddingClient

        return EmbeddingClient
    if name in {"DocumentChunk", "build_chunks_from_docs", "ingest_documents"}:
        from app.rag import ingest

        return getattr(ingest, name)
    if name in {
        "HybridRetriever",
        "get_retriever",
        "hybrid_search",
        "tokenize_for_bm25",
        "vector_only_search",
    }:
        from app.rag import retriever

        return getattr(retriever, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
