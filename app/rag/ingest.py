"""知识库 Markdown 文档切片与向量入库模块。

功能：
1. 读取 `data/docs/` 下所有 Markdown（`*.md`）文档；
2. 按 Markdown 标题层级优先分段，并结合约 400 字符滑动窗口（重叠 60 字符）切分长章节；
   对超长 Markdown 表格自动保留表头上下文，防止跨片丢失列名；
3. 保留结构化 `metadata`（`source` 文件名、`section` 章节标题、`doc_title`、`chunk_index`）；
4. 调用 `.env` 配置的通用 OpenAI 兼容 `EMBEDDING_*` 接口（含分批与 429 退避重试）生成向量；
5. 写入本地持久化 `ChromaDB`（cosine 距离空间），默认支持幂等重建避免重复切片。

命令行使用：
    python -m app.rag.ingest
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.config import Settings, get_settings
from app.rag.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HR_PATTERN = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_TABLE_SEP_PATTERN = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


@dataclass(slots=True)
class DocumentChunk:
    """单个知识库文档切片。"""

    chunk_id: str
    text: str
    source: str
    section: str
    doc_title: str
    chunk_index: int

    def to_metadata(self) -> dict[str, Any]:
        """转换为 ChromaDB 存储所需的 metadata 字典。"""
        return {
            "source": self.source,
            "section": self.section,
            "doc_title": self.doc_title,
            "chunk_index": self.chunk_index,
        }


def _clean_block_lines(lines: list[str]) -> str:
    """清理章节首尾空行及纯分隔线。"""
    cleaned: list[str] = []
    for line in lines:
        if _HR_PATTERN.match(line):
            continue
        cleaned.append(line.rstrip())
    text = "\n".join(cleaned).strip()
    # 压缩连续 3 个以上空行为 2 个换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def parse_markdown_sections(markdown_text: str, filename: str) -> tuple[str, list[tuple[str, str]]]:
    """按 Markdown 标题层级分段，返回 `(doc_title, [(section_title, section_body), ...])`。"""
    lines = markdown_text.splitlines()
    doc_title = Path(filename).stem
    h2_title = ""
    h3_title = ""
    current_section = "概述"
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        body = _clean_block_lines(buffer)
        buffer.clear()
        if body:
            sections.append((current_section, body))

    for raw_line in lines:
        match = _HEADING_PATTERN.match(raw_line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            if level == 1:
                flush()
                doc_title = title
                h2_title = ""
                h3_title = ""
                current_section = "概述"
            elif level == 2:
                flush()
                h2_title = title
                h3_title = ""
                current_section = h2_title
            elif level == 3:
                flush()
                h3_title = title
                current_section = f"{h2_title} > {h3_title}" if h2_title else h3_title
            else:
                # 4 级及以下标题保留在当前章节正文中作为小节标记
                buffer.append(raw_line)
        else:
            buffer.append(raw_line)

    flush()
    return doc_title, sections


def _split_into_text_and_table_blocks(text: str) -> list[tuple[str, str]]:
    """将章节正文拆分为普通文本块 (`text`) 与 Markdown 表格块 (`table`) 序列。"""
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    idx = 0
    n = len(lines)
    text_buf: list[str] = []

    def flush_text() -> None:
        joined = "\n".join(text_buf).strip()
        text_buf.clear()
        if joined:
            blocks.append(("text", joined))

    while idx < n:
        # 判断当前行与下一行是否构成 Markdown 表格表头 + 分隔行
        if (
            idx + 1 < n
            and "|" in lines[idx]
            and _TABLE_SEP_PATTERN.match(lines[idx + 1])
        ):
            flush_text()
            table_lines = [lines[idx], lines[idx + 1]]
            idx += 2
            while idx < n and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            blocks.append(("table", "\n".join(table_lines).strip()))
        else:
            text_buf.append(lines[idx])
            idx += 1

    flush_text()
    return blocks


def _split_table_block(table_text: str, chunk_size: int) -> list[str]:
    """切分超长 Markdown 表格，确保每个子切片均保留完整表头并带有 1 行上下文重叠。"""
    if len(table_text) <= chunk_size:
        return [table_text]

    lines = table_text.splitlines()
    if len(lines) <= 3:
        return [table_text]

    header_lines = lines[:2]
    header_text = "\n".join(header_lines)
    data_rows = lines[2:]

    chunks: list[str] = []
    row_idx = 0
    total_rows = len(data_rows)

    while row_idx < total_rows:
        current_rows: list[str] = []
        current_len = len(header_text)

        while row_idx < total_rows:
            row = data_rows[row_idx]
            projected = current_len + 1 + len(row)
            if current_rows and projected > chunk_size:
                break
            current_rows.append(row)
            current_len = projected
            row_idx += 1

        chunks.append(header_text + "\n" + "\n".join(current_rows))

        # 若仍有剩余数据行且本次放入超过 1 行，则回退 1 行作为滑动重叠，防止边界信息割裂
        if row_idx < total_rows and len(current_rows) > 1:
            row_idx -= 1

    return chunks


def _sliding_window_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """对普通文本执行自然边界优先的滑动窗口切分。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    overlap = min(chunk_overlap, max(0, chunk_size // 2))
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    boundary_chars = ("\n\n", "\n", "。", "；", "！", "？", ". ", "; ")

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # 优先在窗口后半部分寻找自然断句边界
            min_cut = start + int(chunk_size * 0.55)
            best_cut = -1
            for sep in boundary_chars:
                pos = text.rfind(sep, min_cut, end)
                if pos != -1:
                    best_cut = pos + len(sep)
                    break
            if best_cut > start:
                end = best_cut

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break

        # 若剩余内容较短且已接近末尾，直接让下一窗口覆盖到末尾并避免产生极小尾巴
        next_start = max(end - overlap, start + 1)
        if text_len - next_start <= int(chunk_size * 0.35) and chunks:
            # 若剩余部分极短，确保完整纳入最后一块
            tail = text[next_start:].strip()
            if tail and tail not in chunks[-1]:
                chunks.append(tail)
            break

        # 轻微跳过前导空白
        while next_start < text_len and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def split_section_text(
    section_body: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[str]:
    """将单个章节正文切分为长度约 `chunk_size`、重叠约 `chunk_overlap` 的文本片段。"""
    section_body = section_body.strip()
    if not section_body:
        return []
    if len(section_body) <= chunk_size:
        return [section_body]

    blocks = _split_into_text_and_table_blocks(section_body)
    pieces: list[str] = []

    for block_type, block_content in blocks:
        if block_type == "table":
            table_pieces = _split_table_block(block_content, chunk_size=chunk_size)
            # 若上一段普通引导文字很短，可与第一块表格合并（总长不超过 chunk_size + 80）
            if (
                pieces
                and table_pieces
                and len(pieces[-1]) <= 120
                and len(pieces[-1]) + 2 + len(table_pieces[0]) <= chunk_size + 80
            ):
                pieces[-1] = f"{pieces[-1]}\n\n{table_pieces[0]}"
                pieces.extend(table_pieces[1:])
            else:
                pieces.extend(table_pieces)
        else:
            text_pieces = _sliding_window_split(
                block_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            pieces.extend(text_pieces)

    return [p for p in pieces if p.strip()]


def build_chunks_from_docs(
    docs_dir: Path,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[DocumentChunk]:
    """扫描目录下全部 Markdown 文档并生成结构化切片列表。"""
    if not docs_dir.exists():
        raise FileNotFoundError(f"知识库文档目录不存在: {docs_dir}")

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"知识库文档目录中未找到任何 .md 文件: {docs_dir}")

    all_chunks: list[DocumentChunk] = []

    for doc_idx, md_path in enumerate(md_files, start=1):
        raw_text = md_path.read_text(encoding="utf-8")
        source_name = md_path.name
        doc_title, sections = parse_markdown_sections(raw_text, source_name)

        doc_chunk_idx = 0
        for section_title, section_body in sections:
            # 预留章节标题头长度，使整体 chunk 长度贴近 chunk_size
            header_prefix = f"【{doc_title} | {section_title}】\n"
            effective_size = max(180, chunk_size - len(header_prefix))
            pieces = split_section_text(
                section_body,
                chunk_size=effective_size,
                chunk_overlap=chunk_overlap,
            )

            for piece in pieces:
                full_text = f"{header_prefix}{piece}"
                digest = hashlib.md5(
                    f"{source_name}:{doc_chunk_idx}:{full_text}".encode("utf-8")
                ).hexdigest()[:8]
                chunk_id = f"doc{doc_idx:02d}_c{doc_chunk_idx:03d}_{digest}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=full_text,
                        source=source_name,
                        section=section_title,
                        doc_title=doc_title,
                        chunk_index=doc_chunk_idx,
                    )
                )
                doc_chunk_idx += 1

    return all_chunks


def ingest_documents(
    settings: Settings | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    """执行文档切片、向量化并幂等写入本地 ChromaDB。

    Args:
        settings: 可选配置实例，默认读取全局 `.env` 配置。
        reset: 是否先清理同名旧集合再重建（默认 True，保证幂等无重复切片）。

    Returns:
        包含入库统计信息的字典。
    """
    cfg = settings or get_settings()
    docs_dir = cfg.docs_dir_path
    persist_dir = cfg.chroma_dir_path
    collection_name = cfg.CHROMA_COLLECTION_NAME

    chunks = build_chunks_from_docs(
        docs_dir=docs_dir,
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )
    if not chunks:
        raise RuntimeError("未从知识库文档中切分出任何有效内容。")

    logger.info(
        "已完成文档切片：共 %d 个切片（chunk_size=%d, chunk_overlap=%d），开始调用 Embedding 接口...",
        len(chunks),
        cfg.CHUNK_SIZE,
        cfg.CHUNK_OVERLAP,
    )

    embedder = EmbeddingClient(settings=cfg)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)
    embedding_dim = len(embeddings[0]) if embeddings else 0

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if reset:
        existing_names = [
            col.name if hasattr(col, "name") else str(col)
            for col in client.list_collections()
        ]
        if collection_name in existing_names:
            client.delete_collection(name=collection_name)
            logger.info("已清理旧集合 `%s`，准备幂等重建。", collection_name)

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[c.to_metadata() for c in chunks],
    )

    per_doc_counts: dict[str, int] = {}
    for c in chunks:
        per_doc_counts[c.source] = per_doc_counts.get(c.source, 0) + 1

    avg_len = round(sum(len(c.text) for c in chunks) / len(chunks), 1)
    max_len = max(len(c.text) for c in chunks)
    min_len = min(len(c.text) for c in chunks)

    return {
        "docs_dir": str(docs_dir),
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "reset": reset,
        "doc_count": len(per_doc_counts),
        "chunk_count": len(chunks),
        "collection_total": collection.count(),
        "embedding_model": cfg.EMBEDDING_MODEL,
        "embedding_dim": embedding_dim,
        "chunk_len_min": min_len,
        "chunk_len_avg": avg_len,
        "chunk_len_max": max_len,
        "per_doc_counts": per_doc_counts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="星云智联知识库 Markdown 切片与 ChromaDB 向量入库工具"
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="保留旧集合并执行增量 upsert（默认会先删除重建以保证幂等无重复）",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args()
    summary = ingest_documents(reset=not args.no_reset)

    print("\n" + "=" * 60)
    print("星云智联知识库向量入库完成！")
    print("=" * 60)
    print(f"知识库目录    : {summary['docs_dir']}")
    print(f"Chroma 持久化 : {summary['persist_dir']}")
    print(f"集合名称      : {summary['collection_name']} (幂等重建={summary['reset']})")
    print(f"Embedding 模型: {summary['embedding_model']} (维度={summary['embedding_dim']})")
    print(f"文档总数      : {summary['doc_count']} 份")
    print(f"入库切片总数  : {summary['chunk_count']} 条 (集合当前总数={summary['collection_total']})")
    print(
        f"切片字符长度  : min={summary['chunk_len_min']} / "
        f"avg={summary['chunk_len_avg']} / max={summary['chunk_len_max']}"
    )
    print("\n各文档切片分布：")
    for source, count in summary["per_doc_counts"].items():
        print(f"  - {source}: {count} 条")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
