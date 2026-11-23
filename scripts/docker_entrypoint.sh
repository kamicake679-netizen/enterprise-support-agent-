#!/bin/sh
set -eu

SQLITE_DB_PATH="${SQLITE_DB_PATH:-data/business.db}"
CHROMA_PERSIST_DIR="${CHROMA_PERSIST_DIR:-data/chroma_db}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "============================================================"
echo "星云智联（NebulaLink）智能客服 Agent 容器启动中..."
echo "============================================================"

# 1. 自动初始化模拟业务数据库（如不存在）
if [ ! -f "$SQLITE_DB_PATH" ]; then
    echo "[Init 1/2] 未检测到业务数据库 ($SQLITE_DB_PATH)，正在初始化模拟订单与工单库..."
    python app/db/mock_db.py
else
    echo "[Init 1/2] 检测到已存在的业务数据库: $SQLITE_DB_PATH，跳过初始化。"
fi

# 2. 自动执行知识库切片与向量入库（如 ChromaDB 目录不存在或未初始化）
if [ ! -d "$CHROMA_PERSIST_DIR" ] || [ ! -f "$CHROMA_PERSIST_DIR/chroma.sqlite3" ]; then
    echo "[Init 2/2] 未检测到向量知识库 ($CHROMA_PERSIST_DIR)，正在执行文档切片与向量入库..."
    python -m app.rag.ingest
else
    echo "[Init 2/2] 检测到已存在的向量知识库: $CHROMA_PERSIST_DIR，跳过入库。"
fi

# 3. 启动 FastAPI 服务
echo "[Start] 正在启动 Uvicorn 服务 -> http://${HOST}:${PORT}"
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
