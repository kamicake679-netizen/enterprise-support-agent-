"""FastAPI 后端服务入口（星云智联企业级智能客服与知识库 Agent）。

提供核心接口：
- `GET /`：托管单文件“透明思考”前端演示页 `web/index.html`
- `GET /health`：健康检查（返回服务状态、文档切片数量、模拟订单数量与工单数量）
- `GET /api/orders`：返回前 10 条模拟订单号及状态摘要，供前端展示可测试订单清单
- `POST /api/chat`：接收 `{session_id, message}`，调用 Agent 并返回回答、引用溯源、工具调用轨迹和耗时
- `DELETE /api/sessions/{session_id}`：清空指定会话历史

内置：
- 内存级单 IP 滑动窗口频率限制（默认每分钟最多 20 次对话请求），防止公网演示额度被刷爆；
- 全局友好异常 JSON 拦截。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent.core import (
    AgentMode,
    CitationItem,
    ToolCallTrace,
    get_agent,
)
from app.config import BASE_DIR, get_settings
from app.db.mock_db import get_connection, init_db, list_orders
from app.rag.retriever import get_retriever

logger = logging.getLogger(__name__)

WEB_INDEX_PATH: Path = BASE_DIR / "web" / "index.html"

# =============================================================================
# 1. 内存级单 IP 频率限制器与会话存储
# =============================================================================


class InMemoryRateLimiter:
    """线程安全的内存级滑动窗口限流器（按客户端 IP 计数）。"""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60) -> None:
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(1, window_seconds)
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_gc_ts: float = time.monotonic()

    def check_and_record(self, client_ip: str) -> tuple[bool, int, int]:
        """检查并记录一次请求。

        Returns:
            `(allowed, remaining, retry_after_seconds)`
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # 定期清理长期不活跃的 IP 桶，防止内存增长
            if now - self._last_gc_ts > self.window_seconds * 2:
                stale_ips = [
                    ip for ip, dq in self._buckets.items() if not dq or dq[-1] < cutoff
                ]
                for ip in stale_ips:
                    del self._buckets[ip]
                self._last_gc_ts = now

            bucket = self._buckets.setdefault(client_ip, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])) + 1)
                return False, 0, retry_after

            bucket.append(now)
            remaining = max(0, self.max_requests - len(bucket))
            return True, remaining, 0


class InMemorySessionStore:
    """线程安全的内存级多轮会话历史存储（LRU 淘汰防止内存泄漏）。"""

    def __init__(self, max_sessions: int = 200, max_messages_per_session: int = 24) -> None:
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self._store: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """获取指定会话的历史消息副本。"""
        with self._lock:
            if session_id not in self._store:
                return []
            self._store.move_to_end(session_id)
            return [dict(item) for item in self._store[session_id]]

    def append_turn(self, session_id: str, user_message: str, assistant_answer: str) -> None:
        """向指定会话追加一轮问答并按上限裁剪。"""
        with self._lock:
            history = self._store.get(session_id, [])
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_answer})
            if len(history) > self.max_messages_per_session:
                history = history[-self.max_messages_per_session :]
            self._store[session_id] = history
            self._store.move_to_end(session_id)
            while len(self._store) > self.max_sessions:
                self._store.popitem(last=False)

    def clear_session(self, session_id: str) -> bool:
        """清空指定会话历史，返回是否存在该会话。"""
        with self._lock:
            return self._store.pop(session_id, None) is not None


rate_limiter = InMemoryRateLimiter(max_requests=20, window_seconds=60)
session_store = InMemorySessionStore(max_sessions=200, max_messages_per_session=24)


def _extract_client_ip(request: Request) -> str:
    """提取客户端真实 IP（兼容反向代理头）。"""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    real_ip = request.headers.get("x-real-ip")
    if real_ip and real_ip.strip():
        return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


# =============================================================================
# 2. 请求与响应数据模型
# =============================================================================


class ChatRequest(BaseModel):
    """对话请求体。"""

    session_id: str = Field(
        default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}",
        min_length=1,
        max_length=128,
        description="会话唯一标识，用于维持多轮上下文",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户当前输入的问题",
    )
    mode: AgentMode = Field(
        default="optimized",
        description="运行模式：optimized（混合检索+强溯源）或 baseline（纯向量基线）",
    )


class ChatResponse(BaseModel):
    """对话结构化响应体（含透明思考面板所需完整元数据）。"""

    session_id: str = Field(..., description="当前会话 ID")
    answer: str = Field(..., description="Agent 最终自然语言回复")
    citations: list[CitationItem] = Field(
        default_factory=list, description="命中的知识库引用溯源片段"
    )
    tool_calls_trace: list[ToolCallTrace] = Field(
        default_factory=list, description="按步骤排列的工具调用轨迹"
    )
    thinking_events: list[dict[str, Any]] = Field(
        default_factory=list, description="本轮 Agent 透明思考事件流"
    )
    latency_ms: int = Field(..., description="端到端响应耗时（毫秒）")
    mode: AgentMode = Field(default="optimized", description="本次运行模式")
    steps_used: int = Field(default=1, description="本轮消耗的推理步数")
    handoff_triggered: bool = Field(default=False, description="是否触发人工客服转接")
    created_ticket_ids: list[str] = Field(
        default_factory=list, description="本轮成功创建的售后工单号列表"
    )


class OrderSummary(BaseModel):
    """模拟订单摘要模型（供前端可测试订单清单展示）。"""

    order_id: str
    user_name: str
    product_model: str
    product_color: str
    amount_paid: float
    order_status: str
    lifecycle_tag: str
    is_activated: bool
    has_nebulacare: bool
    promo_tag: str
    order_time: str


class OrdersResponse(BaseModel):
    """订单摘要列表响应模型。"""

    total: int
    returned: int
    orders: list[OrderSummary]


class HealthResponse(BaseModel):
    """健康检查响应模型。"""

    status: Literal["ok", "degraded"]
    service: str
    chunk_count: int
    order_count: int
    ticket_count: int
    llm_model: str
    rerank_enabled: bool
    warnings: list[str] = Field(default_factory=list)


# =============================================================================
# 3. FastAPI 应用与生命周期管理
# =============================================================================


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用启动时幂等初始化模拟数据库并预热检索器与 Agent 单例。"""
    try:
        db_path = init_db(reset=False)
        logger.info("业务数据库已就绪: %s", db_path)
    except Exception as exc:
        logger.warning("启动初始化 SQLite 失败: %s", exc)

    try:
        retriever = get_retriever()
        logger.info("混合检索器预热完成，共加载 %d 条切片。", retriever.total_chunks)
    except Exception as exc:
        logger.warning("混合检索器预热失败（请确认已入库 ChromaDB）: %s", exc)

    try:
        agent = get_agent()
        logger.info("SupportAgent 预热完成，当前模型: %s", agent.model)
    except Exception as exc:
        logger.warning("SupportAgent 预热失败（请检查 .env 中 LLM_API_KEY 配置）: %s", exc)

    yield


app = FastAPI(
    title="星云智联（NebulaLink）企业级智能客服 Agent API",
    description="基于 Hybrid RAG（向量 + BM25 + RRF）与 ReAct Function Calling 的透明思考智能客服系统。",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 4. 全局友好异常处理
# =============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """将参数校验错误转换为清晰的中文 JSON。"""
    first_err = exc.errors()[0] if exc.errors() else {}
    field_path = ".".join(str(x) for x in first_err.get("loc", []))
    msg = first_err.get("msg", "请求参数格式不正确")
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": f"请求参数校验失败（{field_path}）：{msg}",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """统一 HTTPException 的 JSON 返回格式。"""
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {
            "error": f"http_{exc.status_code}",
            "message": str(exc.detail),
        }
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """兜底拦截未处理异常，避免暴露内部堆栈。"""
    logger.exception("未处理的后端异常: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "服务内部处理出现临时异常，请稍后重试。",
        },
    )


# =============================================================================
# 5. 核心路由接口
# =============================================================================


@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """托管单文件“透明思考”前端演示页。"""
    if not WEB_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "frontend_not_found",
                "message": "前端页面 web/index.html 尚未创建，请先完成前端文件部署。",
            },
        )
    return FileResponse(path=str(WEB_INDEX_PATH), media_type="text/html; charset=utf-8")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """健康检查：返回服务状态、知识库切片数、模拟订单数与工单数。"""
    settings = get_settings()
    warnings: list[str] = []
    chunk_count = 0
    order_count = 0
    ticket_count = 0

    try:
        retriever = get_retriever()
        chunk_count = retriever.total_chunks
    except Exception as exc:
        warnings.append(f"知识库检索器未就绪: {exc}")

    try:
        init_db(reset=False)
        with get_connection() as conn:
            order_count = int(conn.execute("SELECT COUNT(*) FROM orders;").fetchone()[0])
            ticket_count = int(conn.execute("SELECT COUNT(*) FROM tickets;").fetchone()[0])
    except Exception as exc:
        warnings.append(f"模拟业务数据库未就绪: {exc}")

    status: Literal["ok", "degraded"] = "ok" if not warnings else "degraded"
    return HealthResponse(
        status=status,
        service="NebulaLink Enterprise Support Agent",
        chunk_count=chunk_count,
        order_count=order_count,
        ticket_count=ticket_count,
        llm_model=settings.LLM_MODEL,
        rerank_enabled=settings.RERANK_ENABLED,
        warnings=warnings,
    )


@app.get("/api/orders", response_model=OrdersResponse, tags=["Business"])
async def get_demo_orders(
    limit: int = Query(default=10, ge=1, le=20, description="返回模拟订单条数，默认 10 条"),
) -> OrdersResponse:
    """返回前 N 条（默认 10 条）模拟订单号及状态摘要，供前端展示可测试订单清单。"""
    try:
        init_db(reset=False)
        with get_connection() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM orders;").fetchone()[0])
        raw_orders = list_orders(limit=limit)
    except Exception as exc:
        logger.exception("读取模拟订单失败: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "db_query_failed",
                "message": "读取模拟订单清单失败，请检查业务数据库状态。",
            },
        ) from exc

    summaries = [
        OrderSummary(
            order_id=str(o["order_id"]),
            user_name=str(o["user_name"]),
            product_model=str(o["product_model"]),
            product_color=str(o["product_color"]),
            amount_paid=float(o["amount_paid"]),
            order_status=str(o["order_status"]),
            lifecycle_tag=str(o["lifecycle_tag"]),
            is_activated=bool(o["is_activated"]),
            has_nebulacare=bool(o["has_nebulacare"]),
            promo_tag=str(o["promo_tag"]),
            order_time=str(o["order_time"]),
        )
        for o in raw_orders
    ]
    return OrdersResponse(total=total, returned=len(summaries), orders=summaries)


@app.post("/api/chat", response_model=ChatResponse, tags=["Agent"])
async def chat_with_agent(payload: ChatRequest, request: Request) -> JSONResponse:
    """核心客服对话接口：含单 IP 限流、多轮历史管理、透明思考轨迹与引用溯源。"""
    client_ip = _extract_client_ip(request)
    allowed, remaining, retry_after = rate_limiter.check_and_record(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": (
                    f"请求过于频繁（单 IP 每分钟最多 {rate_limiter.max_requests} 次对话请求），"
                    f"请在 {retry_after} 秒后再试。"
                ),
                "retry_after_seconds": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(rate_limiter.max_requests),
                "X-RateLimit-Remaining": "0",
            },
        )

    cleaned_message = payload.message.strip()
    if not cleaned_message:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_message",
                "message": "请输入有效的咨询内容，不能为空白字符。",
            },
        )

    session_id = payload.session_id.strip() or f"sess_{uuid.uuid4().hex[:12]}"
    history = session_store.get_history(session_id)
    thinking_events: list[dict[str, Any]] = []

    def _collect_event(event: dict[str, Any]) -> None:
        thinking_events.append(event)

    try:
        agent = get_agent()
        agent_resp = await asyncio.to_thread(
            agent.run,
            cleaned_message,
            history=history,
            mode=payload.mode,
            event_callback=_collect_event,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "agent_not_configured",
                "message": f"Agent 服务尚未完成模型密钥配置：{exc}",
            },
        ) from exc
    except RuntimeError as exc:
        logger.error("Agent 调用上游模型失败: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_llm_unavailable",
                "message": "上游大模型接口当前繁忙或暂时不可用，请稍后重试。",
            },
        ) from exc
    except Exception as exc:
        logger.exception("Agent 处理对话时发生异常: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "agent_execution_error",
                "message": "智能客服处理请求时遇到内部异常，请稍后重试。",
            },
        ) from exc

    session_store.append_turn(session_id, cleaned_message, agent_resp.answer)

    resp_model = ChatResponse(
        session_id=session_id,
        answer=agent_resp.answer,
        citations=agent_resp.citations,
        tool_calls_trace=agent_resp.tool_calls_trace,
        thinking_events=thinking_events,
        latency_ms=agent_resp.latency_ms,
        mode=agent_resp.mode,
        steps_used=agent_resp.steps_used,
        handoff_triggered=agent_resp.handoff_triggered,
        created_ticket_ids=agent_resp.created_ticket_ids,
    )
    return JSONResponse(
        status_code=200,
        content=resp_model.model_dump(),
        headers={
            "X-RateLimit-Limit": str(rate_limiter.max_requests),
            "X-RateLimit-Remaining": str(remaining),
        },
    )


@app.post("/api/chat/stream", tags=["Agent"])
async def chat_with_agent_stream(payload: ChatRequest, request: Request):
    """SSE 实时流式对话接口：实时推送思考步骤、工具调用轨迹、命中片段与打字机流式回复。"""
    client_ip = _extract_client_ip(request)
    allowed, remaining, retry_after = rate_limiter.check_and_record(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": (
                    f"请求过于频繁（单 IP 每分钟最多 {rate_limiter.max_requests} 次对话请求），"
                    f"请在 {retry_after} 秒后再试。"
                ),
                "retry_after_seconds": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(rate_limiter.max_requests),
                "X-RateLimit-Remaining": "0",
            },
        )

    cleaned_message = payload.message.strip()
    if not cleaned_message:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_message",
                "message": "请输入有效的咨询内容，不能为空白字符。",
            },
        )

    session_id = payload.session_id.strip() or f"sess_{uuid.uuid4().hex[:12]}"
    history = session_store.get_history(session_id)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    thinking_events: list[dict[str, Any]] = []

    def _on_event(event: dict[str, Any]) -> None:
        if event.get("type") != "token":
            thinking_events.append(event)
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run_worker() -> None:
        try:
            agent = get_agent()
            agent_resp = agent.run(
                cleaned_message,
                history=history,
                mode=payload.mode,
                event_callback=_on_event,
                stream_answer=True,
            )
            session_store.append_turn(session_id, cleaned_message, agent_resp.answer)
            resp_model = ChatResponse(
                session_id=session_id,
                answer=agent_resp.answer,
                citations=agent_resp.citations,
                tool_calls_trace=agent_resp.tool_calls_trace,
                thinking_events=thinking_events,
                latency_ms=agent_resp.latency_ms,
                mode=agent_resp.mode,
                steps_used=agent_resp.steps_used,
                handoff_triggered=agent_resp.handoff_triggered,
                created_ticket_ids=agent_resp.created_ticket_ids,
            )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "done", "data": resp_model.model_dump()},
            )
        except ValueError as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "message": f"Agent 服务尚未完成模型密钥配置：{exc}"},
            )
        except RuntimeError as exc:
            logger.error("流式调用上游模型失败: %s", exc)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "message": "上游大模型接口当前繁忙或暂时不可用，请稍后重试。"},
            )
        except Exception as exc:
            logger.exception("流式对话处理异常: %s", exc)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "message": "智能客服处理请求时遇到内部异常，请稍后重试。"},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def _event_generator():
        worker_task = asyncio.create_task(asyncio.to_thread(_run_worker))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            await worker_task

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-RateLimit-Limit": str(rate_limiter.max_requests),
            "X-RateLimit-Remaining": str(remaining),
        },
    )


@app.delete("/api/sessions/{session_id}", tags=["Agent"])
async def reset_session(session_id: str) -> dict[str, Any]:
    """清空指定会话的上下文历史，开启全新对话。"""
    cleared = session_store.clear_session(session_id.strip())
    return {
        "session_id": session_id.strip(),
        "cleared": cleared,
        "message": "会话上下文已重置。" if cleared else "该会话暂无历史记录，已准备就绪。",
    }
