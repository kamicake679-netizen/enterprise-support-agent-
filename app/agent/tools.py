"""Agent 业务工具层定义与执行调度模块（OpenAI Function Calling）。

包含 4 个核心业务工具及其 Pydantic 严格参数校验模型：
1. `search_knowledge_base(query: str)`：调用 Step 2 混合检索返回带出处的知识片段；
2. `query_order(order_id: str)`：查询 SQLite 业务库 `orders` 与关联工单状态；
3. `create_after_sales_ticket(order_id: str, issue_type: str, summary: str)`：创建售后工单并返回工单号；
4. `handoff_human(reason: str)`：知识库未命中、复杂纠纷或用户要求人工时的转人工兜底。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.db.mock_db import create_ticket, get_order_by_id, list_tickets_by_order
from app.rag.retriever import HybridRetriever, get_retriever

# 允许的标准售后问题类型（严格对齐 06_人工客服介入与工单升级标准.md）
IssueType = Literal[
    "quality_issue",
    "return_refund",
    "price_protection",
    "logistics",
    "safety_urgent",
    "complaint",
]

# 问题类型 -> 默认工单优先级与 SLA 映射
_ISSUE_PRIORITY_MAP: dict[str, tuple[str, str]] = {
    "safety_urgent": ("P0", "15 分钟内由安全专员电话回访"),
    "complaint": ("P1", "30 分钟内由高级客诉专员接管"),
    "logistics": ("P1", "2 小时内核实物流并反馈"),
    "return_refund": ("P2", "4 小时内人工审核并反馈"),
    "price_protection": ("P2", "4 小时内人工核对账务"),
    "quality_issue": ("P2", "24 小时内生成寄修/换新方案"),
}

# 拦截 LLM 在用户未提供订单号时可能臆造的占位字符串
_INVALID_ORDER_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "UNKNOWN",
        "NONE",
        "NULL",
        "N/A",
        "NA",
        "UNDEFINED",
        "ORDER_ID",
        "YOUR_ORDER_ID",
        "未知",
        "无",
        "未提供",
        "暂无",
        "不知道",
    }
)


# =============================================================================
# 1. Pydantic 严格参数校验模型
# =============================================================================


class SearchKnowledgeBaseArgs(BaseModel):
    """知识库检索工具参数校验模型。"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "要检索的星云智联产品参数、退换货政策、质保维修、故障排查、价保发票或人工升级规范查询词，"
            "建议包含具体产品型号（如 NL-Pro2026）或政策关键词。"
        ),
    )

    @field_validator("query", mode="before")
    @classmethod
    def _clean_query(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("query 必须为字符串类型")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query 不能为空白字符串")
        return cleaned


class QueryOrderArgs(BaseModel):
    """订单查询工具参数校验模型。"""

    order_id: str = Field(
        ...,
        min_length=4,
        max_length=32,
        description="星云智联真实订单号（由用户明确提供，通常以 NL 开头，例如 NL20261118006）。严禁编造或猜测。",
    )

    @field_validator("order_id", mode="before")
    @classmethod
    def _validate_order_id(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("order_id 必须为字符串类型")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("order_id 不能为空")
        if cleaned in _INVALID_ORDER_PLACEHOLDERS:
            raise ValueError("缺少有效订单号：严禁使用占位符，请先向用户礼貌追问真实订单号")
        return cleaned


class CreateAfterSalesTicketArgs(BaseModel):
    """创建售后工单工具参数校验模型。"""

    order_id: str = Field(
        ...,
        min_length=4,
        max_length=32,
        description="关联的星云智联真实订单号（如 NL20261118006），须先核实订单存在。",
    )
    issue_type: IssueType = Field(
        ...,
        description=(
            "售后问题类型，必须为以下 6 个枚举值之一："
            "quality_issue（质量故障）、return_refund（退换货申请与争议）、"
            "price_protection（价保/发票/账务）、logistics（物流异常与丢件）、"
            "safety_urgent（发烫/鼓包/过敏等安全隐患）、complaint（投诉升级）。"
        ),
    )
    summary: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="简明扼要的售后问题描述摘要，应包含用户核心诉求、设备型号及关键情况。",
    )

    @field_validator("order_id", mode="before")
    @classmethod
    def _validate_order_id(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("order_id 必须为字符串类型")
        cleaned = value.strip().upper()
        if not cleaned or cleaned in _INVALID_ORDER_PLACEHOLDERS:
            raise ValueError("创建工单需要真实有效的订单号，请先向用户确认订单号")
        return cleaned

    @field_validator("issue_type", mode="before")
    @classmethod
    def _normalize_issue_type(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("issue_type 必须为字符串类型")
        return value.strip().lower()

    @field_validator("summary", mode="before")
    @classmethod
    def _clean_summary(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("summary 必须为字符串类型")
        cleaned = value.strip()
        if len(cleaned) < 5:
            raise ValueError("summary 问题描述摘要至少需要 5 个字符")
        return cleaned


class HandoffHumanArgs(BaseModel):
    """转人工兜底工具参数校验模型。"""

    reason: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="需要转接人工客服的具体原因（例如：知识库未命中该问题、用户明确要求人工、安全隐患或费用争议协商）。",
    )

    @field_validator("reason", mode="before")
    @classmethod
    def _clean_reason(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("reason 必须为字符串类型")
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("reason 转人工原因不能少于 2 个字符")
        return cleaned


# =============================================================================
# 2. 工具核心业务执行函数
# =============================================================================


def search_knowledge_base(
    query: str,
    *,
    retrieval_mode: Literal["hybrid", "vector_only"] = "hybrid",
    top_k: int | None = None,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    """检索星云智联官方知识库并返回带出处的结构化知识片段。

    Args:
        query: 查询语句。
        retrieval_mode: `"hybrid"`（默认，向量 + BM25 + RRF + 相关性阈值过滤）
            或 `"vector_only"`（基线模式，仅走纯向量 Top-K）。
        top_k: 返回条数，默认读取全局配置 `FINAL_TOP_K`（3）。
        retriever: 可选外部注入的 `HybridRetriever` 实例（方便单测或复用）。

    Returns:
        结构化检索结果字典，包含 `found`、`count`、`results`、`message` 等字段。
    """
    validated = SearchKnowledgeBaseArgs(query=query)
    active_retriever = retriever or get_retriever()
    effective_top_k = top_k if top_k is not None else active_retriever.settings.FINAL_TOP_K

    if retrieval_mode == "vector_only":
        # Baseline 模式使用纯向量 Top-K 且关闭阈值拦截，便于 Step 5 量化对比优化效果
        hits = active_retriever.vector_only_search(
            query=validated.query,
            top_k=effective_top_k,
            score_threshold=0.0,
        )
    else:
        hits = active_retriever.search(
            query=validated.query,
            top_k=effective_top_k,
        )

    if not hits:
        return {
            "found": False,
            "query": validated.query,
            "retrieval_mode": retrieval_mode,
            "count": 0,
            "results": [],
            "message": (
                "知识库中未检索到与该问题相关的官方产品参数或政策条款。"
                "请如实告知用户未查到相关信息，严禁自行编造，必要时调用 handoff_human 转接人工客服。"
            ),
        }

    formatted_results: list[dict[str, Any]] = []
    for item in hits:
        formatted_results.append(
            {
                "source": item.get("source", ""),
                "section": item.get("section", ""),
                "text": item.get("text", ""),
                "score": item.get("score", 0.0),
                "confidence": item.get("confidence", "medium"),
                "chunk_id": item.get("chunk_id", ""),
            }
        )

    return {
        "found": True,
        "query": validated.query,
        "retrieval_mode": retrieval_mode,
        "count": len(formatted_results),
        "results": formatted_results,
        "message": f"共检索到 {len(formatted_results)} 条官方知识片段，回答时请严格依据内容并标注 [来源: 文件名]。",
    }


def _compute_days_since(time_str: str | None) -> float | None:
    """计算指定时间字符串距当前系统时间的天数（保留 1 位小数）。"""
    if not time_str:
        return None
    try:
        dt = datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    delta = datetime.now() - dt
    return round(max(0.0, delta.total_seconds() / 86400.0), 1)


def query_order(
    order_id: str,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """查询业务数据库中的订单详情、商品型号、下单与签收时间及关联工单。"""
    validated = QueryOrderArgs(order_id=order_id)
    order = get_order_by_id(validated.order_id, db_path=db_path)

    if not order:
        return {
            "found": False,
            "order_id": validated.order_id,
            "message": (
                f"未在系统中查到订单号 {validated.order_id}，"
                "请提醒用户核对订单号是否准确（星云智联订单号通常以 NL 开头）。"
            ),
        }

    existing_tickets = list_tickets_by_order(validated.order_id, db_path=db_path)
    days_since_delivered = _compute_days_since(order.get("delivered_time"))

    return {
        "found": True,
        "order_id": order["order_id"],
        "user_name": order["user_name"],
        "user_phone": order["user_phone"],
        "product_model": order["product_model"],
        "product_color": order["product_color"],
        "amount_paid": order["amount_paid"],
        "order_status": order["order_status"],
        "lifecycle_tag": order["lifecycle_tag"],
        "is_activated": bool(order["is_activated"]),
        "has_nebulacare": bool(order["has_nebulacare"]),
        "promo_tag": order["promo_tag"],
        "order_time": order["order_time"],
        "delivered_time": order["delivered_time"],
        "days_since_delivered": days_since_delivered,
        "tracking_number": order["tracking_number"],
        "existing_tickets": [
            {
                "ticket_id": t["ticket_id"],
                "issue_type": t["issue_type"],
                "priority": t["priority"],
                "status": t["status"],
                "created_at": t["created_at"],
            }
            for t in existing_tickets
        ],
    }


def create_after_sales_ticket(
    order_id: str,
    issue_type: str,
    summary: str,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """在 `tickets` 表中创建一条新的售后工单并返回工单号与 SLA 信息。"""
    validated = CreateAfterSalesTicketArgs(
        order_id=order_id,
        issue_type=issue_type,  # type: ignore[arg-type]
        summary=summary,
    )

    # 先校验订单存在，防止违反外键约束或为虚构订单建单
    order = get_order_by_id(validated.order_id, db_path=db_path)
    if not order:
        return {
            "created": False,
            "order_id": validated.order_id,
            "message": f"订单号 {validated.order_id} 不存在，无法创建售后工单，请先与用户核实真实订单号。",
        }

    priority, sla_tip = _ISSUE_PRIORITY_MAP.get(
        validated.issue_type, ("P2", "24 小时内由售后专员跟进处理")
    )

    ticket = create_ticket(
        order_id=validated.order_id,
        issue_type=validated.issue_type,
        description=validated.summary,
        priority=priority,
        db_path=db_path,
    )

    return {
        "created": True,
        "ticket_id": ticket["ticket_id"],
        "order_id": ticket["order_id"],
        "product_model": order["product_model"],
        "issue_type": ticket["issue_type"],
        "priority": ticket["priority"],
        "status": ticket["status"],
        "summary": ticket["description"],
        "created_at": ticket["created_at"],
        "sla_tip": sla_tip,
        "message": (
            f"售后工单 {ticket['ticket_id']} 已创建成功（优先级 {ticket['priority']}），"
            f"预计处理时效：{sla_tip}。"
        ),
    }


def handoff_human(reason: str) -> dict[str, Any]:
    """当知识库未命中、用户明确要求人工或遇到复杂纠纷/安全隐患时触发转人工兜底。"""
    validated = HandoffHumanArgs(reason=reason)
    handoff_id = f"HO{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    lower_reason = validated.reason.lower()
    if any(kw in lower_reason for kw in ("发烫", "冒烟", "鼓包", "烧焦", "漏液", "过敏", "安全")):
        queue_name = "安全应急专席（P0 最高优）"
        estimated_wait = "15 分钟内电话回访"
    elif any(kw in lower_reason for kw in ("投诉", "12315", "曝光", "起诉", "维权", "情绪")):
        queue_name = "高级客诉专席（P1 高优）"
        estimated_wait = "预计 30 分钟内专员介入"
    else:
        queue_name = "星云智联人工客服专席"
        estimated_wait = "工作时间（9:00-22:00）预计 3-5 分钟内接入"

    return {
        "handoff_triggered": True,
        "handoff_id": handoff_id,
        "queue": queue_name,
        "reason": validated.reason,
        "estimated_wait": estimated_wait,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": (
            f"已为您发起人工客服转接（排队凭据：{handoff_id}，接入队列：{queue_name}，{estimated_wait}）。"
        ),
    }


# =============================================================================
# 3. OpenAI Function Calling Schema 构造与统一执行器
# =============================================================================


def _clean_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """构建紧凑且高兼容性的 JSON Schema（兼容 OpenAI / Gemini / DeepSeek 等端点）。"""
    properties: dict[str, Any] = {}
    for prop_name, prop_def in (schema.get("properties") or {}).items():
        clean_prop: dict[str, Any] = {}
        if "type" in prop_def:
            clean_prop["type"] = prop_def["type"]
        if "description" in prop_def:
            clean_prop["description"] = prop_def["description"]
        if "enum" in prop_def:
            clean_prop["enum"] = prop_def["enum"]
        properties[prop_name] = clean_prop

    return {
        "type": "object",
        "properties": properties,
        "required": list(schema.get("required") or []),
    }


def _build_openai_tool(
    name: str,
    description: str,
    args_model: type[BaseModel],
) -> dict[str, Any]:
    """将 Pydantic 参数校验模型转换为标准 OpenAI Function Calling 工具定义。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _clean_json_schema(args_model.model_json_schema()),
        },
    }


OPENAI_TOOLS: list[dict[str, Any]] = [
    _build_openai_tool(
        name="search_knowledge_base",
        description=(
            "检索星云智联官方知识库（包含产品型号参数、七天无理由退换货政策、质保维修与延保、"
            "常见故障自助排查、大促价保与发票规则、人工客服介入与工单升级标准），返回带文件出处的知识片段。"
        ),
        args_model=SearchKnowledgeBaseArgs,
    ),
    _build_openai_tool(
        name="query_order",
        description=(
            "根据真实订单号查询星云智联业务数据库中的订单详情，包括订单状态、商品型号、实付金额、"
            "下单时间、签收时间（及距今天数）、是否激活、是否含 NebulaCare+ 延保、物流单号与已有工单。"
            "注意：若用户未提供订单号，请先礼貌追问，切勿猜测。"
        ),
        args_model=QueryOrderArgs,
    ),
    _build_openai_tool(
        name="create_after_sales_ticket",
        description=(
            "在售后工单系统（tickets 表）中为已核实存在的订单创建一条售后工单，并返回工单号与预计处理 SLA。"
            "仅在用户明确需要售后建单或符合升级标准且已获得真实订单号时调用。"
        ),
        args_model=CreateAfterSalesTicketArgs,
    ),
    _build_openai_tool(
        name="handoff_human",
        description=(
            "触发转接人工客服兜底流程。当知识库未命中用户问题、用户明确要求转人工、"
            "涉及发烫/鼓包等安全隐患、或出现复杂纠纷/超权限协商时调用。"
        ),
        args_model=HandoffHumanArgs,
    ),
]


@dataclass
class ToolExecutionResult:
    """单次工具调用的标准化执行封装，供 Agent 核心循环记录轨迹与提取出处。"""

    tool_name: str
    parsed_args: dict[str, Any]
    raw_result: dict[str, Any]
    result_json: str
    summary: str
    success: bool
    citations: list[dict[str, Any]] = field(default_factory=list)
    handoff_triggered: bool = False
    created_ticket_id: str | None = None


def _summarize_tool_result(tool_name: str, result: dict[str, Any], success: bool) -> str:
    """生成面向日志、评测与前端透明思考面板的简明出参摘要。"""
    if not success:
        return f"执行失败：{result.get('error') or result.get('message') or '未知错误'}"

    if tool_name == "search_knowledge_base":
        count = int(result.get("count", 0))
        if count == 0:
            return "未命中相关知识片段（已触发相关性阈值过滤）"
        sources = []
        for item in result.get("results") or []:
            src = str(item.get("source", ""))
            if src and src not in sources:
                sources.append(src)
        return f"命中 {count} 条知识片段，来源：{', '.join(sources)}"

    if tool_name == "query_order":
        if not result.get("found"):
            return f"未查到订单 {result.get('order_id', '')}"
        activated_str = "已激活" if result.get("is_activated") else "未激活"
        days = result.get("days_since_delivered")
        days_str = f"，签收约 {days} 天" if days is not None else ""
        return (
            f"查到订单 {result.get('order_id')}：{result.get('product_model')} | "
            f"状态={result.get('order_status')}（{result.get('lifecycle_tag')}）| "
            f"{activated_str}{days_str}"
        )

    if tool_name == "create_after_sales_ticket":
        if not result.get("created"):
            return f"建单未成功：{result.get('message', '')}"
        return (
            f"工单创建成功：{result.get('ticket_id')} | "
            f"类型={result.get('issue_type')} | 优先级={result.get('priority')}"
        )

    if tool_name == "handoff_human":
        return (
            f"已触发转人工：{result.get('handoff_id')} | "
            f"队列={result.get('queue')} | 原因={result.get('reason')}"
        )

    return str(result.get("message") or "执行完成")


def execute_tool(
    tool_name: str,
    raw_arguments: str | dict[str, Any],
    *,
    retrieval_mode: Literal["hybrid", "vector_only"] = "hybrid",
    retriever: HybridRetriever | None = None,
    db_path: str | Path | None = None,
) -> ToolExecutionResult:
    """解析参数、校验并执行指定业务工具，任何校验或运行异常均安全封装返回给 LLM。"""
    # 1. 解析 JSON 入参
    parsed_args: dict[str, Any]
    if isinstance(raw_arguments, dict):
        parsed_args = dict(raw_arguments)
    else:
        raw_str = (raw_arguments or "").strip() or "{}"
        try:
            loaded = json.loads(raw_str)
            if not isinstance(loaded, dict):
                raise ValueError("工具参数必须为 JSON 对象")
            parsed_args = loaded
        except Exception as exc:
            err_payload = {
                "error": f"工具参数 JSON 解析失败: {exc}",
                "raw_arguments": raw_str,
            }
            return ToolExecutionResult(
                tool_name=tool_name,
                parsed_args={},
                raw_result=err_payload,
                result_json=json.dumps(err_payload, ensure_ascii=False),
                summary=_summarize_tool_result(tool_name, err_payload, success=False),
                success=False,
            )

    # 2. 分发执行 + Pydantic 严格校验
    try:
        if tool_name == "search_knowledge_base":
            raw_result = search_knowledge_base(
                query=parsed_args.get("query", ""),
                retrieval_mode=retrieval_mode,
                retriever=retriever,
            )
            citations = list(raw_result.get("results") or [])
            return ToolExecutionResult(
                tool_name=tool_name,
                parsed_args=parsed_args,
                raw_result=raw_result,
                result_json=json.dumps(raw_result, ensure_ascii=False),
                summary=_summarize_tool_result(tool_name, raw_result, success=True),
                success=True,
                citations=citations,
            )

        if tool_name == "query_order":
            raw_result = query_order(
                order_id=parsed_args.get("order_id", ""),
                db_path=db_path,
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                parsed_args=parsed_args,
                raw_result=raw_result,
                result_json=json.dumps(raw_result, ensure_ascii=False),
                summary=_summarize_tool_result(tool_name, raw_result, success=True),
                success=True,
            )

        if tool_name == "create_after_sales_ticket":
            raw_result = create_after_sales_ticket(
                order_id=parsed_args.get("order_id", ""),
                issue_type=parsed_args.get("issue_type", ""),
                summary=parsed_args.get("summary", ""),
                db_path=db_path,
            )
            created_id = raw_result.get("ticket_id") if raw_result.get("created") else None
            return ToolExecutionResult(
                tool_name=tool_name,
                parsed_args=parsed_args,
                raw_result=raw_result,
                result_json=json.dumps(raw_result, ensure_ascii=False),
                summary=_summarize_tool_result(tool_name, raw_result, success=True),
                success=bool(raw_result.get("created")),
                created_ticket_id=created_id,
            )

        if tool_name == "handoff_human":
            raw_result = handoff_human(reason=parsed_args.get("reason", ""))
            return ToolExecutionResult(
                tool_name=tool_name,
                parsed_args=parsed_args,
                raw_result=raw_result,
                result_json=json.dumps(raw_result, ensure_ascii=False),
                summary=_summarize_tool_result(tool_name, raw_result, success=True),
                success=True,
                handoff_triggered=bool(raw_result.get("handoff_triggered")),
            )

        err_payload = {"error": f"未知的工具名称: {tool_name}"}
        return ToolExecutionResult(
            tool_name=tool_name,
            parsed_args=parsed_args,
            raw_result=err_payload,
            result_json=json.dumps(err_payload, ensure_ascii=False),
            summary=_summarize_tool_result(tool_name, err_payload, success=False),
            success=False,
        )

    except ValidationError as exc:
        err_msgs = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        err_payload = {
            "error": f"参数校验未通过: {err_msgs}",
            "hint": "请根据工具参数定义修正入参后再试；若缺少真实订单号，请先向用户追问。",
        }
        return ToolExecutionResult(
            tool_name=tool_name,
            parsed_args=parsed_args,
            raw_result=err_payload,
            result_json=json.dumps(err_payload, ensure_ascii=False),
            summary=_summarize_tool_result(tool_name, err_payload, success=False),
            success=False,
        )
    except Exception as exc:
        err_payload = {"error": f"工具执行异常: {exc}"}
        return ToolExecutionResult(
            tool_name=tool_name,
            parsed_args=parsed_args,
            raw_result=err_payload,
            result_json=json.dumps(err_payload, ensure_ascii=False),
            summary=_summarize_tool_result(tool_name, err_payload, success=False),
            success=False,
        )
