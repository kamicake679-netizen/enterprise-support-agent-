"""Agent 核心调度循环模块（ReAct + OpenAI Function Calling）。

核心能力：
1. 通用 OpenAI 兼容 `LLM_*` 接口调用，内置 429 限流 / 5xx / 网络超时指数退避重试；
2. 健壮性保障：`max_steps=5` 防死循环（到达上限时自动收尾总结），对话历史滑动窗口（默认保留最近 6 轮）；
3. 完整结构化输出：`answer`、`citations`、`tool_calls_trace`、`latency_ms` 等；
4. 支持 `mode="optimized"` 与 `mode="baseline"` 一键切换，方便 Step 5 评测对比；
5. 预留 `event_callback` 回调钩子，方便 Step 4 FastAPI SSE 透明思考前端直接对接。
"""

from __future__ import annotations

import logging
import random
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field

from app.agent.prompts import build_system_prompt
from app.agent.tools import OPENAI_TOOLS, execute_tool
from app.config import Settings, get_settings
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

AgentMode = Literal["optimized", "baseline"]
EventCallback = Callable[[dict[str, Any]], None]


# =============================================================================
# 1. 结构化响应数据模型
# =============================================================================


class CitationItem(BaseModel):
    """命中的知识库引用片段。"""

    source: str = Field(..., description="来源文档文件名")
    section: str = Field(default="", description="命中的章节路径")
    text: str = Field(..., description="切片正文内容")
    score: float = Field(default=0.0, description="综合检索得分")
    confidence: str = Field(default="medium", description="置信度级别（high/medium/low）")
    chunk_id: str = Field(default="", description="切片唯一 ID")


class ToolCallTrace(BaseModel):
    """单次工具调用执行轨迹。"""

    step: int = Field(..., description="所属推理步数（从 1 开始）")
    tool_name: str = Field(..., description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具入参字典")
    result_summary: str = Field(..., description="工具出参摘要")
    success: bool = Field(default=True, description="本次调用是否成功")
    latency_ms: int = Field(default=0, description="本次工具执行耗时（毫秒）")


class AgentResponse(BaseModel):
    """Agent 单轮对话完整结构化响应。"""

    answer: str = Field(..., description="最终面向用户的自然语言回复")
    citations: list[CitationItem] = Field(
        default_factory=list, description="本轮命中的去重知识库引用列表"
    )
    tool_calls_trace: list[ToolCallTrace] = Field(
        default_factory=list, description="按执行顺序排列的工具调用轨迹"
    )
    latency_ms: int = Field(..., description="整轮 Agent 调度总耗时（毫秒）")
    mode: AgentMode = Field(default="optimized", description="本次运行模式（optimized/baseline）")
    steps_used: int = Field(default=1, description="本轮消耗的推理步数")
    handoff_triggered: bool = Field(default=False, description="本轮是否触发了人工转接")
    created_ticket_ids: list[str] = Field(
        default_factory=list, description="本轮成功创建的售后工单号列表"
    )


# =============================================================================
# 2. Agent 核心编排类
# =============================================================================


class SupportAgent:
    """星云智联企业级智能客服 Agent 主调度器。"""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        max_steps: int | None = None,
        history_window: int | None = None,
        max_retries: int | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        api_key = self.settings.LLM_API_KEY.strip()
        if not api_key:
            raise ValueError(
                "LLM_API_KEY 未配置，请先在根目录 .env 文件中填写有效的对话大模型 API Key。"
            )

        self.model = self.settings.LLM_MODEL
        self.fallback_model = self.settings.LLM_FALLBACK_MODEL.strip()
        self.fallback_cooldown_seconds: float = 60.0
        self._primary_cooldown_until: float = 0.0
        self.temperature = self.settings.LLM_TEMPERATURE
        self.max_tokens = self.settings.LLM_MAX_TOKENS
        self.max_steps = max(1, max_steps if max_steps is not None else self.settings.AGENT_MAX_STEPS)
        self.history_window = max(
            1, history_window if history_window is not None else self.settings.AGENT_HISTORY_WINDOW
        )
        self.max_retries = max(
            1, max_retries if max_retries is not None else self.settings.LLM_MAX_RETRIES
        )
        self.retriever = retriever
        self.db_path = db_path

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.settings.LLM_BASE_URL,
            timeout=30.0,
            max_retries=0,
        )

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        """从 HTTP 响应头解析 Retry-After 秒数。"""
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

    @staticmethod
    def _emit_event(callback: EventCallback | None, event: dict[str, Any]) -> None:
        """安全发射透明思考过程事件，回调异常不影响主链路。"""
        if callback is None:
            return
        try:
            callback(event)
        except Exception as exc:
            logger.debug("event_callback 回调异常已忽略: %s", exc)

    def _trim_history(
        self,
        history: Sequence[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """对话历史滑动窗口裁剪：仅保留最近 N 轮（默认 6 轮）有效 user/assistant 对话。"""
        if not history:
            return []

        cleaned: list[dict[str, Any]] = []
        for item in history:
            role = str(item.get("role", "")).strip()
            if role not in {"user", "assistant"}:
                continue
            content = item.get("content")
            if content is None:
                continue
            text = str(content).strip()
            if not text:
                continue
            cleaned.append({"role": role, "content": text})

        if not cleaned:
            return []

        # 以 user 消息为轮次边界，截取最近 `self.history_window` 轮对话
        user_indices = [idx for idx, msg in enumerate(cleaned) if msg["role"] == "user"]
        if len(user_indices) > self.history_window:
            start_idx = user_indices[-self.history_window]
            cleaned = cleaned[start_idx:]

        # 若最前面残留了孤立的 assistant 消息则剥离
        while cleaned and cleaned[0]["role"] != "user":
            cleaned.pop(0)

        return cleaned

    @staticmethod
    def _serialize_assistant_message(message: Any) -> dict[str, Any]:
        """将 OpenAI SDK 返回的 assistant 消息安全序列化为可回传字典。

        保留可能存在的扩展字段（例如部分 Gemini 兼容端点要求回传的 `extra_content` / `thought_signature`），
        同时清理空值，确保 OpenAI / Gemini / DeepSeek / Qwen 均可稳定多轮调用。
        """
        dumped: dict[str, Any] = {}
        if hasattr(message, "model_dump"):
            raw = message.model_dump(exclude_none=True)
            dumped["role"] = raw.get("role", "assistant")
            if "content" in raw:
                dumped["content"] = raw["content"]
            else:
                dumped["content"] = ""
            if raw.get("tool_calls"):
                dumped["tool_calls"] = raw["tool_calls"]
            # 透传厂商兼容字段（如 Gemini 的 extra_content、DeepSeek 思考链的 reasoning_content）
            if "extra_content" in raw:
                dumped["extra_content"] = raw["extra_content"]
            if "reasoning_content" in raw:
                dumped["reasoning_content"] = raw["reasoning_content"]
            return dumped

        dumped["role"] = getattr(message, "role", "assistant")
        dumped["content"] = getattr(message, "content", None) or ""
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            dumped["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return dumped

    def _chat_completion_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        use_tools: bool = True,
    ) -> Any:
        """发起带指数退避重试的 OpenAI 兼容 Chat Completion 请求（支持可选备用模型自动无缝降级）。"""
        has_fallback = bool(self.fallback_model and self.fallback_model != self.model)
        now_mono = time.monotonic()

        if has_fallback and now_mono < self._primary_cooldown_until:
            # 主模型处于短暂过载冷却期内，优先走备用模型，冷却结束后自动探测恢复主模型
            candidate_models = [self.fallback_model, self.model]
        elif has_fallback:
            candidate_models = [self.model, self.fallback_model]
        else:
            candidate_models = [self.model]

        last_error: Exception | None = None

        for model_idx, current_model in enumerate(candidate_models):
            is_last_model = model_idx + 1 >= len(candidate_models)
            # 当后面还有备用模型时，当前模型最多快速重试 3 次（约 6 秒）即无缝切备用，避免用户长等待
            attempts_for_model = self.max_retries if is_last_model else min(3, self.max_retries)

            request_kwargs: dict[str, Any] = {
                "model": current_model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if use_tools:
                request_kwargs["tools"] = OPENAI_TOOLS
                request_kwargs["tool_choice"] = "auto"

            for attempt in range(1, attempts_for_model + 1):
                try:
                    return self._client.chat.completions.create(**request_kwargs)
                except RateLimitError as exc:
                    last_error = exc
                    if attempt < attempts_for_model:
                        retry_after = self._extract_retry_after(exc)
                        base_wait = (
                            retry_after if retry_after is not None else min(45.0, 2.0 ** attempt)
                        )
                        wait_seconds = base_wait + random.uniform(0.2, 1.0)
                        logger.warning(
                            "LLM(%s) 触发 429 限流（第 %d/%d 次），%.2f 秒后重试...",
                            current_model,
                            attempt,
                            attempts_for_model,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                    else:
                        logger.warning(
                            "LLM(%s) 连续 %d 次触发 429 限流，本模型重试耗尽。",
                            current_model,
                            attempts_for_model,
                        )
                except (APITimeoutError, APIConnectionError) as exc:
                    last_error = exc
                    if attempt < attempts_for_model:
                        wait_seconds = min(30.0, 1.5 ** attempt) + random.uniform(0.1, 0.6)
                        logger.warning(
                            "LLM(%s) 网络/超时波动（第 %d/%d 次）：%s，%.2f 秒后重试...",
                            current_model,
                            attempt,
                            attempts_for_model,
                            exc,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                    else:
                        logger.warning(
                            "LLM(%s) 连续 %d 次网络/超时异常，本模型重试耗尽。",
                            current_model,
                            attempts_for_model,
                        )
                except APIStatusError as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", 0)
                    if status_code == 429 or status_code >= 500:
                        if attempt < attempts_for_model:
                            retry_after = self._extract_retry_after(exc)
                            base_wait = (
                                retry_after
                                if retry_after is not None
                                else min(45.0, 2.0 ** attempt)
                            )
                            wait_seconds = base_wait + random.uniform(0.2, 0.8)
                            logger.warning(
                                "LLM(%s) 返回状态码 %s（第 %d/%d 次），%.2f 秒后重试...",
                                current_model,
                                status_code,
                                attempt,
                                attempts_for_model,
                                wait_seconds,
                            )
                            time.sleep(wait_seconds)
                        else:
                            logger.warning(
                                "LLM(%s) 连续 %d 次返回状态码 %s，本模型重试耗尽。",
                                current_model,
                                attempts_for_model,
                                status_code,
                            )
                    else:
                        raise

            if not is_last_model:
                next_model = candidate_models[model_idx + 1]
                if current_model == self.model:
                    self._primary_cooldown_until = time.monotonic() + self.fallback_cooldown_seconds
                logger.warning(
                    "模型 %s 持续不可用，自动无缝切换至备用模型 %s（主模型冷却 %.0f 秒后自动探测恢复）...",
                    current_model,
                    next_model,
                    self.fallback_cooldown_seconds,
                )

        raise RuntimeError(
            f"LLM 对话接口（候选模型：{', '.join(candidate_models)}）重试耗尽后仍失败: {last_error}"
        ) from last_error

    def _chat_completion_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        use_tools: bool = True,
        step: int = 1,
        event_callback: EventCallback | None = None,
    ) -> Any:
        """发起流式 Chat Completion 请求：纯文本回答实时推送 token，tool_calls 自动增量拼装。"""
        has_fallback = bool(self.fallback_model and self.fallback_model != self.model)
        now_mono = time.monotonic()

        if has_fallback and now_mono < self._primary_cooldown_until:
            candidate_models = [self.fallback_model, self.model]
        elif has_fallback:
            candidate_models = [self.model, self.fallback_model]
        else:
            candidate_models = [self.model]

        last_error: Exception | None = None

        for model_idx, current_model in enumerate(candidate_models):
            is_last_model = model_idx + 1 >= len(candidate_models)
            attempts_for_model = self.max_retries if is_last_model else min(3, self.max_retries)

            request_kwargs: dict[str, Any] = {
                "model": current_model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }
            if use_tools:
                request_kwargs["tools"] = OPENAI_TOOLS
                request_kwargs["tool_choice"] = "auto"

            for attempt in range(1, attempts_for_model + 1):
                emitted_any_token = False
                try:
                    stream = self._client.chat.completions.create(**request_kwargs)
                    content_parts: list[str] = []
                    tc_builders: dict[int, dict[str, Any]] = {}

                    for chunk in stream:
                        choices = getattr(chunk, "choices", None)
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)
                        if delta is None:
                            continue

                        delta_tool_calls = getattr(delta, "tool_calls", None)
                        if delta_tool_calls:
                            for tc_delta in delta_tool_calls:
                                idx = getattr(tc_delta, "index", 0)
                                idx = 0 if idx is None else int(idx)
                                entry = tc_builders.setdefault(
                                    idx,
                                    {
                                        "id": "",
                                        "type": "function",
                                        "name": "",
                                        "arguments": [],
                                    },
                                )
                                if getattr(tc_delta, "id", None):
                                    entry["id"] = str(tc_delta.id)
                                if getattr(tc_delta, "type", None):
                                    entry["type"] = str(tc_delta.type)
                                fn_delta = getattr(tc_delta, "function", None)
                                if fn_delta is not None:
                                    if getattr(fn_delta, "name", None):
                                        entry["name"] += str(fn_delta.name)
                                    if getattr(fn_delta, "arguments", None):
                                        entry["arguments"].append(str(fn_delta.arguments))

                        delta_content = getattr(delta, "content", None)
                        if delta_content:
                            text_piece = str(delta_content)
                            content_parts.append(text_piece)
                            # 若当前未进入 tool_calls，实时向前端推送打字机增量文本
                            if not tc_builders:
                                emitted_any_token = True
                                self._emit_event(
                                    event_callback,
                                    {
                                        "type": "token",
                                        "step": step,
                                        "content": text_piece,
                                    },
                                )

                    assembled_tool_calls: list[Any] = []
                    for idx in sorted(tc_builders.keys()):
                        item = tc_builders[idx]
                        assembled_tool_calls.append(
                            SimpleNamespace(
                                id=item["id"] or f"call_{step}_{idx}",
                                type=item["type"] or "function",
                                function=SimpleNamespace(
                                    name=item["name"],
                                    arguments="".join(item["arguments"]),
                                ),
                            )
                        )

                    return SimpleNamespace(
                        role="assistant",
                        content="".join(content_parts),
                        tool_calls=assembled_tool_calls or None,
                    )
                except RateLimitError as exc:
                    last_error = exc
                    if emitted_any_token or attempt >= attempts_for_model:
                        break
                    retry_after = self._extract_retry_after(exc)
                    base_wait = (
                        retry_after if retry_after is not None else min(45.0, 2.0 ** attempt)
                    )
                    time.sleep(base_wait + random.uniform(0.2, 1.0))
                except (APITimeoutError, APIConnectionError) as exc:
                    last_error = exc
                    if emitted_any_token or attempt >= attempts_for_model:
                        break
                    time.sleep(min(30.0, 1.5 ** attempt) + random.uniform(0.1, 0.6))
                except APIStatusError as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", 0)
                    if emitted_any_token:
                        raise
                    if status_code == 429 or status_code >= 500:
                        if attempt >= attempts_for_model:
                            break
                        retry_after = self._extract_retry_after(exc)
                        base_wait = (
                            retry_after if retry_after is not None else min(45.0, 2.0 ** attempt)
                        )
                        time.sleep(base_wait + random.uniform(0.2, 0.8))
                    else:
                        raise

            if not is_last_model:
                if current_model == self.model:
                    self._primary_cooldown_until = time.monotonic() + self.fallback_cooldown_seconds

        raise RuntimeError(
            f"LLM 流式对话接口（候选模型：{', '.join(candidate_models)}）重试耗尽后仍失败: {last_error}"
        ) from last_error

    def run(
        self,
        user_message: str,
        *,
        history: Sequence[dict[str, Any]] | None = None,
        mode: AgentMode = "optimized",
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        stream_answer: bool = False,
    ) -> AgentResponse:
        """执行单轮 ReAct + Function Calling Agent 调度循环。

        Args:
            user_message: 用户当前输入文本。
            history: 可选历史对话列表（格式 `[{"role": "user"|"assistant", "content": "..."}]`），
                内部自动执行滑动窗口裁剪（默认保留最近 6 轮）。
            mode: `"optimized"`（混合检索 + 阈值过滤 + 防幻觉强溯源 Prompt）
                或 `"baseline"`（纯向量检索 + 极简基线 Prompt）。
            max_steps: 可选覆盖最大推理/工具调用步数（默认 5）。
            event_callback: 可选事件流回调，用于实时观测思考步骤、工具调用与流式 token。
            stream_answer: 为 True 时启用 OpenAI 流式输出，通过 `event_callback` 实时推送 `token` 事件。

        Returns:
            `AgentResponse` 完整结构化响应对象。
        """
        start_ts = time.perf_counter()
        cleaned_user_msg = (user_message or "").strip()
        if not cleaned_user_msg:
            return AgentResponse(
                answer="您好！请问有什么关于星云智联耳机产品、订单或售后问题可以帮您？",
                citations=[],
                tool_calls_trace=[],
                latency_ms=0,
                mode=mode,
                steps_used=0,
            )

        step_limit = max(1, max_steps if max_steps is not None else self.max_steps)
        retrieval_mode: Literal["hybrid", "vector_only"] = (
            "vector_only" if mode == "baseline" else "hybrid"
        )

        # 构造上下文消息：System Prompt + 最近 N 轮滑动窗口历史 + 当前用户输入
        system_prompt = build_system_prompt(mode=mode)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._trim_history(history))
        messages.append({"role": "user", "content": cleaned_user_msg})

        citations_map: dict[str, CitationItem] = {}
        tool_traces: list[ToolCallTrace] = []
        created_ticket_ids: list[str] = []
        handoff_triggered = False
        final_answer = ""
        steps_used = 0

        for step in range(1, step_limit + 1):
            steps_used = step
            self._emit_event(
                event_callback,
                {
                    "type": "thinking",
                    "step": step,
                    "message": (
                        f"第 {step} 步：正在分析意图并决策调用工具..."
                        if step == 1
                        else f"第 {step} 步：正在结合工具结果生成答复..."
                    ),
                },
            )

            if stream_answer:
                assistant_msg = self._chat_completion_stream_with_retry(
                    messages,
                    use_tools=True,
                    step=step,
                    event_callback=event_callback,
                )
            else:
                completion = self._chat_completion_with_retry(messages, use_tools=True)
                assistant_msg = completion.choices[0].message
            tool_calls = getattr(assistant_msg, "tool_calls", None) or []

            # 若模型未发起工具调用，直接产出最终回答
            if not tool_calls:
                final_answer = (getattr(assistant_msg, "content", None) or "").strip()
                break

            # 将包含 tool_calls 的 assistant 消息写回上下文
            messages.append(self._serialize_assistant_message(assistant_msg))

            # 顺序执行本轮所有 tool_calls
            for tc in tool_calls:
                fn_name = tc.function.name
                raw_args = tc.function.arguments or "{}"

                self._emit_event(
                    event_callback,
                    {
                        "type": "tool_start",
                        "step": step,
                        "tool_name": fn_name,
                        "raw_arguments": raw_args,
                    },
                )

                tool_t0 = time.perf_counter()
                exec_res = execute_tool(
                    tool_name=fn_name,
                    raw_arguments=raw_args,
                    retrieval_mode=retrieval_mode,
                    retriever=self.retriever,
                    db_path=self.db_path,
                )
                tool_ms = int((time.perf_counter() - tool_t0) * 1000)

                trace = ToolCallTrace(
                    step=step,
                    tool_name=fn_name,
                    arguments=exec_res.parsed_args,
                    result_summary=exec_res.summary,
                    success=exec_res.success,
                    latency_ms=tool_ms,
                )
                tool_traces.append(trace)

                # 收集去重引用片段
                citations_changed = False
                for cit in exec_res.citations:
                    key = str(cit.get("chunk_id") or f"{cit.get('source')}|{cit.get('section')}|{cit.get('text', '')[:40]}")
                    if key not in citations_map:
                        citations_map[key] = CitationItem(
                            source=str(cit.get("source", "")),
                            section=str(cit.get("section", "")),
                            text=str(cit.get("text", "")),
                            score=float(cit.get("score", 0.0)),
                            confidence=str(cit.get("confidence", "medium")),
                            chunk_id=str(cit.get("chunk_id", "")),
                        )
                        citations_changed = True

                if exec_res.handoff_triggered:
                    handoff_triggered = True
                if exec_res.created_ticket_id and exec_res.created_ticket_id not in created_ticket_ids:
                    created_ticket_ids.append(exec_res.created_ticket_id)

                self._emit_event(
                    event_callback,
                    {
                        "type": "tool_result",
                        "step": step,
                        "tool_name": fn_name,
                        "arguments": exec_res.parsed_args,
                        "result_summary": exec_res.summary,
                        "summary": exec_res.summary,
                        "success": exec_res.success,
                        "latency_ms": tool_ms,
                    },
                )

                if citations_changed:
                    self._emit_event(
                        event_callback,
                        {
                            "type": "citations",
                            "citations": [c.model_dump() for c in citations_map.values()],
                        },
                    )

                # 按 OpenAI Tool Protocol 将工具结果追加回上下文
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": exec_res.result_json,
                    }
                )

        # 若达到 max_steps 仍未返回纯文本，执行一次不带工具的强制收尾总结，防止死循环或空答复
        if not final_answer:
            logger.warning("Agent 达到最大步数上限 max_steps=%d，触发收尾总结调用。", step_limit)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "请停止继续调用工具，仅基于以上已经获取到的真实工具查询结果，"
                        "直接向用户给出清晰、诚实且带 [来源: 文件名] 标注的最终答复。"
                    ),
                }
            )
            try:
                if stream_answer:
                    final_msg = self._chat_completion_stream_with_retry(
                        messages,
                        use_tools=False,
                        step=steps_used + 1,
                        event_callback=event_callback,
                    )
                    final_answer = (getattr(final_msg, "content", None) or "").strip()
                else:
                    final_completion = self._chat_completion_with_retry(messages, use_tools=False)
                    final_answer = (final_completion.choices[0].message.content or "").strip()
            except Exception as exc:
                logger.error("收尾总结调用失败: %s", exc)

        if not final_answer:
            final_answer = (
                "抱歉，当前系统处理您的请求时遇到临时波动，建议您稍后重试或直接联系星云智联人工客服专员协助处理。"
            )

        total_ms = int((time.perf_counter() - start_ts) * 1000)
        response = AgentResponse(
            answer=final_answer,
            citations=list(citations_map.values()),
            tool_calls_trace=tool_traces,
            latency_ms=total_ms,
            mode=mode,
            steps_used=steps_used,
            handoff_triggered=handoff_triggered,
            created_ticket_ids=created_ticket_ids,
        )

        self._emit_event(
            event_callback,
            {
                "type": "final_answer",
                "answer": response.answer,
                "latency_ms": response.latency_ms,
                "steps_used": response.steps_used,
            },
        )
        return response


@lru_cache(maxsize=1)
def get_agent() -> SupportAgent:
    """获取进程内复用的 SupportAgent 单例。"""
    return SupportAgent()


def run_agent(
    user_message: str,
    *,
    history: Sequence[dict[str, Any]] | None = None,
    mode: AgentMode = "optimized",
    max_steps: int | None = None,
    event_callback: EventCallback | None = None,
) -> AgentResponse:
    """便捷函数：使用全局单例 Agent 执行单轮对话调度。"""
    return get_agent().run(
        user_message=user_message,
        history=history,
        mode=mode,
        max_steps=max_steps,
        event_callback=event_callback,
    )
