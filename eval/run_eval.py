"""星云智联（NebulaLink）自动化评测流水线脚本。

核心特性：
1. 支持 `--mode baseline|optimized|both`、`--limit N`、`--delay 1.0`；
2. 单线程顺序执行，内置 429 限流与网络抖动指数退避重试；
3. 每跑完 1 题立刻追加写入 `eval/cache_{mode}.jsonl` 并刷盘，支持中断后断点续跑；
4. 自动隔离临时 SQLite 数据库副本，避免 `create_after_sales_ticket` 污染正式业务库；
5. 使用 LLM-as-a-Judge 结合规则自动评估 4 项核心指标：
   - 检索命中率（Hit Rate@3）
   - 回答准确率（Accuracy）
   - 工具调用准确率（Tool Call Accuracy）
   - 拒答抗幻觉率（Refusal Rate）

用法示例：
    python -m eval.run_eval --mode both --limit 4 --delay 1.0
    python eval/run_eval.py --mode both --delay 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

# 确保无论通过 `python eval/run_eval.py` 还是 `python -m eval.run_eval` 都能正确导入 app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.core import AgentResponse, SupportAgent  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402

# 工具名别名映射：兼容数据集中简写 `create_ticket` 与实际工具名 `create_after_sales_ticket`
TOOL_NAME_ALIASES: dict[str, set[str]] = {
    "create_ticket": {"create_ticket", "create_after_sales_ticket"},
    "create_after_sales_ticket": {"create_ticket", "create_after_sales_ticket"},
    "search_knowledge_base": {"search_knowledge_base"},
    "query_order": {"query_order"},
    "handoff_human": {"handoff_human"},
}

EvalMode = Literal["baseline", "optimized"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.run_eval")


# =============================================================================
# 1. LLM-as-a-Judge 裁判器（复用 .env 的 LLM_* 配置）
# =============================================================================


class LLMJudge:
    """基于 OpenAI 兼容接口的 LLM-as-a-Judge 自动评估器。"""

    def __init__(self, settings: Settings | None = None, max_retries: int = 5) -> None:
        self.settings = settings or get_settings()
        api_key = self.settings.LLM_API_KEY.strip()
        if not api_key:
            raise ValueError("LLM_API_KEY 未配置，无法初始化 LLM-as-a-Judge。")

        self.model = self.settings.LLM_MODEL
        self.fallback_model = self.settings.LLM_FALLBACK_MODEL.strip()
        self.max_retries = max(1, max_retries)
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.settings.LLM_BASE_URL,
            timeout=60.0,
            max_retries=0,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """从模型输出中稳健提取 JSON 对象。"""
        cleaned = (text or "").strip()
        # 优先剥离 ```json ... ``` 代码块
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S)
        if fenced:
            cleaned = fenced.group(1).strip()
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        return json.loads(cleaned)

    def _call_judge_llm(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """调用裁判模型，内置 429/网络波动指数退避重试与可选备用模型降级。"""
        candidate_models = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            candidate_models.append(self.fallback_model)

        last_error: Exception | None = None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for model_idx, current_model in enumerate(candidate_models):
            is_last = model_idx + 1 >= len(candidate_models)
            attempts = self.max_retries if is_last else max(4, self.max_retries)

            create_kwargs: dict[str, Any] = {
                "model": current_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 512,
            }
            if "qwen3" in current_model.lower():
                create_kwargs["extra_body"] = {"enable_thinking": False}

            for attempt in range(1, attempts + 1):
                try:
                    resp = self._client.chat.completions.create(**create_kwargs)
                    content = (resp.choices[0].message.content or "").strip()
                    return self._extract_json(content)
                except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                    last_error = exc
                    if attempt < attempts:
                        wait_s = min(45.0, 2.0 ** attempt) + random.uniform(0.3, 1.0)
                        logger.warning(
                            "Judge LLM(%s) 请求受限/波动（第 %d/%d 次）：%s，%.1f 秒后重试...",
                            current_model,
                            attempt,
                            attempts,
                            exc,
                            wait_s,
                        )
                        time.sleep(wait_s)
                except APIStatusError as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", 0)
                    if status_code == 429 or status_code >= 500:
                        if attempt < attempts:
                            wait_s = min(45.0, 2.0 ** attempt) + random.uniform(0.3, 1.0)
                            logger.warning(
                                "Judge LLM(%s) 状态码 %s（第 %d/%d 次），%.1f 秒后重试...",
                                current_model,
                                status_code,
                                attempt,
                                attempts,
                                wait_s,
                            )
                            time.sleep(wait_s)
                    else:
                        raise
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    if attempt < attempts:
                        logger.warning(
                            "Judge LLM(%s) 输出 JSON 解析失败（第 %d/%d 次），正在重试...",
                            current_model,
                            attempt,
                            attempts,
                        )
                        time.sleep(1.0)

        raise RuntimeError(f"LLM-as-a-Judge 重试耗尽仍失败: {last_error}") from last_error

    def judge_accuracy(
        self,
        *,
        question: str,
        ground_truth: str,
        agent_answer: str,
        category: str,
    ) -> tuple[bool, str]:
        """评估可回答题（normal_kb / exact_detail / tool_call）的回答准确率。"""
        system_prompt = (
            "你是一位严格、客观的企业级智能客服评测裁判。"
            "请对比【标准答案】与【模型回答】，判断模型回答是否在核心事实上与标准答案一致。\n"
            "评判规则：\n"
            "1. 关键事实（型号、价格、数值参数、时长、百分比、政策条件、订单状态、物流单号、工单优先级等）必须准确无误；\n"
            "2. 允许表述措辞不同、补充合理礼貌用语或额外出处标注，只要不与标准答案矛盾且未遗漏核心要点即判为 true；\n"
            "3. 若出现关键数值错误、核心条件遗漏、结论相反或编造错误信息，则判为 false。\n"
            "请严格只输出一个 JSON 对象，格式为：{\"is_accurate\": true/false, \"reason\": \"不超过60字的简明理由\"}。"
        )
        user_prompt = (
            f"【题型】{category}\n"
            f"【用户问题】{question}\n"
            f"【标准答案】{ground_truth}\n"
            f"【模型回答】{agent_answer}\n\n"
            "请输出 JSON 评判结果："
        )
        result = self._call_judge_llm(system_prompt, user_prompt)
        is_accurate = bool(result.get("is_accurate", False))
        reason = str(result.get("reason", "")).strip()
        return is_accurate, reason

    def judge_refusal(
        self,
        *,
        question: str,
        ground_truth: str,
        agent_answer: str,
    ) -> tuple[bool, bool, str]:
        """评估知识库外问题（unanswerable）的拒答抗幻觉表现。

        Returns:
            `(passed, hallucinated, reason)`：只有诚实拒答且未编造时 `passed=True`。
        """
        system_prompt = (
            "你是一位严格的反幻觉评测裁判，专门评估客服模型面对【知识库未覆盖问题】时的表现。\n"
            "评判规则：\n"
            "1. is_honest_refusal（是否诚实拒答）：模型是否明确告知用户知识库/官方资料中未查到该信息、无法确认或建议联系人工/关注官方公告；\n"
            "2. hallucinated（是否产生幻觉编造）：模型是否编造了不存在的产品发布时间、售价、竞品对比结论、线下门店地址、创始人/公司规模、mAh 电池容量、兼容性或分期/以旧换新政策等；\n"
            "   注意：若模型仅客观复述知识库中真实存在的星云智联已有产品参数作为补充说明，同时明确说明目标问题“未收录/无法确认”，不算幻觉。\n"
            "请严格只输出一个 JSON 对象，格式为："
            "{\"is_honest_refusal\": true/false, \"hallucinated\": true/false, \"reason\": \"不超过60字的简明理由\"}。"
        )
        user_prompt = (
            f"【用户问题】{question}\n"
            f"【参考拒答标准】{ground_truth}\n"
            f"【模型回答】{agent_answer}\n\n"
            "请输出 JSON 评判结果："
        )
        result = self._call_judge_llm(system_prompt, user_prompt)
        is_honest = bool(result.get("is_honest_refusal", False))
        hallucinated = bool(result.get("hallucinated", True))
        reason = str(result.get("reason", "")).strip()
        passed = is_honest and (not hallucinated)
        return passed, hallucinated, reason


# =============================================================================
# 2. 规则指标评估函数
# =============================================================================


def evaluate_hit_at_3(
    expected_source: str | None,
    citations: list[dict[str, Any]],
) -> bool | None:
    """评估检索命中率（Hit Rate@3）。

    - 若 `expected_source` 为 None/空，表示该题不考核知识库检索，返回 None。
    - 否则检查前 3 条引用片段的 `source` 是否命中 `expected_source`。
    """
    if not expected_source:
        return None
    target = expected_source.strip()
    top3_sources = [str(c.get("source", "")).strip() for c in citations[:3]]
    return target in top3_sources


def evaluate_tool_call(
    expected_tool: str | None,
    tool_calls_trace: list[dict[str, Any]],
) -> bool | None:
    """评估工具调用准确率（Tool Call Accuracy）。

    - 若 `expected_tool` 为 None/空，表示该题不强制要求调用特定工具，返回 None。
    - 否则检查调用轨迹中是否存在对应工具（支持别名映射）且执行成功。
    """
    if not expected_tool:
        return None
    accepted_names = TOOL_NAME_ALIASES.get(expected_tool.strip(), {expected_tool.strip()})
    for trace in tool_calls_trace:
        t_name = str(trace.get("tool_name", "")).strip()
        if t_name in accepted_names and bool(trace.get("success", True)):
            return True
    return False


# =============================================================================
# 3. 缓存读写与断点续跑
# =============================================================================


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """读取 JSONL 缓存，返回 `{question_id: record}` 映射。"""
    records: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return records

    with cache_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = str(obj.get("id", "")).strip()
                if qid:
                    records[qid] = obj
            except json.JSONDecodeError as exc:
                logger.warning("缓存文件 %s 第 %d 行解析失败已忽略：%s", cache_path, line_no, exc)
    return records


def append_cache_record(cache_path: Path, record: dict[str, Any]) -> None:
    """每完成 1 题立即追加写入 JSONL 并强制刷盘，确保随时可断点续跑。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# =============================================================================
# 4. 单模式评测主循环
# =============================================================================


def run_single_question_with_retry(
    agent: SupportAgent,
    question: str,
    mode: EvalMode,
    max_retries: int = 3,
) -> AgentResponse:
    """外层保护性重试，防止 Agent 在偶发网络/限流抖动下直接中断整批评测。"""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return agent.run(user_message=question, mode=mode)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait_s = min(30.0, 2.0 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    "Agent(%s) 运行异常（第 %d/%d 次）：%s，%.1f 秒后重试...",
                    mode,
                    attempt,
                    max_retries,
                    exc,
                    wait_s,
                )
                time.sleep(wait_s)
    raise RuntimeError(f"Agent({mode}) 连续 {max_retries} 次执行失败: {last_exc}") from last_exc


def evaluate_mode(
    *,
    mode: EvalMode,
    dataset: list[dict[str, Any]],
    cache_path: Path,
    delay: float,
    settings: Settings,
    retriever: HybridRetriever,
    judge: LLMJudge,
    no_resume: bool = False,
) -> list[dict[str, Any]]:
    """对指定模式（baseline / optimized）执行顺序评测并实时缓存。"""
    if no_resume and cache_path.exists():
        logger.info("[%s] 已启用 --no-resume，清空旧缓存：%s", mode, cache_path)
        cache_path.unlink()

    cached = load_cache(cache_path)
    total = len(dataset)
    logger.info(
        "========== 开始评测模式：%s | 题目总数：%d | 已命中缓存：%d ==========",
        mode,
        total,
        sum(1 for item in dataset if item["id"] in cached),
    )

    # 复制一份独立的 SQLite 临时数据库，防止 create_after_sales_ticket 污染正式业务库
    src_db = settings.sqlite_db_file
    if not src_db.exists():
        raise FileNotFoundError(f"业务数据库不存在：{src_db}，请先初始化 data/business.db。")

    with tempfile.TemporaryDirectory(prefix=f"eval_db_{mode}_") as tmp_dir:
        tmp_db_path = Path(tmp_dir) / "business_eval.db"
        shutil.copy2(src_db, tmp_db_path)

        agent = SupportAgent(
            settings=settings,
            retriever=retriever,
            db_path=tmp_db_path,
        )

        results: list[dict[str, Any]] = []
        executed_count = 0

        for idx, item in enumerate(dataset, start=1):
            qid = item["id"]
            category = item["category"]
            question = item["question"]
            ground_truth = item["ground_truth"]
            expected_source = item.get("expected_source")
            expected_tool = item.get("expected_tool")

            if qid in cached:
                logger.info("[%s] (%d/%d) %s 已在缓存中，跳过。", mode, idx, total, qid)
                results.append(cached[qid])
                continue

            logger.info("[%s] (%d/%d) 正在评测 %s [%s]...", mode, idx, total, qid, category)
            t0 = time.perf_counter()
            resp = run_single_question_with_retry(agent=agent, question=question, mode=mode)
            wall_ms = int((time.perf_counter() - t0) * 1000)

            citations_dump = [c.model_dump() for c in resp.citations]
            traces_dump = [t.model_dump() for t in resp.tool_calls_trace]
            tools_called = [t["tool_name"] for t in traces_dump]

            # 1) Hit Rate@3（规则）
            hit_at_3 = evaluate_hit_at_3(expected_source, citations_dump)

            # 2) Tool Call Accuracy（规则）
            tool_call_correct = evaluate_tool_call(expected_tool, traces_dump)

            # 3) Accuracy / 4) Refusal Rate（LLM-as-a-Judge）
            is_accurate: bool | None = None
            refusal_passed: bool | None = None
            hallucinated: bool | None = None
            judge_reason = ""

            if category == "unanswerable":
                refusal_passed, hallucinated, judge_reason = judge.judge_refusal(
                    question=question,
                    ground_truth=ground_truth,
                    agent_answer=resp.answer,
                )
            else:
                is_accurate, judge_reason = judge.judge_accuracy(
                    question=question,
                    ground_truth=ground_truth,
                    agent_answer=resp.answer,
                    category=category,
                )

            record: dict[str, Any] = {
                "id": qid,
                "mode": mode,
                "category": category,
                "question": question,
                "ground_truth": ground_truth,
                "expected_source": expected_source,
                "expected_tool": expected_tool,
                "agent_answer": resp.answer,
                "citations": citations_dump,
                "top3_sources": [c.get("source", "") for c in citations_dump[:3]],
                "tool_calls_trace": traces_dump,
                "tools_called": tools_called,
                "steps_used": resp.steps_used,
                "handoff_triggered": resp.handoff_triggered,
                "created_ticket_ids": resp.created_ticket_ids,
                "latency_ms": resp.latency_ms,
                "wall_ms": wall_ms,
                "metrics": {
                    "hit_at_3": hit_at_3,
                    "is_accurate": is_accurate,
                    "tool_call_correct": tool_call_correct,
                    "refusal_passed": refusal_passed,
                    "hallucinated": hallucinated,
                },
                "judge_reason": judge_reason,
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            }

            append_cache_record(cache_path, record)
            cached[qid] = record
            results.append(record)
            executed_count += 1

            status_tag = (
                f"Refusal={'PASS' if refusal_passed else 'FAIL'}"
                if category == "unanswerable"
                else f"Acc={'PASS' if is_accurate else 'FAIL'}"
            )
            logger.info(
                "[%s] %s 完成 | %s | Hit@3=%s | Tool=%s | 耗时=%dms | 工具=%s",
                mode,
                qid,
                status_tag,
                hit_at_3,
                tool_call_correct,
                resp.latency_ms,
                tools_called,
            )

            # 单线程顺序防限流间隔
            if delay > 0 and idx < total:
                time.sleep(delay)

        logger.info(
            "========== 模式 %s 完成：本次新跑 %d 题，累计 %d/%d 题，缓存：%s ==========",
            mode,
            executed_count,
            len(results),
            total,
            cache_path,
        )
        return results


# =============================================================================
# 5. CLI 入口
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="星云智联智能客服自动化评测流水线（支持断点续跑与 LLM-as-a-Judge）"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized", "both"],
        default="both",
        help="评测模式：baseline、optimized 或 both（默认 both）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅评测数据集前 N 题（例如 --limit 4 用于小规模试跑；不传则跑全量）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="每道题评测完后的等待秒数，防 API 并发限流（默认 1.0 秒）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "eval_dataset.json"),
        help="评测数据集 JSON 文件路径",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval"),
        help="评测缓存 JSONL 输出目录（默认 eval/）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有缓存并清空重跑（默认自动断点续跑）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测数据集不存在：{dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须为正整数")
        dataset = dataset[: args.limit]

    settings = get_settings()
    logger.info(
        "加载配置完成 | LLM=%s | Embedding=%s | Rerank=%s | 评测题数=%d | delay=%.2fs",
        settings.LLM_MODEL,
        settings.EMBEDDING_MODEL,
        settings.RERANK_ENABLED,
        len(dataset),
        args.delay,
    )

    retriever = HybridRetriever(settings=settings)
    judge = LLMJudge(settings=settings)
    cache_dir = Path(args.cache_dir)

    modes_to_run: list[EvalMode] = (
        ["baseline", "optimized"] if args.mode == "both" else [args.mode]
    )

    for mode in modes_to_run:
        cache_path = cache_dir / f"cache_{mode}.jsonl"
        evaluate_mode(
            mode=mode,
            dataset=dataset,
            cache_path=cache_path,
            delay=max(0.0, args.delay),
            settings=settings,
            retriever=retriever,
            judge=judge,
            no_resume=args.no_resume,
        )

    logger.info("评测流水线全部执行完毕。可运行 `python eval/report_generator.py` 生成对比报表。")


if __name__ == "__main__":
    main()
