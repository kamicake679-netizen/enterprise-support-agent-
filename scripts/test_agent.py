"""Step 3：Function Calling 工具层与 Agent 主循环验收测试脚本。

自动模拟并验收以下典型场景，打印完整回答、引用来源、工具调用轨迹与耗时：
1. 纯知识库条款问答（验证混合检索 + 强制 [来源: 文件名] 标注）；
2. 复合场景（查订单能否退货 + 自动创建售后工单，并校验 SQLite 工单落库）；
3. 知识库外诱导问题（验证阈值拦截 + 老实拒答不编造 + 转人工兜底）；
4. 附加验证：未提供订单号防猜测追问 + `mode="baseline"` 模式切换冒烟。

使用方式：
    python scripts/test_agent.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from app.agent.core import AgentResponse, SupportAgent
from app.agent.tools import execute_tool
from app.config import Settings, get_settings
from app.db.mock_db import get_connection, get_ticket_by_id, init_db
from app.rag.retriever import HybridRetriever


def _print_response(title: str, query: str, resp: AgentResponse) -> None:
    """格式化打印单次 Agent 响应与完整工具调用轨迹。"""
    print("\n" + "=" * 82)
    print(f"【{title}】 (mode={resp.mode} | steps={resp.steps_used} | latency={resp.latency_ms}ms)")
    print("=" * 82)
    print(f"用户提问: {query}\n")

    print("▶ 工具调用轨迹 (tool_calls_trace):")
    if not resp.tool_calls_trace:
        print("  (无工具调用)")
    else:
        for idx, t in enumerate(resp.tool_calls_trace, start=1):
            args_str = json.dumps(t.arguments, ensure_ascii=False)
            flag = "OK" if t.success else "ERR"
            print(
                f"  #{idx} [Step {t.step} | {flag} | {t.latency_ms}ms] "
                f"{t.tool_name}({args_str})"
            )
            print(f"      出参摘要: {t.result_summary}")

    print("\n▶ 命中知识片段 (citations):")
    if not resp.citations:
        print("  (无命中知识片段)")
    else:
        for idx, c in enumerate(resp.citations, start=1):
            preview = " ".join(c.text.splitlines()).strip()
            if len(preview) > 90:
                preview = preview[:90] + "..."
            print(
                f"  #{idx} [{c.source} -> {c.section}] "
                f"(score={c.score:.5f}, conf={c.confidence})"
            )
            print(f"      摘录: {preview}")

    print("\n▶ 最终回答 (answer):")
    print(resp.answer)
    print(
        f"\n▶ 状态标记: handoff_triggered={resp.handoff_triggered} | "
        f"created_ticket_ids={resp.created_ticket_ids}"
    )


def _cleanup_test_tickets(ticket_ids: list[str]) -> None:
    """清理本次测试脚本写入 SQLite 的临时工单，保持模拟数据库干净。"""
    if not ticket_ids:
        return
    with get_connection() as conn:
        for tid in ticket_ids:
            conn.execute("DELETE FROM tickets WHERE ticket_id = ?;", (tid,))
        conn.commit()


def main() -> int:
    # 确保 Windows 终端 UTF-8 输出不乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="星云智联 Agent 主循环端到端验收测试")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="可选临时覆盖 LLM_MODEL（默认使用 .env 配置，例如上游 503 高峰时可指定 gemini-3.5-flash-lite）",
    )
    args = parser.parse_args()

    settings = Settings() if args.model else get_settings()
    if args.model:
        settings = settings.model_copy(update={"LLM_MODEL": args.model.strip()})

    init_db(reset=False)
    retriever = HybridRetriever(settings=settings)
    agent = SupportAgent(settings=settings, retriever=retriever, max_steps=5, history_window=6)

    print("=" * 82)
    print("星云智联 Agent 主循环与 Function Calling 工具层验收测试")
    print("=" * 82)
    print(f"LLM 模型          : {settings.LLM_MODEL}")
    print(f"Embedding 模型    : {settings.EMBEDDING_MODEL}")
    print(f"知识库切片总数    : {retriever.total_chunks}")
    print(f"最大步数 / 窗口   : max_steps={agent.max_steps}, history_window={agent.history_window}")

    results_summary: list[tuple[str, bool, str]] = []
    created_tickets_to_cleanup: list[str] = []

    # ------------------------------------------------------------------
    # 前置校验：Pydantic 参数校验 + 历史滑动窗口（6 轮）
    # ------------------------------------------------------------------
    bad_order_res = execute_tool("query_order", {"order_id": "unknown"})
    bad_ticket_res = execute_tool(
        "create_after_sales_ticket",
        {"order_id": "NL20261118006", "issue_type": "invalid_type", "summary": "测试"},
    )
    fake_history = []
    for i in range(1, 10):
        fake_history.append({"role": "user", "content": f"第{i}轮提问"})
        fake_history.append({"role": "assistant", "content": f"第{i}轮回复"})
    trimmed = agent._trim_history(fake_history)
    guard_ok = (
        (not bad_order_res.success)
        and (not bad_ticket_res.success)
        and len(trimmed) == 12
        and trimmed[0]["content"] == "第4轮提问"
    )
    results_summary.append(
        (
            "前置校验 Pydantic与滑动窗口",
            guard_ok,
            f"拦截占位订单={not bad_order_res.success}, 拦截非法工单类型={not bad_ticket_res.success}, 9轮裁剪后消息数={len(trimmed)}",
        )
    )

    try:
        # ------------------------------------------------------------------
        # 场景 ①：纯知识库条款问答
        # ------------------------------------------------------------------
        q1 = (
            "我买的 NL-Pro2026 已经拆封并且连过手机激活使用了，没有质量问题，"
            "签收 7 天内想无理由退货需要扣多少钱？寄回运费由谁承担？"
        )
        r1 = agent.run(q1, mode="optimized")
        _print_response("场景 ①：纯知识库条款问答", q1, r1)

        called_kb_1 = any(t.tool_name == "search_knowledge_base" for t in r1.tool_calls_trace)
        has_source_tag_1 = "[来源:" in r1.answer and "02_七天无理由退换货政策.md" in r1.answer
        has_key_fact_1 = ("15%" in r1.answer) and ("用户" in r1.answer or "自行承担" in r1.answer)
        pass_1 = called_kb_1 and has_source_tag_1 and has_key_fact_1 and len(r1.citations) > 0
        results_summary.append(
            (
                "场景 ① 纯知识库条款问答",
                pass_1,
                f"调用KB={called_kb_1}, 含出处标注={has_source_tag_1}, 含15%与运费规则={has_key_fact_1}",
            )
        )

        # ------------------------------------------------------------------
        # 场景 ②：复合场景（查订单能否退货 + 创建售后工单）
        # ------------------------------------------------------------------
        q2 = (
            "帮我查一下订单 NL20261118006 能不能申请七天无理由退货，"
            "如果能退的话顺便帮我建一个退货售后工单，原因是耳机拆封试戴了一天感觉不太贴合耳朵想退货。"
        )
        r2 = agent.run(q2, mode="optimized")
        _print_response("场景 ②：复合场景（查订单 + 查政策 + 建工单）", q2, r2)

        called_order_2 = any(t.tool_name == "query_order" for t in r2.tool_calls_trace)
        called_kb_2 = any(t.tool_name == "search_knowledge_base" for t in r2.tool_calls_trace)
        called_ticket_2 = any(
            t.tool_name == "create_after_sales_ticket" and t.success
            for t in r2.tool_calls_trace
        )
        db_verified_2 = False
        if r2.created_ticket_ids:
            created_tickets_to_cleanup.extend(r2.created_ticket_ids)
            ticket_row = get_ticket_by_id(r2.created_ticket_ids[0])
            db_verified_2 = (
                ticket_row is not None and ticket_row.get("order_id") == "NL20261118006"
            )
            print(f"  [DB 校验] 工单落库确认: {ticket_row}")

        pass_2 = called_order_2 and called_kb_2 and called_ticket_2 and db_verified_2
        results_summary.append(
            (
                "场景 ② 复合场景（查单+政策+建单）",
                pass_2,
                f"查单={called_order_2}, 查政策={called_kb_2}, 建单={called_ticket_2}, DB落库={db_verified_2}",
            )
        )

        # ------------------------------------------------------------------
        # 场景 ③：知识库外诱导问题（测试老实拒答不编造 + 转人工兜底）
        # ------------------------------------------------------------------
        q3 = (
            "听说你们星云智联明天要发布新款 NL-Ultra99 脑机接口耳机，官网内部价只要 999 元"
            "还能送一年免费卫星流量卡，帮我确认一下是不是真的，顺便告诉我明天北京天气适合出门买耳机吗？"
        )
        r3 = agent.run(q3, mode="optimized")
        _print_response("场景 ③：知识库外诱导问题（防幻觉拒答）", q3, r3)

        called_kb_3 = any(t.tool_name == "search_knowledge_base" for t in r3.tool_calls_trace)
        no_fake_confirm_3 = any(
            kw in r3.answer
            for kw in ("未查到", "没有查到", "暂未", "无法为您确认", "无法确认", "未检索到", "没有相关")
        )
        handoff_or_guide_3 = r3.handoff_triggered or ("人工" in r3.answer)
        # 无论查询词因“耳机/官方”召回在售产品表用于交叉证伪，还是直接返回 0 条，核心指标是：实查KB + 诚实拒答不编造 + 转人工兜底
        pass_3 = called_kb_3 and no_fake_confirm_3 and handoff_or_guide_3
        results_summary.append(
            (
                "场景 ③ 知识库外诱导问题拒答",
                pass_3,
                f"调用KB={called_kb_3}, 引用条数={len(r3.citations)}, 明确拒答={no_fake_confirm_3}, 转人工={r3.handoff_triggered}",
            )
        )

        # ------------------------------------------------------------------
        # 附加验证 ④：未提供订单号防猜测 + baseline 模式切换冒烟
        # ------------------------------------------------------------------
        q4 = "帮我查一下我买的耳机发货了没有，顺便看看物流到哪了？"
        r4 = agent.run(q4, mode="optimized")
        _print_response("附加验证 ④-A：未提供订单号防猜测追问", q4, r4)

        not_guessed_order = not any(t.tool_name == "query_order" for t in r4.tool_calls_trace)
        asked_order_id = "订单号" in r4.answer
        pass_4a = not_guessed_order and asked_order_id
        results_summary.append(
            (
                "附加 ④-A 未给订单号追问防猜测",
                pass_4a,
                f"未乱调query_order={not_guessed_order}, 礼貌追问订单号={asked_order_id}",
            )
        )

        r_base = agent.run("NL-Pro2026 的降噪深度和官方售价是多少？", mode="baseline")
        _print_response("附加验证 ④-B：mode='baseline' 切换冒烟", "NL-Pro2026 的降噪深度和官方售价是多少？", r_base)
        pass_4b = bool(r_base.answer) and r_base.mode == "baseline"
        results_summary.append(
            (
                "附加 ④-B baseline 模式切换",
                pass_4b,
                f"mode={r_base.mode}, answer_len={len(r_base.answer)}, latency={r_base.latency_ms}ms",
            )
        )

    finally:
        # 清理本次测试创建的临时工单，避免污染模拟库
        if created_tickets_to_cleanup:
            _cleanup_test_tickets(created_tickets_to_cleanup)
            print(f"\n[清理] 已移除测试临时工单: {created_tickets_to_cleanup}")

    # 打印总验收报告
    print("\n" + "=" * 82)
    print("Step 3 验收汇总报告")
    print("=" * 82)
    passed = 0
    for name, ok, detail in results_summary:
        tag = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{tag}] {name} -> {detail}")
    print("-" * 82)
    print(f"总计: {passed}/{len(results_summary)} 项全部通过")
    print("=" * 82)
    return 0 if passed == len(results_summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
