"""RAG 混合检索与溯源模块验收脚本。

覆盖三类核心测试场景：
1. 自然语言售后问题；
2. 精确产品型号与数字条款（如 `NL-Pro2026`、`15%`、`NL-Studio` 等）；
3. 知识库中完全没有的无关问题（验证相似度/分数阈值过滤不会强行召回）。

同时对比混合检索（Hybrid RRF）与纯向量检索（Vector-Only Baseline）的召回表现。

使用方式：
    python scripts/test_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from app.config import get_settings
from app.rag.retriever import HybridRetriever


TEST_CASES: list[dict[str, object]] = [
    # --- 场景 1：自然语言售后问题 ---
    {
        "category": "① 自然语言售后问题",
        "query": "我买的耳机已经拆封连过手机了，没有质量问题，七天内想退货需要扣费吗？",
        "expect_hit": True,
        "expected_sources": ["02_七天无理由退换货政策.md"],
        "expected_keywords": ["15%", "拆封", "退货"],
    },
    {
        "category": "① 自然语言售后问题",
        "query": "耳机放进充电盒以后手机还是连着没断开，自己怎么排查解决？",
        "expect_hit": True,
        "expected_sources": ["04_常见故障自助排查手册.md"],
        "expected_keywords": ["充电盒", "触点", "酒精"],
    },
    {
        "category": "① 自然语言售后问题",
        "query": "双11买的耳机降价了怎么申请退差价，88VIP消费券算在里面吗？",
        "expect_hit": True,
        "expected_sources": ["05_大促价保与发票规则.md"],
        "expected_keywords": ["价保", "30 个自然日", "88VIP"],
    },
    # --- 场景 2：精确产品型号与数字条款 ---
    {
        "category": "② 精确产品型号与数字条款",
        "query": "NL-Pro2026 降噪深度 -52dB 防水等级和官方售价是多少？",
        "expect_hit": True,
        "expected_sources": ["01_产品型号与核心参数对比.md"],
        "expected_keywords": ["NL-Pro2026", "-52dB", "IP55"],
    },
    {
        "category": "② 精确产品型号与数字条款",
        "query": "15% 包装及卫生消杀损耗费 10% 整新费",
        "expect_hit": True,
        "expected_sources": ["02_七天无理由退换货政策.md"],
        "expected_keywords": ["15%", "10%"],
    },
    {
        "category": "② 精确产品型号与数字条款",
        "query": "NL-Pro2026 单只左耳补购价 ¥399 和 NebulaCare+ 延保价格",
        "expect_hit": True,
        "expected_sources": ["03_质保维修与延保服务条款.md"],
        "expected_keywords": ["NL-Pro2026", "399", "NebulaCare+"],
    },
    {
        "category": "② 精确产品型号与数字条款（极短型号+数字）",
        "query": "NL-Pro2026 15%",
        "expect_hit": True,
        "expected_sources": [
            "01_产品型号与核心参数对比.md",
            "02_七天无理由退换货政策.md",
        ],
        "expected_keywords": ["NL-Pro2026", "15%"],
    },
    # --- 场景 3：知识库完全没有的无关问题 ---
    {
        "category": "③ 知识库无关问题（应拦截/返回空）",
        "query": "今天北京的天气怎么样，明天出门需要带伞吗？",
        "expect_hit": False,
        "expected_sources": [],
        "expected_keywords": [],
    },
    {
        "category": "③ 知识库无关问题（应拦截/返回空）",
        "query": "请告诉我正宗红烧肉的做法步骤和所需调料。",
        "expect_hit": False,
        "expected_sources": [],
        "expected_keywords": [],
    },
    {
        "category": "③ 知识库无关问题（应拦截/返回空）",
        "query": "2026年世界杯决赛在哪个城市举办？冠军是谁？",
        "expect_hit": False,
        "expected_sources": [],
        "expected_keywords": [],
    },
]


def _preview(text: str, max_len: int = 130) -> str:
    compact = " ".join(text.splitlines()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "..."


def main() -> int:
    settings = get_settings()
    retriever = HybridRetriever(settings=settings)

    print("=" * 78)
    print("星云智联 RAG 混合检索与溯源模块验收测试")
    print("=" * 78)
    print(f"已加载切片总数      : {retriever.total_chunks}")
    print(f"Embedding 模型      : {settings.EMBEDDING_MODEL}")
    print(
        f"召回与融合配置      : Vector Top {settings.VECTOR_TOP_K} + "
        f"BM25 Top {settings.BM25_TOP_K} -> RRF(k={settings.RRF_K}) -> Top {settings.FINAL_TOP_K}"
    )
    print(f"RERANK_ENABLED      : {settings.RERANK_ENABLED}")
    print(
        f"相关性阈值配置      : vector>={settings.VECTOR_SCORE_THRESHOLD}, "
        f"bm25>={settings.BM25_SCORE_THRESHOLD} (floor={settings.VECTOR_FLOOR_FOR_BM25})"
    )

    passed = 0
    total = len(TEST_CASES)

    for idx, case in enumerate(TEST_CASES, start=1):
        category = str(case["category"])
        query = str(case["query"])
        expect_hit = bool(case["expect_hit"])
        expected_sources: list[str] = list(case["expected_sources"])  # type: ignore[arg-type]
        expected_keywords: list[str] = list(case["expected_keywords"])  # type: ignore[arg-type]

        print("\n" + "-" * 78)
        print(f"[测试 {idx}/{total}] {category}")
        print(f"Query: {query}")

        hybrid_res = retriever.search(query, top_k=3)
        baseline_res = retriever.vector_only_search(query, top_k=3)

        # 打印混合检索结果
        print("\n  ▶ 混合检索结果 (Hybrid RRF Top 3):")
        if not hybrid_res:
            # 额外查看被拦截前的最高候选分数，便于人工验收阈值效果
            raw_probe = retriever.search(query, top_k=1, return_low_confidence=True)
            if raw_probe:
                top1 = raw_probe[0]
                print(
                    "    [空结果 - 已触发相关性阈值拦截] "
                    f"(最高候选被拦: vector={top1['vector_score']}, "
                    f"bm25={top1['bm25_score']}, source={top1['source']})"
                )
            else:
                print("    [空结果]")
        else:
            for rank, item in enumerate(hybrid_res, start=1):
                print(
                    f"    #{rank} [score={item['score']:.5f} | vec={item['vector_score']:.4f} | "
                    f"bm25={item['bm25_score']:.3f} | conf={item['confidence']}]"
                )
                print(f"       出处: {item['source']} -> {item['section']}")
                print(f"       内容: {_preview(item['text'])}")

        # 打印 Baseline 摘要对比
        print("\n  ▶ 纯向量 Baseline 对比 (vector_only_search Top 3):")
        if not baseline_res:
            print("    [空结果 - 已触发向量相似度阈值拦截]")
        else:
            for rank, item in enumerate(baseline_res, start=1):
                print(
                    f"    #{rank} [vec_score={item['score']:.4f} | conf={item['confidence']}] "
                    f"{item['source']} -> {item['section']}"
                )

        # 自动判定当前用例是否通过
        if expect_hit:
            hit_sources = {item["source"] for item in hybrid_res}
            joined_text = "\n".join(item["text"] for item in hybrid_res)
            source_ok = any(src in hit_sources for src in expected_sources)
            keyword_ok = any(kw.lower() in joined_text.lower() for kw in expected_keywords)
            ok = bool(hybrid_res) and source_ok and keyword_ok
        else:
            ok = len(hybrid_res) == 0

        status_tag = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"\n  => 判定结果: [{status_tag}]")

    print("\n" + "=" * 78)
    print(f"验收汇总: {passed}/{total} 条用例通过")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
