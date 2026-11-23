"""环境与模型 API 连通性自检脚本。

功能：
1. 检查 `.env` 文件与基础配置是否加载正常；
2. 测试 `LLM_*` 对话模型接口是否连通（含普通对话 + Tool/Function Calling 测试）；
3. 测试 `EMBEDDING_*` 向量模型接口是否连通并输出向量维度；
4. 输出 `RERANK_ENABLED` 状态及本地数据库/文档目录检查结果。

使用方式：
    python scripts/check_env.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from dotenv import load_dotenv
from openai import OpenAI

from app.config import get_settings


def _is_placeholder_key(key: str) -> bool:
    """判断 API Key 是否为空或仍为模板占位符。"""
    cleaned = key.strip()
    return not cleaned or cleaned in {
        "your_gemini_api_key_here",
        "your_deepseek_api_key_here",
        "你的DeepSeek_API_Key",
        "your_api_key_here",
        "sk-...",
    }


def check_llm(api_key: str, base_url: str, model: str) -> bool:
    """测试 OpenAI 兼容对话接口与 Function Calling 能力。"""
    print(f"\n[1/3] 测试对话接口 (LLM) -> model={model}, base_url={base_url}")
    if _is_placeholder_key(api_key):
        print("  [FAIL] LLM_API_KEY 未配置或仍为占位符，请先在 .env 中填入真实 Key。")
        return False

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 1) 基础对话测试
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是星云智联客服助手，请用一句话简短回复。"},
                {"role": "user", "content": "你好，请回复'连接正常'。"},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        reply = (resp.choices[0].message.content or "").strip()
        print(f"  [OK] 基础对话响应: {reply}")
    except Exception as exc:
        print(f"  [FAIL] 基础对话请求失败: {exc}")
        return False

    # 2) Function Calling / Tool Call 测试
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_order_status",
                "description": "根据订单号查询星云智联订单的发货与签收状态",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "订单号，例如 NL20261118006",
                        }
                    },
                    "required": ["order_id"],
                },
            },
        }
    ]
    try:
        tool_resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "帮我查一下订单 NL20261118006 的状态。",
                }
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=256,
        )
        msg = tool_resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            print(f"  [WARN] 模型未触发 tool_calls，返回文本: {msg.content}")
            return False

        first_call = tool_calls[0]
        fn_name = first_call.function.name
        fn_args = json.loads(first_call.function.arguments or "{}")
        print(f"  [OK] Function Calling 触发成功: {fn_name}({fn_args})")
        return True
    except Exception as exc:
        print(f"  [FAIL] Function Calling 测试失败: {exc}")
        return False


def check_embedding(api_key: str, base_url: str, model: str) -> bool:
    """测试 OpenAI 兼容向量接口。"""
    print(f"\n[2/3] 测试向量接口 (Embedding) -> model={model}, base_url={base_url}")
    if _is_placeholder_key(api_key):
        print("  [FAIL] EMBEDDING_API_KEY 未配置或仍为占位符，请先在 .env 中填入真实 Key。")
        return False

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.embeddings.create(
            model=model,
            input="星云智联 NL-Pro2026 降噪深度 -52dB，支持七天无理由退货。",
        )
        vec = resp.data[0].embedding
        print(f"  [OK] Embedding 请求成功，向量维度: {len(vec)}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Embedding 请求失败: {exc}")
        return False


def check_local_env() -> None:
    """检查 Rerank 开关及本地数据目录。"""
    settings = get_settings()
    print("\n[3/3] 检查本地配置与数据底座")
    if settings.RERANK_ENABLED:
        print(f"  [INFO] RERANK_ENABLED=true -> model={settings.RERANK_MODEL}")
    else:
        print("  [OK] RERANK_ENABLED=false -> 将使用本地免费 RRF 融合算法对向量+BM25结果精排")

    docs = list(settings.docs_dir_path.glob("*.md")) if settings.docs_dir_path.exists() else []
    print(f"  [OK] 知识库文档目录: {settings.docs_dir_path} (共 {len(docs)} 份 Markdown 文档)")
    db_exists = settings.sqlite_db_file.exists()
    db_flag = "已就绪" if db_exists else "尚未生成（可运行 python app/db/mock_db.py 初始化）"
    print(f"  [OK] SQLite 业务库: {settings.sqlite_db_file} ({db_flag})")


def main() -> int:
    env_file = _ROOT_DIR / ".env"
    if not env_file.exists():
        print("[WARN] 未检测到 .env 文件！")
        print("请先执行: copy .env.example .env（或 cp .env.example .env），并填入您的 API Key。")
        return 1

    load_dotenv(env_file, override=True)
    settings = get_settings()

    llm_ok = check_llm(settings.LLM_API_KEY, settings.LLM_BASE_URL, settings.LLM_MODEL)
    emb_ok = check_embedding(
        settings.EMBEDDING_API_KEY,
        settings.EMBEDDING_BASE_URL,
        settings.EMBEDDING_MODEL,
    )
    check_local_env()

    print("\n" + "=" * 50)
    if llm_ok and emb_ok:
        print("全部核心 API 自检通过！可以进入后续 RAG 与 Agent 开发。")
        return 0
    print("部分自检未通过，请根据上方提示检查 .env 配置后重试。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
