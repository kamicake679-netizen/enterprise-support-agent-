"""模拟业务数据库初始化与数据访问模块（SQLite）。

包含 `orders`（订单表）与 `tickets`（售后工单表），
支持幂等初始化 20 条覆盖全生命周期状态的星云智联（NebulaLink）模拟订单数据。
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# 兼容直接执行 `python app/db/mock_db.py` 时的模块导入路径
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    """解析 SQLite 数据库文件路径并确保父目录存在。"""
    if db_path is not None:
        path = Path(db_path)
    else:
        try:
            from app.config import get_settings

            path = get_settings().sqlite_db_file
        except ImportError:
            raw = os.getenv("SQLITE_DB_PATH", "data/business.db")
            p = Path(raw)
            path = p if p.is_absolute() else _ROOT_DIR / p
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """获取 SQLite 数据库连接（启用 Row 工厂与外键约束）。"""
    path = _resolve_db_path(db_path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    """创建 orders 与 tickets 数据表。"""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_name TEXT NOT NULL,
            user_phone TEXT NOT NULL,
            product_model TEXT NOT NULL,
            product_color TEXT NOT NULL,
            amount_paid REAL NOT NULL,
            order_status TEXT NOT NULL,
            lifecycle_tag TEXT NOT NULL,
            is_activated INTEGER NOT NULL DEFAULT 0,
            has_nebulacare INTEGER NOT NULL DEFAULT 0,
            promo_tag TEXT NOT NULL DEFAULT '常规订单',
            order_time TEXT NOT NULL,
            delivered_time TEXT,
            tracking_number TEXT
        );

        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'P2',
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '待处理',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
        """
    )


def _build_seed_orders(now: datetime) -> list[tuple[Any, ...]]:
    """构造 20 条覆盖不同业务状态的模拟订单数据（时间基于当前时间动态偏移）。"""

    def fmt(dt: datetime | None) -> str | None:
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None

    # 格式：(order_id, user_name, user_phone, product_model, product_color,
    #        amount_paid, order_status, lifecycle_tag, is_activated, has_nebulacare,
    #        promo_tag, order_time, delivered_time, tracking_number)
    return [
        # --- 1. 待发货（3 条） ---
        (
            "NL20261122001", "陈默", "138****1001", "NL-Pro2026", "星耀黑",
            1299.0, "待发货", "待发货", 0, 1, "双11返场",
            fmt(now - timedelta(hours=3)), None, None,
        ),
        (
            "NL20261122002", "林悦", "139****2002", "NL-Air3", "云雾白",
            599.0, "待发货", "待发货", 0, 0, "常规订单",
            fmt(now - timedelta(hours=8)), None, None,
        ),
        (
            "NL20261121003", "张浩然", "136****3003", "NL-Studio", "钛空银",
            1899.0, "待发货", "待发货", 0, 1, "常规订单",
            fmt(now - timedelta(days=1)), None, None,
        ),

        # --- 2. 已发货 / 运输中（2 条） ---
        (
            "NL20261120004", "周子涵", "137****4004", "NL-Pro2026", "冰川银",
            1199.0, "已发货", "运输中", 0, 0, "双11_30天价保",
            fmt(now - timedelta(days=2)), None, "SF142098765001",
        ),
        (
            "NL20261120005", "吴佳琪", "158****5005", "NL-Air3", "樱花粉",
            549.0, "已发货", "运输中", 0, 0, "双11_30天价保",
            fmt(now - timedelta(days=2, hours=6)), None, "JD009812345002",
        ),

        # --- 3. 签收 3 天内（4 条：覆盖未激活全额退与已激活扣15%场景） ---
        (
            "NL20261118006", "李明哲", "135****6006", "NL-Pro2026", "星耀黑",
            1299.0, "已签收", "签收3天内", 1, 0, "常规订单",
            fmt(now - timedelta(days=4)), fmt(now - timedelta(days=1)), "SF142098765003",
        ),
        (
            "NL20261118007", "王语嫣", "186****7007", "NL-Air3", "云雾白",
            599.0, "已签收", "签收3天内", 0, 0, "常规订单",
            fmt(now - timedelta(days=4)), fmt(now - timedelta(days=2)), "SF142098765004",
        ),
        (
            "NL20261117008", "赵启明", "188****8008", "NL-Studio", "深海蓝",
            1899.0, "已签收", "签收3天内", 1, 1, "常规订单",
            fmt(now - timedelta(days=5)), fmt(now - timedelta(days=2, hours=12)), "JD009812345005",
        ),
        (
            "NL20261117009", "孙思思", "133****9009", "NL-Pro2026", "极光紫",
            1149.0, "已签收", "签收3天内", 0, 0, "双11_30天价保",
            fmt(now - timedelta(days=5)), fmt(now - timedelta(days=3)), "SF142098765006",
        ),

        # --- 4. 签收 8-14 天（3 条：过7天无理由，处于15天质量换货期/30天价保期） ---
        (
            "NL20261108010", "刘洋", "131****1010", "NL-Pro2026", "星耀黑",
            1299.0, "已签收", "签收8-14天", 1, 0, "双11_30天价保",
            fmt(now - timedelta(days=12)), fmt(now - timedelta(days=10)), "SF142098765007",
        ),
        (
            "NL20261107011", "陈晓彤", "150****1111", "NL-Air3", "薄荷绿",
            599.0, "已签收", "签收8-14天", 1, 1, "常规订单",
            fmt(now - timedelta(days=14)), fmt(now - timedelta(days=11)), "JD009812345008",
        ),
        (
            "NL20261105012", "黄文博", "159****1212", "NL-Studio", "钛空银",
            1799.0, "已签收", "签收8-14天", 1, 0, "双11_30天价保",
            fmt(now - timedelta(days=16)), fmt(now - timedelta(days=13)), "SF142098765009",
        ),

        # --- 5. 签收超过 15 天（质保期内，4 条） ---
        (
            "NL20260915013", "杨帆", "189****1313", "NL-Pro2026", "冰川银",
            1299.0, "已签收", "签收超过15天", 1, 1, "常规订单",
            fmt(now - timedelta(days=68)), fmt(now - timedelta(days=65)), "SF142098765010",
        ),
        (
            "NL20260801014", "徐若琳", "130****1414", "NL-Air3", "云雾白",
            599.0, "已签收", "签收超过15天", 1, 0, "常规订单",
            fmt(now - timedelta(days=112)), fmt(now - timedelta(days=110)), "JD009812345011",
        ),
        (
            "NL20260618015", "马俊杰", "182****1515", "NL-Studio", "星耀黑",
            1699.0, "已签收", "签收超过15天", 1, 0, "618_30天价保",
            fmt(now - timedelta(days=157)), fmt(now - timedelta(days=154)), "SF142098765012",
        ),
        (
            "NL20260310016", "朱敏", "134****1616", "NL-Pro2026", "星耀黑",
            1299.0, "已签收", "签收超过15天", 1, 0, "常规订单",
            fmt(now - timedelta(days=257)), fmt(now - timedelta(days=254)), "SF142098765013",
        ),

        # --- 6. 已过 1 年质保期（2 条：1 条无延保已过保，1 条有 NebulaCare+ 仍在2年延保内） ---
        (
            "NL20250901017", "郭志强", "151****1717", "NL-Air3", "星耀黑",
            599.0, "已签收", "已过1年质保期", 1, 0, "常规订单",
            fmt(now - timedelta(days=447)), fmt(now - timedelta(days=444)), "JD009812345014",
        ),
        (
            "NL20250815018", "何诗琪", "185****1818", "NL-Pro2026", "冰川银",
            1299.0, "已签收", "已过1年质保期", 1, 1, "常规订单",
            fmt(now - timedelta(days=464)), fmt(now - timedelta(days=460)), "SF142098765015",
        ),

        # --- 7. 售后处理中 / 已退款（2 条） ---
        (
            "NL20261101019", "高天宇", "132****1919", "NL-Pro2026", "星耀黑",
            1299.0, "售后处理中", "售后处理中", 1, 1, "常规订单",
            fmt(now - timedelta(days=21)), fmt(now - timedelta(days=18)), "SF142098765016",
        ),
        (
            "NL20261110020", "罗小雨", "156****2020", "NL-Air3", "樱花粉",
            599.0, "已退款", "售后处理中", 0, 0, "常规订单",
            fmt(now - timedelta(days=12)), fmt(now - timedelta(days=9)), "JD009812345017",
        ),
    ]


def _build_seed_tickets(now: datetime) -> list[tuple[Any, ...]]:
    """构造 3 条初始售后工单记录。"""

    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    return [
        (
            "TK20261119001",
            "NL20261101019",
            "quality_issue",
            "P2",
            "用户反馈 NL-Pro2026 左耳降噪偶发电流啸叫，已按排查手册重置无效，申请寄修检测。",
            "处理中",
            fmt(now - timedelta(days=3)),
            fmt(now - timedelta(days=1)),
        ),
        (
            "TK20261114002",
            "NL20261110020",
            "return_refund",
            "P2",
            "用户未拆封申请七天无理由退货，售后已签收质检完好并完成全额退款。",
            "已完结",
            fmt(now - timedelta(days=8)),
            fmt(now - timedelta(days=5)),
        ),
        (
            "TK20261121003",
            "NL20261108010",
            "price_protection",
            "P2",
            "用户咨询双11订单降价 150 元一键价保差价核对。",
            "待处理",
            fmt(now - timedelta(days=1)),
            fmt(now - timedelta(days=1)),
        ),
    ]


def init_db(reset: bool = False, db_path: str | Path | None = None) -> Path:
    """初始化数据库表结构并写入模拟数据（幂等）。"""
    resolved_path = _resolve_db_path(db_path)
    with get_connection(resolved_path) as conn:
        _create_tables(conn)
        if reset:
            conn.execute("DELETE FROM tickets;")
            conn.execute("DELETE FROM orders;")

        existing_orders = conn.execute("SELECT COUNT(*) FROM orders;").fetchone()[0]
        if existing_orders == 0:
            now = datetime.now()
            conn.executemany(
                """
                INSERT INTO orders (
                    order_id, user_name, user_phone, product_model, product_color,
                    amount_paid, order_status, lifecycle_tag, is_activated, has_nebulacare,
                    promo_tag, order_time, delivered_time, tracking_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                _build_seed_orders(now),
            )

        existing_tickets = conn.execute("SELECT COUNT(*) FROM tickets;").fetchone()[0]
        if existing_tickets == 0:
            now = datetime.now()
            conn.executemany(
                """
                INSERT INTO tickets (
                    ticket_id, order_id, issue_type, priority,
                    description, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                _build_seed_tickets(now),
            )
        conn.commit()
    return resolved_path


def get_order_by_id(order_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    """按订单号精确查询单条订单详情。"""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?;",
            (order_id.strip(),),
        ).fetchone()
    return dict(row) if row else None


def list_orders(
    status: str | None = None,
    limit: int = 20,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """查询订单列表，支持按状态过滤。"""
    with get_connection(db_path) as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM orders WHERE order_status = ? ORDER BY order_time DESC LIMIT ?;",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY order_time DESC LIMIT ?;",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_ticket_by_id(ticket_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    """按工单号查询工单详情。"""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?;",
            (ticket_id.strip(),),
        ).fetchone()
    return dict(row) if row else None


def list_tickets_by_order(order_id: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """查询指定订单关联的所有售后工单。"""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM tickets WHERE order_id = ? ORDER BY created_at DESC;",
            (order_id.strip(),),
        ).fetchall()
    return [dict(r) for r in rows]


def create_ticket(
    order_id: str,
    issue_type: str,
    description: str,
    priority: str = "P2",
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """创建一条新的售后工单并返回工单详情。"""
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM tickets;").fetchone()[0]
        ticket_id = f"TK{now.strftime('%Y%m%d')}{count + 1:03d}"
        conn.execute(
            """
            INSERT INTO tickets (
                ticket_id, order_id, issue_type, priority,
                description, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (ticket_id, order_id.strip(), issue_type, priority, description, "待处理", now_str, now_str),
        )
        conn.commit()
    return get_ticket_by_id(ticket_id, db_path=db_path) or {}


if __name__ == "__main__":
    db_file = init_db(reset=True)
    orders = list_orders(limit=20)
    print(f"[OK] 数据库初始化成功: {db_file}")
    print(f"[OK] 订单总条数: {len(orders)}")

    # 按生命周期标签统计分布
    tag_counts: dict[str, int] = {}
    for o in orders:
        tag = o["lifecycle_tag"]
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print(f"[OK] 订单状态分布: {tag_counts}")

    print("\n--- 前 5 条订单记录预览 ---")
    for o in orders[:5]:
        print(
            f"{o['order_id']} | {o['user_name']} | {o['product_model']} ({o['product_color']}) | "
            f"¥{o['amount_paid']:.0f} | 状态={o['order_status']} ({o['lifecycle_tag']}) | "
            f"已激活={bool(o['is_activated'])} | 延保={bool(o['has_nebulacare'])}"
        )

    with get_connection(db_file) as _conn:
        ticket_rows = _conn.execute("SELECT * FROM tickets;").fetchall()
    print(f"\n[OK] 初始工单总条数: {len(ticket_rows)}")
    for t in ticket_rows:
        print(f"{t['ticket_id']} | 订单={t['order_id']} | {t['priority']} | {t['issue_type']} | {t['status']}")
