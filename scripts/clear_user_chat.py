"""一次性脚本: 清掉指定 user 的聊天数据 + 长期记忆。

清理范围:
  - chat_messages (会话里所有消息)
  - chat_sessions  (会话本身)
  - langgraph AsyncRedisSaver 里该 user 各 session 的 checkpointer state
      (thread_id == session_id,不删的话下次同 session_id 起新会话会
       拉回旧 messages 造成"鬼影对话")
  - RediSearch 索引 agent_long_memory_v2 里该 user 的所有事实(舌象/体质等)

不动:
  - users 表 (账号保留)

用法:
    .venv/bin/python scripts/clear_user_chat.py <phone_or_userid> [--yes]

例:
    .venv/bin/python scripts/clear_user_chat.py 18911125843
    .venv/bin/python scripts/clear_user_chat.py 18911125843 --yes
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import select  # noqa: E402

from app.agent.zygchat import CHECKPOINTER, long_memory_store  # noqa: E402
from app.model.chat_log import ChatMessage, ChatSession  # noqa: E402
from app.model.db import SessionLocal  # noqa: E402
from app.model.users import User, get_user_by_phone, get_user_by_userid  # noqa: E402


def _resolve_userid(arg: str) -> tuple[str, dict]:
    """按 phone 或 userid 解析,返回 (userid, user_row)。"""
    # 先按 phone (11 位数字)
    if arg.isdigit() and len(arg) >= 7:
        u = get_user_by_phone(arg)
        if u:
            return u["userid"], u
    # 再按 userid
    u = get_user_by_userid(arg)
    if u:
        return u["userid"], u
    raise SystemExit(f"找不到用户: phone={arg} 或 userid={arg}")


async def _delete_checkpointer_state(session_ids: list[str]) -> None:
    """按 session_id (= thread_id) 清掉 checkpointer 里的历史 state。"""
    if not session_ids:
        return
    # adelete_thread 是 async; AsyncRedisSaver 在 CHECKPOINTER 里。
    if not hasattr(CHECKPOINTER, "adelete_thread"):
        print("[warn] checkpointer 不支持 adelete_thread,跳过")
        return
    for sid in session_ids:
        try:
            await CHECKPOINTER.adelete_thread(sid)
            print(f"  [checkpointer] deleted thread_id={sid}")
        except Exception as e:
            print(f"  [checkpointer] delete {sid} failed: {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="清掉指定 user 的聊天数据")
    parser.add_argument("target", help="phone 或 userid")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过交互确认")
    args = parser.parse_args()

    userid, user_row = _resolve_userid(args.target)
    print(f"目标用户: userid={userid}  phone={user_row.get('phone')}  name={user_row.get('name')}")

    # 1) 列出该 user 的会话 + 长期记忆
    with SessionLocal() as db:
        sessions = db.execute(
            select(ChatSession).where(ChatSession.user_id == userid)
        ).scalars().all()
        session_ids = [s.id for s in sessions]
        msg_count = db.execute(
            select(ChatMessage).where(ChatMessage.session_id.in_(session_ids))
        ).scalars().all()
    # 长期记忆条数(只读不删,作为 dry-run 信息)
    mem_facts = long_memory_store.similarity_search(
        query="健康",
        filter={"user_id": userid},
        k=1000,
    )

    print(f"待清理: chat_sessions={len(session_ids)} 个, chat_messages={len(msg_count)} 条, "
          f"long_memory_facts={len(mem_facts)} 条")
    for s in sessions:
        print(f"  - session {s.id}  (created={s.created_at})")
    if mem_facts:
        print("  [long_memory] 样例事实:")
        for d in mem_facts[:5]:
            print(f"    - {d['page_content']}")
        if len(mem_facts) > 5:
            print(f"    ... 还有 {len(mem_facts) - 5} 条")

    if not session_ids and not mem_facts:
        print("该用户既无聊天会话也无长期记忆,无需清理。")
        return

    if not args.yes:
        confirm = input("确认删除? 输入 yes 继续: ").strip()
        if confirm != "yes":
            print("已取消。")
            return
    else:
        print("--yes 已指定,直接执行删除。")

    # 2) 删 messages
    deleted_msgs = 0
    if session_ids:
        with SessionLocal() as db:
            deleted_msgs = db.execute(
                ChatMessage.__table__.delete().where(ChatMessage.session_id.in_(session_ids))
            ).rowcount
            db.commit()
        print(f"[mysql] deleted chat_messages: {deleted_msgs} 条")

    # 3) 删 sessions
    deleted_sess = 0
    if session_ids:
        with SessionLocal() as db:
            deleted_sess = db.execute(
                ChatSession.__table__.delete().where(ChatSession.user_id == userid)
            ).rowcount
            db.commit()
        print(f"[mysql] deleted chat_sessions: {deleted_sess} 条")

    # 4) 删 checkpointer state
    if session_ids:
        asyncio.run(_delete_checkpointer_state(session_ids))

    # 5) 删长期记忆(向量库)
    deleted_mem = long_memory_store.delete_all_for_user(userid)
    print(f"[redis] deleted long_memory_facts: {deleted_mem} 条")

    # 6) 校验
    with SessionLocal() as db:
        rest_sess = db.execute(
            select(ChatSession).where(ChatSession.user_id == userid)
        ).scalars().all()
        rest_user = db.execute(
            select(User).where(User.userid == userid)
        ).scalars().one()
        rest_mem = long_memory_store.similarity_search(
            query="健康", filter={"user_id": userid}, k=10,
        )
        print(f"[verify] chat_sessions 剩余: {len(rest_sess)}")
        print(f"[verify] long_memory_facts 剩余: {len(rest_mem)}")
        print(f"[verify] users 行保留: phone={rest_user.phone}  role={rest_user.role}")

    print("done.")


if __name__ == "__main__":
    main()
