from __future__ import annotations

import json
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.model.db import Base, SessionLocal


# =====================================================================
# ChatSession / ChatMessage ORM 模型
#    - ChatSession：一次完整对话（user ↔ AI），按 user_id 隔离
#    - ChatMessage：会话里的单条消息；按 (session_id, id) 复合索引加速翻历史
# =====================================================================

class ChatSession(Base):
    """对话会话表。

    :param id: 会 UUID（主键）。
    :param user_id: 业务 userid（无 FK 关联 users 表，因为 user 可被物理删除）。
    :param system_prompt: 该会话创建时使用的 system prompt（审计用，目前不读）。
    :param created_at / updated_at: 服务端自动维护。
    """

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # cascade: 删会话 → 一级删除其下所有消息（FK CASCADE 也兜底）
    messages: Mapped[List["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )


class ChatMessage(Base):
    """会话中的单条消息。

    content 一律是字符串。带媒体的用户消息、含追问的 AI 消息会被序列化成
    JSON 字符串入库（type / text / images / videos / follow_up 等字段）。
    前端按需解析。"""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    tool_calls: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    __table_args__ = (
        # 复合索引：翻历史"按 session 取 id 倒序前 N 条"是热路径
        Index("ix_chat_messages_session_id_id", "session_id", "id"),
    )


# =====================================================================
# 数据访问层（每个函数自管 session）
# =====================================================================

def create_session(user_id: str, session_id: str, system_prompt: Optional[str] = None) -> None:
    """新建一个会话。

    :param user_id: 业务 userid。
    :param session_id: 调用方生成（一般是 UUID hex）。
    :param system_prompt: 可选，写入该会话的初始 system prompt。
    """
    with SessionLocal() as db:
        db.add(ChatSession(id=session_id, user_id=user_id, system_prompt=system_prompt))
        db.commit()


def add_message(
    session_id: str,
    role: str,
    content: str,
    *,
    tool_call_id: Optional[str] = None,
    tool_calls: Optional[Any] = None,
    name: Optional[str] = None,
) -> dict:
    """插入一条消息，返回入库后的完整记录（id、created_at 等）。

    :return: dict 形式的 row（含 id / session_id / role / content / ...），
             方便上游立即把 id 写回 ORM 外的数据流（如 SSE end 事件）。
    """
    with SessionLocal() as db:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
            name=name,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return {
            "id": msg.id,
            "session_id": msg.session_id,
            "role": msg.role,
            "content": msg.content,
            "tool_call_id": msg.tool_call_id,
            "tool_calls": msg.tool_calls,
            "name": msg.name,
            "created_at": msg.created_at,
        }


def list_messages_for_session(session_id: str) -> List[dict]:
    """按 id 升序列出某会话的全部消息（agent state 重建时使用）。"""
    with SessionLocal() as db:
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
            .all()
        )
        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "tool_call_id": r.tool_call_id,
                "tool_calls": r.tool_calls,
                "name": r.name,
                "created_at": r.created_at,
            }
            for r in rows
        ]


def get_session(session_id: str, user_id: str) -> Optional[dict]:
    """按 (session_id, user_id) 查会话，不属于该用户或不存在返回 None。

    设计要点：用 (session_id, user_id) 复合过滤，**防止跨用户越权读会话**。
    不要单独按 session_id 查再让上层校验归属——容易漏。
    """
    with SessionLocal() as db:
        row = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .one_or_none()
        )
        if row is None:
            return None
        return {
            "id": row.id,
            "user_id": row.user_id,
            "system_prompt": row.system_prompt,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }


def count_messages(session_id: str) -> int:
    """统计某会话的消息总数（前端展示用，目前路由层没暴露）。"""
    with SessionLocal() as db:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .count()
        )


def list_sessions_for_user(user_id: str) -> List[dict]:
    """按 updated_at 倒序列出某用户的全部会话（侧边栏多会话用）。"""
    with SessionLocal() as db:
        rows = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )
        return [
            {
                "id": row.id,
                "user_id": row.user_id,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]