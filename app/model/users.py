from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.model.db import Base, SessionLocal


# =====================================================================
# User ORM 模型
#    - 业务 userid：UUID hex（由 _register 服务端生成，避免前端传入重号）
#    - phone：唯一索引，作为登录账号之一
#    - password：必须是 argon2 哈希值，绝不存明文
#    - role：默认 "user"，预留 "admin" 角色给未来扩展
# =====================================================================

class User(Base):
    """用户表 ORM 映射。

    字段说明：
      - id: 自增主键（DB 内部用，前端只认 userid）。
      - userid: 业务 ID（UUID hex），登录 / 路由 / 长期记忆隔离都靠它。
      - name: 昵称（可选，登录不需要）。
      - phone: 11 位手机号（+ 国际区号也接受），唯一。
      - password: argon2 哈希字符串（PHC 格式），永不出库给前端。
      - role: 默认 "user"；后续可扩 "admin" / "doctor" 等。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    userid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    phone: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    creatdate: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="user"
    )


# =====================================================================
# User 数据访问层
#    - 创建 / 按 userid 查 / 按 phone 查
#    - 返回 dict 而不是 ORM 对象，避免上层持有 session 句柄
# =====================================================================


def create_user(
    userid: str,
    phone: str,
    password: str,
    role: str = "user",
) -> int:
    """插入用户并返回新自增 id。

    :param userid: 业务 userid（调用方负责保证唯一）。
    :param phone: 唯一手机号。
    :param password: **已经是 argon2 哈希值**，本函数不再二次哈希。
    :param role: 角色字符串，默认 "user"。
    :return: 新插入行的自增 id。
    :raises sqlalchemy.exc.IntegrityError: userid / phone 唯一索引冲突时由 SQLAlchemy 抛出。
    """
    with SessionLocal() as db:
        u = User(userid=userid, phone=phone, password=password, role=role)
        db.add(u)
        db.commit()
        db.refresh(u)
        return u.id


def get_user_by_userid(userid: str) -> Optional[dict]:
    """按业务 userid 查用户。

    :return: 命中返回 dict（含 id / userid / name / phone / role / creatdate / updated_at，
             无 password 字段），未命中返回 None。
    """
    with SessionLocal() as db:
        u = db.query(User).filter(User.userid == userid).one_or_none()
        return _to_dict(u) if u else None


def get_user_by_phone(phone: str) -> Optional[dict]:
    """按手机号查用户；命中返回 dict，未命中返回 None。"""
    with SessionLocal() as db:
        u = db.query(User).filter(User.phone == phone).one_or_none()
        return _to_dict(u) if u else None


def _to_dict(u: User) -> dict:
    """ORM User → 业务 dict（剥掉 password 字段，避免无意中泄漏哈希值）。"""
    return {
        "id": u.id,
        "userid": u.userid,
        "name": u.name,
        "phone": u.phone,
        "role": u.role,
        "creatdate": u.creatdate,
        "updated_at": u.updated_at,
    }