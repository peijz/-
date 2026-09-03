from __future__ import annotations

import logging
import os
import re
import secrets
import time
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Optional

import jwt
import redis
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError

from app.model.db import SessionLocal
from app.model.users import User

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("app.api.auth")


# =====================================================================
# 配置 & 运行时资源
# =====================================================================

APP_ENV = os.getenv("APP_ENV", "production").lower()


def _load_jwt_secret() -> str:
    """加载并校验 JWT_SECRET。生产环境必须显式配置，否则进程启动即 fail-fast。

    为什么必须 fail-fast：
      - 进程内随机 secret → 每次重启换一份 → 所有已签发 token 集体失效。
      - 多 worker 进程（如 APP_WORKERS=2）若各自生成随机 secret，worker 之间
        互不信任 token，会出现"偶发 401"的诡异现象。

    生成安全 secret：
        openssl rand -hex 32     # 64 字符 hex = 256 bit 熵
    """
    secret = os.getenv("JWT_SECRET", "").strip()

    if APP_ENV == "production":
        if not secret:
            raise RuntimeError(
                "JWT_SECRET 未配置。生产环境必须显式设置 JWT_SECRET "
                "(可用 `openssl rand -hex 32` 生成)，否则重启后所有 token 失效。"
            )
        if len(secret) < 32:
            raise RuntimeError(
                f"JWT_SECRET 长度仅 {len(secret)} 字符，生产环境要求 ≥ 32。"
                f"建议用 `openssl rand -hex 32` 重新生成。"
            )

    if not secret:
        # 仅 dev 环境 fall back：进程内随机 secret + 显式告警，避免静默坑
        secret = secrets.token_hex(32)
        logger.warning(
            "[auth] JWT_SECRET 未配置（APP_ENV=%s），使用进程内随机 secret。"
            "重启或新 worker 启动会导致所有已签发 token 失效，仅供开发使用。",
            APP_ENV,
        )
    return secret


JWT_SECRET = _load_jwt_secret()
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_SEC = 1 * 24 * 60 * 60     # 1 天
REFRESH_TOKEN_TTL_SEC = 30 * 24 * 60 * 60   # 30 天
RATE_LIMIT_WINDOW_SEC = 5 * 60              # 5 分钟窗口
RATE_LIMIT_MAX_FAILS = 5                    # 窗口内连续失败 5 次封禁

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis = redis.from_url(REDIS_URL, decode_responses=True)

# OWASP 推荐默认参数：time_cost=3, memory_cost=64MB, parallelism=4
_ph = PasswordHasher()

# 用于"用户不存在"场景的耗时对齐：算一个合法的 argon2 哈希但永不匹配，
# 避免因用户枚举造成响应时间差异（OWASP A07:2021）。
_DUMMY_HASH = _ph.hash(secrets.token_hex(16))

# 手机号 / userid 正则。手机号允许 + 国际区号前缀，长度 6~20 位。
PHONE_RE = re.compile(r"^\+?\d{6,20}$")
USERID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

# FastAPI 内置 Bearer 鉴权依赖：缺失 Authorization 头会自动 401
_bearer = HTTPBearer(auto_error=True)


# =====================================================================
# 当前用户依赖（每个受保护路由都 Depends 这个）
# =====================================================================

def get_current_userid(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """从 ``Authorization: Bearer <token>`` 解析并返回当前 userid。

    校验链：
      1) JWT 签名 + 过期时间（pyjwt 自动校验）。
      2) ``type`` 必须是 "access"（防止 refresh_token 被当成 access 用）。
      3) ``sub`` 必须存在（userid 写在 sub 里）。
    :raises HTTPException: 401 + 对应中文描述。
    """
    try:
        claims: dict[str, Any] = jwt.decode(
            creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token 已过期")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="无效的 token")

    if claims.get("type") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token 类型错误")

    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token 缺少 sub")
    return sub


# =====================================================================
# Pydantic Schemas（请求 / 响应）
# =====================================================================

class RegisterIn(BaseModel):
    """注册请求体。

    :param phone: 6~20 位手机号（可带国际区号）。
    :param password: 8~128 位密码（明文，后端 argon2 哈希后入库）。
    :param name: 昵称，可选。
    :param role: 角色，可选（默认 "user"）。
    """
    phone: Annotated[str, Field(min_length=6, max_length=20)]
    password: Annotated[str, Field(min_length=8, max_length=128)]
    name: Optional[str] = Field(default=None, max_length=64)
    role: Optional[str] = Field(default=None, max_length=32)

    @field_validator("phone")
    @classmethod
    def _phone_format(cls, v: str) -> str:
        if not PHONE_RE.match(v):
            raise ValueError("phone format invalid")
        return v

    @field_validator("role")
    @classmethod
    def _role_default(cls, v: Optional[str]) -> str:
        return v or "user"


class LoginIn(BaseModel):
    """登录请求体：账号可以是手机号也可以是 userid。

    :param account: 手机号（纯数字）或 userid（字母数字下划线点横线）。
    :param password: 明文密码。
    """
    account: Annotated[str, Field(min_length=3, max_length=64)]
    password: Annotated[str, Field(min_length=1, max_length=128)]

    @field_validator("account")
    @classmethod
    def _account_format(cls, v: str) -> str:
        # 纯数字 6-20 位走 phone 分支；否则必须符合 userid 正则
        if PHONE_RE.match(v):
            return v
        if USERID_RE.match(v) and len(v) >= 3:
            return v
        raise ValueError("account must be a phone or a userid")


class RefreshIn(BaseModel):
    """刷新 token 请求体。"""
    refresh_token: str


class UserOut(BaseModel):
    """对外暴露的用户信息（不含 password 哈希）。"""
    id: int
    userid: str
    name: Optional[str]
    phone: str
    role: str
    creatdate: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    """登录 / 刷新 token 响应。

    :param access_token: 用于业务接口鉴权的短效 token。
    :param refresh_token: 用于换新 access_token 的长效 token。
    :param token_type: 固定 "bearer"。
    :param expires_in: access_token 剩余秒数。
    """
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class AuthOut(UserOut):
    """注册 / 登录一次性返回：用户信息 + token 对。"""
    access_token: str
    refresh_token: str
    expires_in: int


# =====================================================================
# 内部工具（限流 / 查用户 / 签 token / 拼响应）
# =====================================================================

def _is_rate_limited(userid: str) -> bool:
    """只读检查：当前累计失败次数是否已达上限。

    Redis 不可用时降级放行，避免 Redis 单点故障阻塞登录（牺牲安全性换可用性）。
    """
    try:
        n = int(_redis.get(f"rl:login:{userid}") or 0)
        return n >= RATE_LIMIT_MAX_FAILS
    except redis.RedisError:
        # Redis 不可用时降级放行，避免单点故障阻塞登录
        return False


def _record_login_failure(userid: str) -> None:
    """登录失败次数 +1，首次写入（n==1）时设 5 分钟 TTL。"""
    try:
        key = f"rl:login:{userid}"
        n = _redis.incr(key)
        if n == 1:
            _redis.expire(key, RATE_LIMIT_WINDOW_SEC)
    except redis.RedisError:
        pass


def _reset_rate_limit(userid: str) -> None:
    """登录成功时清空限流计数。"""
    try:
        _redis.delete(f"rl:login:{userid}")
    except redis.RedisError:
        pass


def _find_user(account: str) -> Optional[User]:
    """按 account 形态（手机号 or userid）查找 User。返回 ORM 对象或 None。"""
    with SessionLocal() as db:
        if PHONE_RE.match(account):
            return db.query(User).filter(User.phone == account).first()
        return db.query(User).filter(User.userid == account).first()


def _issue_tokens(user: User) -> TokenOut:
    """签发 access + refresh token 对。

    claims 中：
      - sub: 业务 userid（所有路由都靠这个识别当前用户）
      - role: 角色（预留 RBAC，当前未使用）
      - iat: 签发时间
      - jti: JWT 唯一 ID（未来 logout 黑名单用）
      - type: "access" / "refresh"，强类型防混用
    """
    now = int(time.time())
    base = {"sub": user.userid, "role": user.role, "iat": now, "jti": uuid.uuid4().hex}
    access = jwt.encode(
        {**base, "type": "access", "exp": now + ACCESS_TOKEN_TTL_SEC},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    refresh = jwt.encode(
        {**base, "type": "refresh", "exp": now + REFRESH_TOKEN_TTL_SEC},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    return TokenOut(access_token=access, refresh_token=refresh, expires_in=ACCESS_TOKEN_TTL_SEC)


def _to_auth_out(user: User, tokens: TokenOut) -> AuthOut:
    """User + TokenOut → AuthOut（注册 / 登录的最终响应结构）。"""
    return AuthOut(
        id=user.id,
        userid=user.userid,
        name=user.name,
        phone=user.phone,
        role=user.role,
        creatdate=user.creatdate,
        updated_at=user.updated_at,
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_in=tokens.expires_in,
    )


# =====================================================================
# 业务逻辑（注册 / 登录 / 刷新）
# =====================================================================

def _register(payload: RegisterIn) -> User:
    """注册一个新用户，userid 由服务端用 UUIDv4 生成。

    :param payload: 注册请求体。
    :return: 已落库的 User ORM 对象（含自增 id / userid 等）。
    :raises HTTPException: 409 phone 已被注册；或 UUID 5 次重试都撞唯一索引（极小概率）。
    """
    with SessionLocal() as db:
        if db.query(User).filter(User.phone == payload.phone).first():
            raise HTTPException(status.HTTP_409_CONFLICT, detail="phone 已注册")

        # userid 服务端生成。UUIDv4 碰撞概率极低，但万一命中 DB 唯一索引则重试。
        for _ in range(5):
            userid = uuid.uuid4().hex
            u = User(
                userid=userid,
                name=payload.name,
                phone=payload.phone,
                password=_ph.hash(payload.password),
                role=payload.role,
            )
            db.add(u)
            try:
                db.commit()
                db.refresh(u)
                return u
            except IntegrityError:
                db.rollback()
                continue
        raise HTTPException(status.HTTP_409_CONFLICT, detail="userid 生成冲突，请重试")


def _login(payload: LoginIn) -> tuple[User, TokenOut]:
    """登录主流程：先防枚举，再限流，最后校验密码。

    顺序很重要：
      1) 用户不存在时也要跑一遍 argon2（用 _DUMMY_HASH），让响应时间跟"存在但密码错"接近，
         防 OWASP A07 用户枚举。
      2) 限流命中时直接 429，不再校验密码（避免 argon2 计算浪费 CPU）。
      3) 密码错误则 +1 限流计数；成功则清零。
    """
    user = _find_user(payload.account)
    # 用户不存在也走一次密码校验，避免时间差枚举
    if user is None:
        try:
            _ph.verify(_DUMMY_HASH, payload.password)
        except (VerifyMismatchError, InvalidHashError, VerificationError):
            pass
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="凭证错误")

    # 先看是否已被限流（命中即不再尝试校验）
    if _is_rate_limited(user.userid):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"登录失败次数过多，请 {RATE_LIMIT_WINDOW_SEC // 60} 分钟后再试",
        )

    try:
        _ph.verify(user.password, payload.password)
    except VerifyMismatchError:
        _record_login_failure(user.userid)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="凭证错误")

    _reset_rate_limit(user.userid)
    return user, _issue_tokens(user)


def _refresh(payload: RefreshIn) -> TokenOut:
    """用 refresh_token 换新的 access + refresh 对。

    - type 校验失败 / 过期 / 用户不存在 → 401。
    - 注意：刷新后旧的 refresh_token 仍然有效（未引入黑名单），
      如要严格一次一用需要再引入 jti 黑名单（见文件末尾 TODO）。
    """
    try:
        claims: dict[str, Any] = jwt.decode(
            payload.refresh_token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="refresh_token 已过期")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="refresh_token 无效")

    if claims.get("type") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token 类型错误")

    userid = claims.get("sub")
    with SessionLocal() as db:
        user = db.query(User).filter(User.userid == userid).first()
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
        return _issue_tokens(user)


# =====================================================================
# 路由
# =====================================================================

@router.post(
    "/register",
    response_model=AuthOut,
    status_code=status.HTTP_201_CREATED,
    summary="用户注册（注册即登录，下发 token）",
)
def register(payload: RegisterIn) -> AuthOut:
    """注册入口：调用 ``_register`` 然后立即签 token。"""
    user = _register(payload)
    tokens = _issue_tokens(user)
    return _to_auth_out(user, tokens)


@router.post(
    "/login",
    response_model=AuthOut,
    summary="登录（userid 或 phone + password）",
)
def login(payload: LoginIn) -> AuthOut:
    """登录入口：返回用户信息 + 新 token 对。"""
    user, tokens = _login(payload)
    return _to_auth_out(user, tokens)


@router.post(
    "/refresh",
    response_model=TokenOut,
    summary="用 refresh_token 换新 access_token（同时下发新 refresh_token）",
)
def refresh(payload: RefreshIn) -> TokenOut:
    """刷新入口：旧 refresh 仍然有效（未引入 jti 黑名单）。"""
    return _refresh(payload)


# TODO: POST /auth/logout —— 将当前 jti 加入 Redis 黑名单，过期后清除