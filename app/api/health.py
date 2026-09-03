from __future__ import annotations

import logging
import os

import redis
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.model.db import SessionLocal

logger = logging.getLogger("app.api.health")

router = APIRouter(tags=["health"])

# 单独一个 redis 客户端用于探活，跟 auth.py 的连接池隔离避免相互阻塞。
# 健康检查调用频次低（默认 <1Hz），单独一个池的开销可忽略。
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)


@router.get("/health/live", summary="进程存活探针")
def liveness() -> dict:
    """K8s livenessProbe 用：进程能响应就 OK，不查依赖。

    返回 200 即代表进程没崩 / 没死锁。
    """
    return {"ok": True}


@router.get("/health", summary="依赖就绪探针")
def readiness(response: Response) -> dict:
    """K8s readinessProbe / LB 探活：DB + Redis 任一不通就 503。

    检查项：
      - MySQL：跑 SELECT 1，验证连接池 + 网络。
      - Redis：PING，验证密码 + RediSearch 模块所在实例。
      - 不查 LLM（避免每次探活消耗 token）。

    返回 dict，会被 UnifiedResponseMiddleware 包成 {code, msg, data}。
    监控系统看 HTTP status_code 即可（200/503）。
    """
    db_ok = False
    redis_ok = False
    db_err = ""
    redis_err = ""

    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:  # noqa: BLE001 — 探活要吞所有错
        db_err = type(e).__name__
        logger.warning("[health] db check failed: %s", e)

    try:
        redis_ok = bool(_redis.ping())
    except Exception as e:  # noqa: BLE001
        redis_err = type(e).__name__
        logger.warning("[health] redis check failed: %s", e)

    body = {
        "ok": db_ok and redis_ok,
        "db": "up" if db_ok else f"down ({db_err})",
        "redis": "up" if redis_ok else f"down ({redis_err})",
    }
    if not (db_ok and redis_ok):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return body
