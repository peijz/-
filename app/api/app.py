from __future__ import annotations

from dotenv import load_dotenv

# 必须在 import app.model 之前，否则 db.py 读不到 DB_URL
load_dotenv()

from fastapi import FastAPI

from app.agent import zygchat
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.logging_middleware import RequestLoggingMiddleware
from app.api.response import UnifiedResponseMiddleware
from app.api.stt import router as stt_router
from app.api.tongue import router as tongue_router
from app.model.db import init_db


def create_app() -> FastAPI:
    """构造 FastAPI 应用，挂载中间件 + 路由 + 启动钩子。

    中间件顺序（自下而上）：
      1) UnifiedResponseMiddleware（内层）：先把响应包成 ``{code, msg, data}``。
      2) RequestLoggingMiddleware（外层）：再记录访问日志，看到的是包络后的响应。

    为什么是这个顺序：
      - 日志中间件要在最外层，否则它读到的响应是 ``{detail:...}`` 而不是包络后的形式，
        不利于审计 / 排障。
      - 注意：FastAPI 中间件的注册顺序跟执行顺序相反（后注册的最先执行）。
    """
    app = FastAPI(title="zhongyiguan API", version="0.1.0")
    app.add_middleware(UnifiedResponseMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(health_router)
    app.include_router(tongue_router)
    app.include_router(stt_router)

    @app.on_event("startup")
    async def _startup() -> None:
        """应用启动时建表 + 初始化 AsyncRedisSaver（在 event loop 里跑）。"""
        init_db()
        await zygchat.ensure_checkpointer_setup()

    return app


# 模块级 app 实例，给 uvicorn 直接 import 用
app = create_app()