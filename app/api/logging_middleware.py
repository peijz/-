from __future__ import annotations

import json
import logging
import os
import secrets
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

# =====================================================================
# 访问日志（app.access logger）
#    - 记录每个请求的 method / path / status / 耗时 / 请求体 / 响应体
#    - SSE / 流式响应：只记 "<stream>" 占位，不缓存响应体
#    - 敏感字段（password）会自动 mask 为 ***
#    - 单条日志超 8KB 自动截断，避免日志风暴
# =====================================================================

# 单条日志最大长度（字节）：超过会标 ...<TRUNCATED,orig=N>
MAX_BODY = 8 * 1024
# OpenAPI 文档路径：跳过 body 读取（这些路径的 body 没意义，且 log 会很吵）
_SKIP_PATHS = {"/docs", "/redoc", "/openapi.json"}
# 中国时区（业务在国内，全用 +8）
_CN_TZ = timezone(timedelta(hours=8))
# 大小写不敏感：键名命中即替换为 ***
_SENSITIVE_KEYS = {"password"}


def _setup_logger() -> logging.Logger:
    """初始化 app.access logger，挂 StreamHandler，使用 LOG_LEVEL 控制级别。

    用 named logger（propagate=False）避免被根 logger 重复处理。
    """
    log = logging.getLogger("app.access")
    if log.handlers:
        return log
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    log.propagate = False
    return log


logger = _setup_logger()


def _now_iso() -> str:
    """当前时间（中国时区），ISO8601 毫秒精度。"""
    return datetime.now(_CN_TZ).isoformat(timespec="milliseconds")


def _mask(obj: Any) -> Any:
    """递归地把敏感键的值替换为 "***"（保留原结构，方便排查）。"""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k.lower() in _SENSITIVE_KEYS:
                out[k] = "***"
            elif isinstance(v, (dict, list)):
                out[k] = _mask(v)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_mask(x) for x in obj]
    return obj


def _try_parse_json(text: str) -> Any:
    """尝试把字符串解析成 JSON；失败返回 None。"""
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def _log_repr(raw_body: bytes) -> str:
    """把请求 / 响应体序列化成一个适合贴进日志的字符串。

    - 是 JSON 就 dict 后走 mask 再 dump；
    - 非 JSON 用 ``<binary,size=N>`` 占位；
    - 超 MAX_BODY 自动截断并标 ...<TRUNCATED,orig=N>。
    """
    parsed = _try_parse_json(raw_body.decode("utf-8", "replace"))
    if parsed is None:
        return f"<binary,size={len(raw_body)}>"
    s = json.dumps(_mask(parsed), ensure_ascii=False, default=str)
    if len(s) > MAX_BODY:
        return s[:MAX_BODY] + f"...<TRUNCATED,orig={len(s)}>"
    return s


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求 / 响应日志中间件。

    日志格式（一行 key=value 风格，便于日志采集系统解析）：
        [APP-ACCESS] <ISO8601> | req_id=<6字节hex> |
        <METHOD> <PATH>?<QUERY> | status=<code> | dur_ms=<int> |
        ip=<client> | ua=<ua> | req_size=<bytes> | resp_size=<bytes> |
        req_body=<...> | resp_body=<...>

    实现要点：
      - 必须先把 body 整个读出来，再构造一个新的 receive 函数喂给下游，
        否则下游拿不到 body（Starlette 的 request.body() 会清空 buffer）。
      - 流式响应（SSE / StreamingResponse）不缓存 body，只记 "<stream>"，
        否则会把流整个缓存进内存。
      - 异常路径：记 traceback 并重新抛出，让上层统一异常处理继续工作。
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _SKIP_PATHS:
            logger.debug(
                "[APP-ACCESS] %s | %s %s | docs | skip body",
                _now_iso(), request.method, request.url.path,
            )
            return await call_next(request)

        req_id = secrets.token_hex(6)
        t0 = time.perf_counter()

        body_bytes = await request.body()
        req_size = len(body_bytes)
        req_log_repr = _log_repr(body_bytes)

        # 重写 receive：下游再读 body 时直接拿到我们缓存好的 bytes
        async def _receive() -> dict:
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request = Request(request.scope, receive=_receive)

        try:
            response = await call_next(request)
        except Exception:
            dur_ms = int((time.perf_counter() - t0) * 1000)
            tb = traceback.format_exc().replace("\n", " | ")
            logger.error(
                "[APP-ACCESS] %s | req_id=%s | %s %s | EXC | dur_ms=%d | %s",
                _now_iso(), req_id, request.method, request.url.path, dur_ms, tb,
            )
            raise

        ctype = response.headers.get("content-type", "")
        is_stream = isinstance(response, StreamingResponse) or "text/event-stream" in ctype

        if is_stream:
            # 流式响应不缓存 body，只占位；保留原 response 给下游
            resp_log_repr = "<stream>"
            resp_size = 0
            final_response = response
        else:
            # 把 body_iterator 一次性读完，重新打包成新的 Response
            chunks: list[bytes] = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            full_body = b"".join(chunks)
            resp_size = len(full_body)
            resp_log_repr = _log_repr(full_body)
            # 移除 content-length（我们重算了 body，可能跟原 header 不一致）
            new_headers = {
                k: v for k, v in response.headers.items()
                if k.lower() != "content-length"
            }
            final_response = Response(
                content=full_body,
                status_code=response.status_code,
                headers=new_headers,
                media_type=response.media_type,
            )

        dur_ms = int((time.perf_counter() - t0) * 1000)
        ip = request.client.host if request.client else "-"
        ua = request.headers.get("user-agent", "-")
        query = ("?" + request.url.query) if request.url.query else ""
        logger.info(
            "[APP-ACCESS] %s | req_id=%s | %s %s%s | status=%d | dur_ms=%d | "
            "ip=%s | ua=%s | req_size=%d | resp_size=%d | req_body=%s | resp_body=%s",
            _now_iso(), req_id, request.method, request.url.path, query,
            final_response.status_code, dur_ms, ip, ua, req_size, resp_size,
            req_log_repr, resp_log_repr,
        )
        return final_response