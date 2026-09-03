from __future__ import annotations

import json
from typing import Any, Optional

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# =====================================================================
# 统一响应包络（{code, msg, data}）
#    - 前端约定所有 JSON 接口都长这样，方便统一解析错误码
#    - 不在域内处理的：SSE 流 / 204 / OpenAPI 文档
# =====================================================================

# OpenAPI 相关路径不走包络，让 Swagger UI / Redoc 能拿到原 schema
_DOCS_PATHS = {"/docs", "/redoc", "/openapi.json"}

# 各 HTTP 状态码对应的中文成功提示；前端拿到 201 时也能显示 "创建成功"
_SUCCESS_MSGS: dict[int, str] = {
    200: "成功",
    201: "创建成功",
    202: "已接受",
    204: "成功",
}


def ok(data: Any = None, code: int = 200, msg: Optional[str] = None) -> dict[str, Any]:
    """构造一个 2xx 成功响应体。

    :param data: 业务数据，可为 None / dict / list / 标量。
    :param code: 业务码（默认跟 HTTP status 一致，前端拿来判定细分场景）。
    :param msg: 文案；为空时按 status code 自动填 "成功" / "创建成功" 等。
    """
    return {"code": code, "msg": msg or _SUCCESS_MSGS.get(code, "成功"), "data": data}


def fail(code: int, msg: str, data: Any = None) -> dict[str, Any]:
    """构造一个失败响应体（HTTP status 通常也是这个 code，但这里 code 是业务码）。

    :param code: 业务码（建议跟 HTTP status 对齐，便于前端统一处理）。
    :param msg: 错误描述，会原样显示给用户。
    :param data: 附加数据（默认 None）。
    """
    return {"code": code, "msg": msg, "data": data}


class UnifiedResponseMiddleware(BaseHTTPMiddleware):
    """把所有 JSON 响应统一包装成 ``{code, msg, data}``。

    处理规则：
      - ``/docs`` / ``/redoc`` / ``/openapi.json``：放行原响应（OpenAPI 工具需要原 JSON）。
      - 状态码 204 / 非 application/json（文件下载、SSE 流）：放行原响应。
      - 2xx 成功响应：把原 payload 塞进 ``data``，msg 默认为 "成功"。
      - 4xx/5xx 失败响应：把 FastAPI 默认的 ``{detail: ...}`` 翻译成 ``msg``，
        ``code`` 直接用 HTTP 状态码，``data`` 置 None。

    实现要点：必须把 ``body_iterator`` 整个读完才能重新打包，所以本中间件会
    一次性把响应体读完再返回新 JSONResponse；对 SSE 不适用（已经前置放行）。
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _DOCS_PATHS:
            return await call_next(request)

        response = await call_next(request)

        ctype = response.headers.get("content-type", "")
        # 204（No Content）或非 JSON 响应：直接放行，不包络
        if response.status_code == 204 or "application/json" not in ctype:
            return response

        # 一次性把 body 读完，因为后续要用新的 JSONResponse 替代原响应
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        try:
            original: Any = json.loads(body) if body else None
        except json.JSONDecodeError:
            # 极端情况：业务返回了非 JSON 但 header 写 application/json，原样回传
            original = None

        wrapped = _wrap(response.status_code, original)
        return JSONResponse(content=wrapped, status_code=response.status_code)


def _wrap(status_code: int, original: Any) -> dict[str, Any]:
    """按 HTTP status 把原响应翻译成统一包络。

    :param status_code: HTTP 状态码。
    :param original: 原响应 payload（dict / str / list / None）。
    :return: ``{code, msg, data}`` 结构。
    """
    if 200 <= status_code < 300:
        # 幂等：如果业务自己已经返回包络格式（如 SSE 流后的最后一个 end 包），
        # 就别再包一层，直接透传。
        if isinstance(original, dict) and {"code", "msg", "data"}.issubset(original):
            return original
        return ok(data=original, code=status_code)

    # 4xx/5xx：从 FastAPI 默认的 {detail: ...} 里抽 msg
    msg = "请求失败"
    if isinstance(original, dict):
        if "detail" in original:
            msg = _flatten_detail(original["detail"])
        elif "msg" in original and isinstance(original["msg"], str):
            msg = original["msg"]
    elif isinstance(original, str) and original:
        msg = original
    return fail(code=status_code, msg=msg)


def _flatten_detail(detail: Any) -> str:
    """FastAPI HTTPException 的 detail 字段形态多样，统一翻译成单行字符串。

    可能遇到的形态：
      - str：直接返回。
      - list[dict]：Pydantic ValidationError 风格 → 取第一个错误的 "字段名: 消息"。
      - list[str]：取第一个元素。
    :param detail: HTTPException(detail=...) 传入的内容。
    :return: 适合直接展示给用户的单行错误描述。
    """
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict) and "msg" in first:
            loc = first.get("loc", [])
            # loc 是 ["body", "field_name", ...]，去掉最外层 "body"
            field = ".".join(str(p) for p in loc[1:]) if len(loc) > 1 else str(loc[0]) if loc else ""
            return f"{field}: {first['msg']}" if field else first["msg"]
        if isinstance(first, str):
            return first
    return "请求失败"