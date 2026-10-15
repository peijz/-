"""语音输入(STT)接口:上传音频 → MiniMax STT → 返回识别文本。

本模块是"前置预处理层",不做任何会话/记忆/chat 业务:
  - 拿到识别文本后,前端会自己再调 POST /chat/send/stream 把识别文本塞给 LangGraph。
  - 因此 STT 与 chat 完全解耦,失败可重试,不会污染用户聊天记录。

为什么不直接在本路由内调 LangGraph:
  - 复用"前置预处理层"心智模型,失败重试成本可控。
  - 后续想加"实时字幕""语音听写中"等流式能力时,只改本模块。
  - 不动 zygchat.chat / chat_stream 一行业务代码。

依赖:
  - 复用 get_current_userid (app/api/auth.py)
  - 复用 UnifiedResponseMiddleware (app/api/response.py) 自动包 {code, msg, data}
"""

from __future__ import annotations

import logging
import os
from typing import Optional as _Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.auth import get_current_userid


router = APIRouter(prefix="/stt", tags=["stt"])
logger = logging.getLogger("app.api.stt")


# =====================================================================
# 配置 & 运行时资源
# =====================================================================

_STT_URL = os.getenv("MINIMAX_STT_URL", "https://api.minimaxi.com/v1/speech_to_text")
_STT_MODEL = os.getenv("MINIMAX_STT_MODEL", "")  # 占位,留空 → 运行时拦截
_STT_API_KEY = os.getenv("MINIMAX_API_KEY", "")
_MAX_UPLOAD_MB = int(os.getenv("STT_MAX_UPLOAD_MB", "50"))
_TIMEOUT_SEC = float(os.getenv("STT_TIMEOUT_SEC", "60"))


# MIME 白名单 —— 与 uniapp getRecorderManager() 默认产物 (aac) + 主流浏览器一致。
# 注意:小程序/浏览器可能撒谎 content_type(发 octet-stream),前端契约要求传正确 MIME。
_ALLOWED_MIME = (
    "audio/mpeg",      # .mp3
    "audio/wav",      # .wav
    "audio/x-wav",    # .wav (某些客户端会发 x-wav)
    "audio/mp4",      # .mp4 / .m4a (iOS Safari/小程序通用)
    "audio/x-m4a",    # .m4a 备选
    "audio/m4a",      # .m4a
    "audio/aac",      # .aac (uniapp 默认录音格式)
)


# MIME → 扩展名,Multipart filename 回退用;前端若没给 filename,服务端兜底一个。
_MIME_TO_EXT = {
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/aac": "aac",
}


# =====================================================================
# 路由
# =====================================================================
@router.post(
    "/upload",
    summary="上传一段音频,返回 MiniMax STT 识别出的文本(不进入 chat 流程)",
)
async def upload_stt(
    audio: UploadFile = File(..., description="音频文件(mp3/wav/m4a/aac 等,≤50MB)"),
    language: str = Form("zh", description="音频主语言(默认 zh)"),
    session_id: _Optional[str] = Form(None, description="前端会话 id,本接口不消费,仅记录"),
    _userid: str = Depends(get_current_userid),
) -> dict:
    """接收 multipart 音频 → 调 MiniMax STT → 返回识别文本。

    设计要点:
      - STT 是纯预处理,识别文本由前端自己调 /chat/send/stream 走 LangGraph。
        本路由不创建 ChatSession,不入库,不动 SessionID——失败重试零副作用。
      - 错误码与 /tongue/analyze 风格一致:400 invalid_type / empty_file,413 too_large,401,502。

    :raises HTTPException:
        400 ``invalid_type`` - MIME 不在白名单
        400 ``empty_file``   - 0 字节
        413 ``too_large``    - 超过 STT_MAX_UPLOAD_MB(默认 50MB)
        401                  - 由 get_current_userid 抛
        502 ``upstream_error: <ClassName>`` - httpx 异常 / 配置缺失
        502 ``upstream_unrecognized: <preview>`` - 响应里找不到 text 字段
    """
    if not _STT_API_KEY:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="upstream_error: MINIMAX_API_KEY 未配置",
        )
    if not _STT_MODEL:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="upstream_error: MINIMAX_STT_MODEL 未配置",
        )

    if audio.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="invalid_type",
        )

    raw = await audio.read()
    if len(raw) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="empty_file")

    max_bytes = _MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail="too_large")

    # filename 兜底:前端某些场景(如 base64 直传)可能不带 filename,服务端拼一个
    # 否则 MiniMax multipart 解析时拿到无名文件可能直接 400。
    filename = audio.filename or f"audio.{_MIME_TO_EXT.get(audio.content_type, 'bin')}"

    # session_id 仅记录,不消费;留痕便于排障(前端可能在错误日志里看到)
    if session_id:
        logger.info("[stt] user=%s session_id=%s size=%dB mime=%s",
                    _userid, session_id, len(raw), audio.content_type)

    try:
        text = await _call_minimax_stt(
            raw=raw,
            filename=filename,
            content_type=audio.content_type or "application/octet-stream",
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=f"upstream_error: {e.__class__.__name__}",
        )
    except RuntimeError as e:
        # 配置缺失 / 响应格式异常 等"非网络"问题,统一归到 upstream_error。
        # msg 由前端按 RestfulResponse 中间件转 msg 字段。
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=f"upstream_error: {e}",
        )

    return {"text": text}


# =====================================================================
# 内部:MiniMax 原生 STT 调用
# =====================================================================
async def _call_minimax_stt(
    *,
    raw: bytes,
    filename: str,
    content_type: str,
) -> str:
    """POST 音频到原生 STT 端点,返回识别文本。

    单测里 monkeypatch 这个名字即可模拟上游,不依赖 httpx 写法细节。

    实现要点:
      - httpx ``files=`` 接受 ``(filename, content, content_type)`` 元组,multipart boundary
        自动生成,不要手动拼 boundary。
      - ``data=`` 走普通 form 字段。
      - 响应是 JSON,典型结构 ``{"text": "...", "base_resp": {...}}``。
        若实际是 task-based(返回 task_id 需要轮询),需要把这里改成提交+轮询两段。
      - 返回值不在 try/except 里包,让 route 层捕获 httpx.HTTPError / RuntimeError,
        统一翻译成 502 upstream_error。
    """
    files = {"file": (filename, raw, content_type)}
    # 上游只接 file + model,可选 response_format=json;language 不在 spec,不能传(MiniMax 会 400)
    data = {"model": _STT_MODEL, "response_format": "json"}

    async with httpx.AsyncClient(timeout=_TIMEOUT_SEC) as client:
        resp = await client.post(
            _STT_URL,
            headers={"Authorization": f"Bearer {_STT_API_KEY}"},
            files=files,
            data=data,
        )
    if resp.status_code >= 400:
        # 上游非 2xx 走 RuntimeError 让 route 层 → 502
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        body = resp.json()
    except Exception as e:
        raise RuntimeError(f"response not JSON: {e.__class__.__name__}")

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        # 给排查的人留个 preview;前端只看到 upstream_unrecognized: 前缀,可统一 catch。
        preview = resp.text[:120].replace("\n", " ")
        raise RuntimeError(f"upstream_unrecognized: {preview}")

    return text.strip()