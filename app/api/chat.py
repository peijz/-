from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.agent import zygchat
from app.api.auth import get_current_userid
from app.model.chat_log import ChatMessage, ChatSession
from app.model.db import SessionLocal

router = APIRouter(prefix="/chat", tags=["chat"])


# =====================================================================
# 微信风格消息结构（前端直接渲染 IM 气泡）
# =====================================================================

class FollowUp(BaseModel):
    """智能体发起的结构化追问，app 端据此渲染选项。

    :param question: 提问文案，1~200 字。
    :param options: 候选选项文本列表，长度 2~6，每项必须非空。
    """
    question: str = Field(..., min_length=1, max_length=200)
    options: list[str] = Field(..., min_length=2, max_length=6)

    @field_validator("options")
    @classmethod
    def _options_nonempty(cls, v: list[str]) -> list[str]:
        """校验每项都是非空字符串并 strip；空项直接 422。"""
        for o in v:
            if not isinstance(o, str) or not o.strip():
                raise ValueError("option must be non-empty string")
        return [o.strip() for o in v]


# 别名：避免字段名 FollowUp 与类型名 FollowUp 在 from __future__ import annotations 下冲突
# （Pydantic 解析字段注解时会把 FollowUp 误解析为字段自身默认值 None）
_FollowUpT = FollowUp


class MessageOut(BaseModel):
    """仿微信 AddMsgList 的单条消息结构，前端可直接渲染聊天气泡。

    字段命名沿用微信的 PascalCase 是为了让前端可以无差别复用 IM SDK 的解析逻辑。
    """

    MsgId: int
    NewMsgId: str
    FromUserName: str
    ToUserName: str
    MsgType: int
    Content: str
    CreateTime: int
    Status: int = 2
    ImgStatus: int = 0
    PlayLength: Optional[int] = None
    MsgSource: str = ""
    PushContent: str = ""
    SessionId: Optional[str] = None  # 非微信字段，多会话前端分组用
    FollowUp: Optional[_FollowUpT] = None  # AI 消息带追问时填充；用户消息为 null

    model_config = {"from_attributes": True}


class MessagesIn(BaseModel):
    """拉取消息列表的请求体（IM 风格：cursor 翻历史）。"""
    userid: str = Field(..., description="目标用户的 userid，必须与 token 中的 sub 一致")
    page_size: int = Field(30, ge=1, le=100, description="每页条数，默认 30，最大 100")
    before_msg_id: Optional[int] = Field(
        default=None,
        description="翻历史 cursor：返回 id < before_msg_id 的消息；首次加载不传",
    )


class MessagesOut(BaseModel):
    """IM 风格分页响应：倒序 + cursor 翻历史。

    :param items: 当前页消息列表（id 降序）。
    :param has_more: 是否还有更早消息可翻。
    :param next_cursor: 下一轮翻历史的 before_msg_id；has_more=false 时为 None。
    """
    items: list[MessageOut]
    has_more: bool = Field(..., description="是否还有更早的消息可翻")
    next_cursor: Optional[int] = Field(
        default=None,
        description="下一轮翻历史的 before_msg_id；has_more=false 时为 null",
    )
    page_size: int


class SendMessageIn(BaseModel):
    """发送消息请求体（app 端给 AI 发）。"""
    userid: str = Field(..., description="目标用户的 userid，必须与 token 中的 sub 一致")
    session_id: Optional[str] = Field(
        default=None,
        description="已存在的会话 id；为空则服务端自动创建新会话",
    )
    text: Optional[str] = Field(default=None, max_length=4000)
    images: list[str] = Field(default_factory=list, description="图片 URL 列表")
    videos: list[str] = Field(default_factory=list, description="视频 URL 列表")

    @field_validator("text")
    @classmethod
    def _text_strip(cls, v: Optional[str]) -> Optional[str]:
        """空字符串视为 None，避免入库全是空白。"""
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("images", "videos")
    @classmethod
    def _url_format(cls, v: list[str]) -> list[str]:
        """URL 必须以 http(s):// 开头（防内网 SSRF 滥用）。"""
        for url in v:
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ValueError("URL 必须以 http:// 或 https:// 开头")
        return v


class SendMessageOut(BaseModel):
    """发送消息响应。

    :param session_id: 本轮会话 id（新建时下发）。
    :param user_message: 入库后的用户消息（已转 IM 风格）。
    :param ai_message: 入库后的 AI 消息；agent 暂时无回复时为 None。
    :param ai_reply: AI 纯文本回复（不含追问块），便于前端快速渲染。
    :param follow_up: 结构化追问；用户选中选项后作为下一条 user message 发回。
    """
    session_id: str
    user_message: MessageOut
    ai_message: Optional[MessageOut] = Field(
        default=None,
        description="agent 返回为空时为 null",
    )
    ai_reply: str = Field(default="", description="AI 的纯文本回复（不含追问块），便于前端快速渲染")
    follow_up: Optional[_FollowUpT] = Field(
        default=None,
        description="智能体结构化追问；用户选中选项后作为下一条 user message 发回",
    )


# =====================================================================
# Helpers：消息 / 数据库行 → IM 风格 MessageOut
# =====================================================================

# 微信风格用户名：所有 AI 消息统一用这个名字作为 sender
_AI_USERNAME = "ai_zhongyi"

# 微信 MsgType 常量
_MSG_TYPE_TEXT = 1
_MSG_TYPE_IMAGE = 3
_MSG_TYPE_VIDEO = 43
_MSG_TYPE_SYSTEM = 10000


def _serialize_content(role: str, raw_content: str) -> tuple[int, str, int]:
    """按 role + 入库 content 推导微信风格的 ``(MsgType, Content, ImgStatus)``。

    规则：
      - 用户消息且入库 content 是 JSON {"type":"media",...}：
          - 有视频 → (VIDEO, "<url1>,<url2>", 1)
          - 有图片 → (IMAGE, "<url1>,<url2>", 1)
          - 否则   → (TEXT,   "<text>", 0)
      - AI 消息含追问：入库是 JSON {"type":"ai_followup",...}，
        Content 仍走 text 字段（追问数据走 MessageOut.FollowUp，前端按需读）。
      - 其他（纯文本 / system / tool）：(TEXT, raw_content, 0)。
    """
    # 用户消息的 media JSON：{"type":"media","text":...,"images":...,"videos":...}
    if role == "user" and raw_content.startswith("{"):
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("type") == "media":
            videos = parsed.get("videos") or []
            images = parsed.get("images") or []
            text = parsed.get("text") or ""
            if videos:
                return _MSG_TYPE_VIDEO, ",".join(videos), 1
            if images:
                return _MSG_TYPE_IMAGE, ",".join(images), 1
            if text:
                return _MSG_TYPE_TEXT, text, 0
            return _MSG_TYPE_TEXT, "", 0
    # assistant 带追问：{"type":"ai_followup","text":"...","follow_up":{...}}
    # Content 仍走纯文本（text 字段），追问数据走 MessageOut.FollowUp，渲染端按需读
    # assistant / tool / system：当前都是纯文本；system 也归到 1，不触发系统提示样式
    return _MSG_TYPE_TEXT, raw_content or "", 0


def _extract_follow_up(role: str, raw_content: str) -> Optional[FollowUp]:
    """从 AI 消息入库 content 中提取 follow_up（若存在）。

    仅当 role == "assistant" 且 content 是 ``{"type":"ai_followup",...}`` JSON 才解析。
    解析失败（含字段缺失 / 类型不对）→ 返回 None，前端会按无追问处理。
    """
    if role != "assistant" or not raw_content or not raw_content.startswith("{"):
        return None
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("type") != "ai_followup":
        return None
    fu = parsed.get("follow_up")
    if not isinstance(fu, dict):
        return None
    try:
        return FollowUp(
            question=fu.get("question", ""),
            options=fu.get("options", []),
        )
    except Exception:
        return None


def _extract_visible_text(role: str, raw_content: str) -> str:
    """从入库 content 中提取纯可见文本（剥离 ai_followup / user media 的 JSON 包装）。"""
    if not raw_content:
        return ""
    if not raw_content.startswith("{"):
        return raw_content
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return raw_content
    if not isinstance(parsed, dict):
        return raw_content
    if parsed.get("type") == "ai_followup":
        return parsed.get("text") or ""
    if parsed.get("type") == "media" and role == "user":
        return parsed.get("text") or ""
    return raw_content


def _row_to_message_out(row: dict[str, Any], userid: str) -> MessageOut:
    """把数据库行（dict）转成 IM 风格 MessageOut。

    :param row: 形如 ``add_message`` 返回的 dict。
    :param userid: 当前用户 userid（用于填充 FromUserName / ToUserName）。
    """
    content = row.get("content", "") or ""
    role = row.get("role", "unknown")
    msg_type, content_out, img_status = _serialize_content(role, content)
    visible_text = _extract_visible_text(role, content) or content_out
    follow_up = _extract_follow_up(role, content)

    is_user = role == "user"
    created_at: datetime = row["created_at"]

    return MessageOut(
        MsgId=row["id"],
        NewMsgId=str(row["id"]),
        FromUserName=userid if is_user else _AI_USERNAME,
        ToUserName=_AI_USERNAME if is_user else userid,
        MsgType=msg_type,
        Content=content_out,
        CreateTime=int(created_at.timestamp()),
        Status=2,
        ImgStatus=img_status,
        MsgSource="",
        PushContent=visible_text[:50],
        SessionId=row.get("session_id"),
        FollowUp=follow_up,
    )


# =====================================================================
# 路由
# =====================================================================

@router.post(
    "/messages",
    response_model=MessagesOut,
    summary="获取某用户的消息（IM 风格：倒序首次加载 + cursor 翻历史）",
)
def list_messages(
    payload: MessagesIn,
    token_userid: str = Depends(get_current_userid),
) -> MessagesOut:
    """拉取某用户的消息列表（按 id 倒序，多取 1 条用于判断 has_more）。

    安全要点：
      - payload.userid 必须 == token_userid（防越权读他人消息）。
      - 用 (session.user_id == userid) 过滤，跨用户无可能。
    """
    if payload.userid != token_userid:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail="无权访问该用户的对话"
        )

    with SessionLocal() as db:
        base = (
            db.query(ChatMessage)
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .filter(ChatSession.user_id == payload.userid)
        )
        if payload.before_msg_id is not None:
            base = base.filter(ChatMessage.id < payload.before_msg_id)

        # 多取 1 条用于判断 has_more，避免额外 COUNT 查询
        rows = base.order_by(ChatMessage.id.desc()).limit(payload.page_size + 1).all()
        has_more = len(rows) > payload.page_size
        items = rows[: payload.page_size]
        next_cursor = items[-1].id if has_more and items else None

        return MessagesOut(
            items=[_row_to_message_out(_row_to_dict(r), payload.userid) for r in items],
            has_more=has_more,
            next_cursor=next_cursor,
            page_size=payload.page_size,
        )


def _row_to_dict(row: ChatMessage) -> dict[str, Any]:
    """ORM ChatMessage → 入库 dict，与 db_log.add_message 返回结构一致。"""
    return {
        "id": row.id,
        "session_id": row.session_id,
        "role": row.role,
        "content": row.content,
        "tool_call_id": row.tool_call_id,
        "tool_calls": row.tool_calls,
        "name": row.name,
        "created_at": row.created_at,
    }


@router.post(
    "/send",
    response_model=SendMessageOut,
    summary="app 端发送消息给 AI；消息可含文字/图片/视频",
)
def send_message(
    payload: SendMessageIn,
    token_userid: str = Depends(get_current_userid),
) -> SendMessageOut:
    """非流式发送：调用 ``zygchat.chat`` 等 AI 回复完整后一次性返回。

    与 ``send_message_stream`` 行为完全一致，仅传输方式不同（一次性 JSON vs SSE）。
    """
    if payload.userid != token_userid:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail="无权访问该用户"
        )
    if not payload.text and not payload.images and not payload.videos:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="text / images / videos 至少需要一项",
        )

    try:
        session_id, user_row, ai_row, ai_reply, follow_up = zygchat.chat(
            user_id=payload.userid,
            text=payload.text,
            images=payload.images,
            videos=payload.videos,
            session_id=payload.session_id,
        )
    except ValueError as e:
        msg = str(e)
        if "不存在" in msg or "不属于" in msg:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=msg)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=msg)

    return SendMessageOut(
        session_id=session_id,
        user_message=_row_to_message_out(user_row, payload.userid),
        ai_message=_row_to_message_out(ai_row, payload.userid) if ai_row else None,
        ai_reply=ai_reply,
        follow_up=follow_up,
    )


@router.post(
    "/send/stream",
    summary="app 端流式发送消息给 AI（SSE，token-by-token）",
    response_class=StreamingResponse,
)
async def send_message_stream(
    payload: SendMessageIn,
    token_userid: str = Depends(get_current_userid),
) -> StreamingResponse:
    """SSE 流式发送：逐 token 推送 AI 回复，event=start/chunk/end/error。

    SSE 事件序列（成功）：
      1) ``data: {"event":"start","session_id":...,"user_message":{...}}`` — 立即派发
      2) ``data: {"event":"chunk","content":"<token>"}`` — 多次
      3) ``data: {"event":"end","ai_message":{...},"follow_up":...}`` — 收尾

    异常路径：``data: {"event":"error","message":"...","content":"...","session_id":"..."}``
    """
    if payload.userid != token_userid:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail="无权访问该用户"
        )
    if not payload.text and not payload.images and not payload.videos:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="text / images / videos 至少需要一项",
        )

    async def event_source():
        try:
            async for evt in zygchat.chat_stream(
                user_id=payload.userid,
                text=payload.text,
                images=payload.images,
                videos=payload.videos,
                session_id=payload.session_id,
            ):
                evt = _normalize_stream_event(evt, payload.userid)
                yield f"data: {json.dumps(evt, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            err = {
                "event": "error",
                "message": f"{type(e).__name__}: {e}",
                "content": f"{type(e).__name__}: {e}",
                "session_id": payload.session_id or "",
            }
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 等代理缓冲
        },
    )


def _normalize_stream_event(evt: dict[str, Any], userid: str) -> dict[str, Any]:
    """把 agent 流式事件里的 user_message / ai_message 转换为微信风格 MessageOut。"""
    if evt.get("event") == "start" and isinstance(evt.get("user_message"), dict):
        evt["user_message"] = _row_to_message_out(evt["user_message"], userid).model_dump()
    elif evt.get("event") == "end":
        ai = evt.get("ai_message")
        if isinstance(ai, dict) and ai:
            evt["ai_message"] = _row_to_message_out(ai, userid).model_dump()
        elif ai is None or not ai:
            evt["ai_message"] = None
    return evt