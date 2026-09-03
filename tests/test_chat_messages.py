"""POST /chat/sessions/{session_id}/messages 的测试用例参考。

运行前：
    pip install pytest

跑：
    pytest tests/test_chat_messages.py -v

DB 用内存 SQLite（StaticPool），不污染真实数据。JWT 直接签，跳过登录。
"""

from __future__ import annotations

import json
import time
from typing import Iterator

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def engine():
    """独立的内存 SQLite 引擎，整个测试模块共用一份表。"""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from app.model.db import Base
    from app.model import chat_log, users  # noqa: F401  注册模型到 Base.metadata
    Base.metadata.create_all(eng)
    yield eng


@pytest.fixture
def db_session(engine) -> Iterator:
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Session()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def client(monkeypatch, db_session):
    """把 app 内部的 SessionLocal 全部替换为测试 session，再返回 TestClient。"""
    from app.model import db as db_mod
    from app.api import auth, chat
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(auth, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(chat, "SessionLocal", lambda: db_session)

    from app.api.app import app
    return TestClient(app)


@pytest.fixture
def seeded(db_session):
    """种一个用户 + 一个会话 + 15 条消息，返回 (userid, session_id)。"""
    userid = "test-user-aaa"
    sid = "session-001"
    from app.model.chat_log import ChatSession, ChatMessage

    if not db_session.query(ChatSession).filter_by(id=sid).first():
        db_session.add(ChatSession(id=sid, user_id=userid, system_prompt=None))
        for i in range(15):
            db_session.add(ChatMessage(
                id=i + 1,  # SQLite 不会给 BigInteger PK 自增，手动指定
                session_id=sid,
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
            ))
        db_session.commit()
    return userid, sid


@pytest.fixture
def make_auth_header():
    """签一个合法的 access_token，绕开登录。返回 (userid) -> headers dict。"""
    from app.api.auth import JWT_ALGORITHM, JWT_SECRET

    def _make(userid: str) -> dict[str, str]:
        now = int(time.time())
        token = jwt.encode(
            {
                "sub": userid,
                "type": "access",
                "role": "user",
                "iat": now,
                "exp": now + 3600,
            },
            JWT_SECRET,
            algorithm=JWT_ALGORITHM,
        )
        return {"Authorization": f"Bearer {token}"}

    return _make


@pytest.fixture
def seeded_media(db_session):
    """种一个用户 + 一个会话 + 9 条覆盖各分支的消息，返回 (userid, session_id)。

    消息编排（id, role, 入库 content）：
      101 user    "普通文本"                        → MsgType=1
      102 user    media {text, images=[2]}         → MsgType=3
      103 user    media {videos=[1]}                → MsgType=43
      104 user    media {text only}                 → MsgType=1
      105 user    media {images+video}              → MsgType=43（video 优先）
      106 user    media {empty text+empty media}    → MsgType=1, Content=""
      107 asst    "AI 回复"                          → FromUserName=ai_zhongyi
      108 system  "系统提示"                         → FromUserName=ai_zhongyi
      109 user    "{malformed json"                 → MsgType=1, Content=raw（回落）

    id 用 100+ 是为了避开 seeded fixture 用的 1-15 区间，避免共享 engine 时的 UNIQUE 冲突。
    """
    userid = "test-user-media"
    sid = "session-media"
    from app.model.chat_log import ChatSession, ChatMessage

    if not db_session.query(ChatSession).filter_by(id=sid).first():
        db_session.add(ChatSession(id=sid, user_id=userid, system_prompt=None))
        db_session.add(ChatMessage(id=101, session_id=sid, role="user", content="普通文本"))
        db_session.add(ChatMessage(
            id=102, session_id=sid, role="user",
            content=json.dumps({"type": "media", "text": "看图",
                                "images": ["https://cdn/a.jpg", "https://cdn/b.jpg"],
                                "videos": []}, ensure_ascii=False),
        ))
        db_session.add(ChatMessage(
            id=103, session_id=sid, role="user",
            content=json.dumps({"type": "media", "text": "",
                                "images": [],
                                "videos": ["https://cdn/v.mp4"]}, ensure_ascii=False),
        ))
        db_session.add(ChatMessage(
            id=104, session_id=sid, role="user",
            content=json.dumps({"type": "media", "text": "只有文本",
                                "images": [], "videos": []}, ensure_ascii=False),
        ))
        db_session.add(ChatMessage(
            id=105, session_id=sid, role="user",
            content=json.dumps({"type": "media", "text": "both",
                                "images": ["https://cdn/i.jpg"],
                                "videos": ["https://cdn/v.mp4"]}, ensure_ascii=False),
        ))
        db_session.add(ChatMessage(
            id=106, session_id=sid, role="user",
            content=json.dumps({"type": "media", "text": "",
                                "images": [], "videos": []}, ensure_ascii=False),
        ))
        db_session.add(ChatMessage(id=107, session_id=sid, role="assistant", content="AI 回复"))
        db_session.add(ChatMessage(id=108, session_id=sid, role="system", content="系统提示"))
        db_session.add(ChatMessage(id=109, session_id=sid, role="user", content="{malformed json"))
        db_session.commit()
    return userid, sid


def _items_by_id(items, msg_id):
    """按 MsgId 找单条消息，断言存在。"""
    matches = [i for i in items if i["MsgId"] == msg_id]
    assert len(matches) == 1, f"MsgId={msg_id} 应唯一，实际 {len(matches)} 条"
    return matches[0]


# ---------- 测试用例 ----------

URL = "/chat/messages"


def test_default_pagination(client, seeded, make_auth_header):
    """首次加载不传 cursor：默认 page_size=30，15 条全返回，DESC 顺序，has_more=false。"""
    userid, _ = seeded
    r = client.post(
        URL,
        headers=make_auth_header(userid),
        json={"userid": userid},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 200
    data = body["data"]
    assert data["page_size"] == 30
    assert data["has_more"] is False
    assert data["next_cursor"] is None
    # DESC：msg-14 在前，msg-0 在尾
    assert len(data["items"]) == 15
    assert data["items"][0]["Content"] == "msg-14"
    assert data["items"][-1]["Content"] == "msg-0"


def test_custom_pagination_first_page(client, seeded, make_auth_header):
    """首次加载 page_size=5：返回最新 5 条 DESC，has_more=true, next_cursor 指向末条。"""
    userid, _ = seeded
    r = client.post(
        URL,
        headers=make_auth_header(userid),
        json={"userid": userid, "page_size": 5},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["page_size"] == 5
    assert data["has_more"] is True
    assert data["next_cursor"] == 11  # 末条 MsgId（DESC 排序里最小）
    assert [m["Content"] for m in data["items"]] == ["msg-14", "msg-13", "msg-12", "msg-11", "msg-10"]


def test_each_item_includes_session_id(client, seeded, make_auth_header):
    """每条消息应带 session_id，app 端能区分来自哪个会话。"""
    userid, sid = seeded
    r = client.post(URL, headers=make_auth_header(userid), json={"userid": userid})
    assert r.status_code == 200
    for item in r.json()["data"]["items"]:
        assert item["SessionId"] == sid


def test_userid_mismatch_returns_403(client, seeded, make_auth_header):
    """body 的 userid 与 token 的 sub 不一致 → 403。"""
    userid, _ = seeded
    r = client.post(
        URL,
        headers=make_auth_header(userid),
        json={"userid": "another-user-id"},
    )
    assert r.status_code == 403
    assert r.json()["msg"] == "无权访问该用户的对话"


def test_missing_userid_returns_422(client, seeded, make_auth_header):
    """body 缺 userid → 422，msg 指明字段。"""
    userid, _ = seeded
    r = client.post(URL, headers=make_auth_header(userid), json={})
    assert r.status_code == 422
    assert "userid" in r.json()["msg"]


def test_missing_token_returns_401(client, seeded):
    """无 Authorization 头 → 401。"""
    r = client.post(URL, json={"userid": "x"})
    assert r.status_code == 401


def test_page_size_over_limit_returns_422(client, seeded, make_auth_header):
    """page_size > 100 → 422。"""
    userid, _ = seeded
    r = client.post(
        URL,
        headers=make_auth_header(userid),
        json={"userid": userid, "page_size": 500},
    )
    assert r.status_code == 422
    assert "page_size" in r.json()["msg"]


# ---------- 微信风格 schema 覆盖测试 ----------

def test_plain_text_uses_msgtype_1(client, seeded_media, make_auth_header):
    """user 纯文本 → MsgType=1, Content 原样, ImgStatus=0。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 101)
    assert m["MsgType"] == 1
    assert m["Content"] == "普通文本"
    assert m["ImgStatus"] == 0


def test_image_media_uses_msgtype_3(client, seeded_media, make_auth_header):
    """user 发的 media 含 images → MsgType=3, Content 是英文逗号分隔的 URL 列表, ImgStatus=1。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 102)
    assert m["MsgType"] == 3
    assert m["Content"] == "https://cdn/a.jpg,https://cdn/b.jpg"
    assert m["ImgStatus"] == 1


def test_video_media_uses_msgtype_43(client, seeded_media, make_auth_header):
    """user 发的 media 含 videos → MsgType=43, ImgStatus=1。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 103)
    assert m["MsgType"] == 43
    assert m["Content"] == "https://cdn/v.mp4"
    assert m["ImgStatus"] == 1


def test_text_only_media_falls_back_to_msgtype_1(client, seeded_media, make_auth_header):
    """media JSON 仅含 text → MsgType=1, Content=text。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 104)
    assert m["MsgType"] == 1
    assert m["Content"] == "只有文本"
    assert m["ImgStatus"] == 0


def test_video_takes_precedence_over_image(client, seeded_media, make_auth_header):
    """同一 media 同时含 images 和 videos → 视频优先，MsgType=43。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 105)
    assert m["MsgType"] == 43
    assert m["Content"] == "https://cdn/v.mp4"
    assert m["ImgStatus"] == 1


def test_empty_media_payload_returns_empty_text(client, seeded_media, make_auth_header):
    """media JSON 全空 → MsgType=1, Content=""，不抛异常。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 106)
    assert m["MsgType"] == 1
    assert m["Content"] == ""
    assert m["ImgStatus"] == 0


def test_malformed_json_falls_back_to_raw_text(client, seeded_media, make_auth_header):
    """user content 形如 JSON 但解析失败 → 回落到 MsgType=1 + 原内容，不丢消息。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 109)
    assert m["MsgType"] == 1
    assert m["Content"] == "{malformed json"


def test_user_role_sets_from_userid_to_ai(client, seeded_media, make_auth_header):
    """user 消息: FromUserName=userid, ToUserName='ai_zhongyi'。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 101)
    assert m["FromUserName"] == userid
    assert m["ToUserName"] == "ai_zhongyi"


def test_assistant_role_sets_from_ai_to_userid(client, seeded_media, make_auth_header):
    """assistant 消息: FromUserName='ai_zhongyi', ToUserName=userid，与 user 镜像。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 107)
    assert m["FromUserName"] == "ai_zhongyi"
    assert m["ToUserName"] == userid


def test_system_role_treated_as_ai_side(client, seeded_media, make_auth_header):
    """system 角色: 同样填 FromUserName=ai_zhongyi（不让系统消息冒充用户）。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 108)
    assert m["FromUserName"] == "ai_zhongyi"
    assert m["ToUserName"] == userid


def test_create_time_is_unix_seconds(client, seeded_media, make_auth_header):
    """CreateTime 必须是秒级 unix 时间戳（int），不是毫秒也不是 ISO 字符串。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 101)
    assert isinstance(m["CreateTime"], int)
    assert 1700000000 < m["CreateTime"] < 2000000000  # 2023–2033


def test_msg_id_and_new_msg_id_match(client, seeded_media, make_auth_header):
    """MsgId 是 int，NewMsgId 是其字符串形式，前端可任选其一做去重 key。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    m = _items_by_id(items, 101)
    assert isinstance(m["MsgId"], int)
    assert m["NewMsgId"] == str(m["MsgId"])


def test_status_and_img_status_defaults(client, seeded_media, make_auth_header):
    """Status=2（已送达）、ImgStatus 跟随 MsgType。"""
    userid, _ = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    text_m = _items_by_id(items, 101)
    assert text_m["Status"] == 2
    assert text_m["ImgStatus"] == 0

    img_m = _items_by_id(items, 102)
    assert img_m["Status"] == 2
    assert img_m["ImgStatus"] == 1


def test_session_id_preserved_per_message(client, seeded_media, make_auth_header):
    """SessionId 在多会话场景下也要带回，前端做会话分组用。"""
    userid, sid = seeded_media
    items = client.post(URL, headers=make_auth_header(userid),
                        json={"userid": userid}).json()["data"]["items"]
    for it in items:
        assert it["SessionId"] == sid


# ---------- IM 翻历史（cursor）测试 ----------


def _load(client, headers, userid, **extra):
    body = {"userid": userid, **extra}
    return client.post(URL, headers=headers, json=body).json()["data"]


def test_first_load_returns_newest_first_desc(client, seeded, make_auth_header):
    """首次加载：items[0] 是 MsgId 最大的那条（DESC 最新优先）。"""
    userid, _ = seeded
    data = _load(client, make_auth_header(userid), userid, page_size=10)
    msg_ids = [m["MsgId"] for m in data["items"]]
    assert msg_ids == sorted(msg_ids, reverse=True)


def test_paginate_using_cursor_walks_history(client, seeded, make_auth_header):
    """三步 cursor 翻页：完整走完 15 条消息，前后不重叠。"""
    userid, _ = seeded
    headers = make_auth_header(userid)

    # 第 1 页：最新 5 条
    page1 = _load(client, headers, userid, page_size=5)
    assert [m["MsgId"] for m in page1["items"]] == [15, 14, 13, 12, 11]
    assert page1["has_more"] is True
    assert page1["next_cursor"] == 11

    # 第 2 页：传 cursor 取更早的
    page2 = _load(client, headers, userid, page_size=5, before_msg_id=page1["next_cursor"])
    assert [m["MsgId"] for m in page2["items"]] == [10, 9, 8, 7, 6]
    assert page2["has_more"] is True
    assert page2["next_cursor"] == 6

    # 第 3 页：剩 5 条，has_more=false
    page3 = _load(client, headers, userid, page_size=5, before_msg_id=page2["next_cursor"])
    assert [m["MsgId"] for m in page3["items"]] == [5, 4, 3, 2, 1]
    assert page3["has_more"] is False
    assert page3["next_cursor"] is None


def test_paginate_with_smaller_page_size(client, seeded, make_auth_header):
    """page_size=1 单条翻页：每步都翻到底。"""
    userid, _ = seeded
    headers = make_auth_header(userid)

    seen = []
    cursor = None
    while True:
        kw = {"page_size": 1}
        if cursor is not None:
            kw["before_msg_id"] = cursor
        data = _load(client, headers, userid, **kw)
        seen.extend(m["MsgId"] for m in data["items"])
        if not data["has_more"]:
            break
        cursor = data["next_cursor"]
        assert cursor is not None

    assert seen == list(range(15, 0, -1))


def test_has_more_false_when_total_under_page_size(client, seeded, make_auth_header):
    """消息数 ≤ page_size 时 has_more=false, next_cursor=null。"""
    userid, _ = seeded
    data = _load(client, make_auth_header(userid), userid, page_size=100)
    assert data["has_more"] is False
    assert data["next_cursor"] is None
    assert len(data["items"]) == 15


def test_before_msg_id_below_min_returns_empty(client, seeded, make_auth_header):
    """before_msg_id 小于最小 MsgId（已翻到底）→ items=[], has_more=false, next_cursor=null。"""
    userid, _ = seeded
    data = _load(client, make_auth_header(userid), userid,
                 page_size=10, before_msg_id=1)
    assert data["items"] == []
    assert data["has_more"] is False
    assert data["next_cursor"] is None


def test_next_cursor_null_when_page_size_exactly_matches_remaining(client, seeded, make_auth_header):
    """刚好翻完时 has_more=false 且 next_cursor=null（多取 1 条判断逻辑）。"""
    userid, _ = seeded
    headers = make_auth_header(userid)

    # page_size=5：先翻 3 页，前 2 页 has_more=true，最后 1 页 has_more=false
    page1 = _load(client, headers, userid, page_size=5)
    page2 = _load(client, headers, userid, page_size=5, before_msg_id=page1["next_cursor"])
    page3 = _load(client, headers, userid, page_size=5, before_msg_id=page2["next_cursor"])

    assert page3["has_more"] is False
    assert page3["next_cursor"] is None
    # 验证数据完整：msg-4..msg-0（id=5..1，DESC 排序）
    assert [m["Content"] for m in page3["items"]] == ["msg-4", "msg-3", "msg-2", "msg-1", "msg-0"]
