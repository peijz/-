"""POST /stt/upload 的测试用例。

运行:  pytest tests/test_stt.py -v

策略(与 test_tongue.py 一致):
  - DB 用内存 SQLite + StaticPool(conftest.py 已设 DB_URL=sqlite:///:memory:)
  - 通过 monkeypatch ``_call_minimax_stt`` 模拟 MiniMax STT 上游,完全离线可跑
  - 音频字节用几字节 mock bytes,不写盘;不验证真实解码(STT 是黑盒)

前置:
  - 本测试不创建 ChatSession / ChatMessage,STT 与 chat 解耦。
  - 测试顶部通过 os.environ.setdefault 注水 STT_URL/MODEL/API_KEY,避免路由因
    配置缺失提前 502,让 mock 能被实际调用到。
"""

from __future__ import annotations

import json
import os
import time
import asyncio
from typing import Iterator

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# 在 import app 前注水 STT 配置 —— _call_minimax_stt 在模块级读 os.getenv,
# 一旦 _STT_MODEL 为空,路由会先 502 拦在 monkeypatch 之前。
# 注意:不能 setdefault("MINIMAX_API_KEY", ...) ! 否则会把 zygchat.EMBEDDINGS.api_key
# 锁成假 key,让 test_tongue.py 调真实 MiniMax embeddings 1004 鉴权失败。
# 真实 key 已由 conftest.py 加载 .env 保证。
os.environ.setdefault("MINIMAX_STT_URL", "https://api.minimaxi.com/v1/speech_to_text")
os.environ.setdefault("MINIMAX_STT_MODEL", "asr-1.0-test")


# 几字节 mock 音频(长度不重要,只走路由校验,真正解析在 mock 上游)。
_MOCK_AUDIO = b"\x00\x01\x02\x03\x04\x05"


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from app.model.db import Base
    from app.model import chat_log, users  # noqa: F401
    from sqlalchemy import event
    from sqlalchemy.sql import text

    Base.metadata.create_all(eng)

    @event.listens_for(chat_log.ChatMessage, "before_insert")
    def _sqlite_autoinc(mapper, connection, target):
        if target.id is None:
            row = connection.execute(
                text("SELECT COALESCE(MAX(id), 0) + 1 FROM chat_messages")
            ).scalar()
            target.id = row

    yield eng


@pytest.fixture
def db_session(engine) -> Iterator:
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Session()
    try:
        yield s
    finally:
        from app.model.chat_log import ChatMessage, ChatSession
        s.query(ChatMessage).delete()
        s.query(ChatSession).delete()
        s.commit()
        s.close()


@pytest.fixture
def client(monkeypatch, db_session):
    """TestClient + monkeypatch 所有 SessionLocal 引用(STT 本路由不用 DB,留给后续扩展)。"""
    from app.model import db as db_mod
    from app.model import chat_log
    from app.api import auth, chat
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(auth, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(chat, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(chat_log, "SessionLocal", lambda: db_session)

    from app.api.app import app
    return TestClient(app)


@pytest.fixture
def auth_header():
    """签 access_token,模拟已登录。"""
    from app.api.auth import JWT_ALGORITHM, JWT_SECRET

    def _make(userid: str = "test-user-stt") -> dict[str, str]:
        now = int(time.time())
        token = jwt.encode(
            {"sub": userid, "type": "access", "role": "user",
             "iat": now, "exp": now + 3600},
            JWT_SECRET, algorithm=JWT_ALGORITHM,
        )
        return {"Authorization": f"Bearer {token}"}

    return _make


# ---------- monkeypatch 工具 ----------

def _stub_minimax_stt(monkeypatch, payload):
    """monkeypatch app.api.stt._call_minimax_stt 为给定返回。

    payload 三种形态:
      - str:         直接返回(模拟 STT 识别出的文本)
      - Exception:    抛(模拟上游错误)
      - dict:        {"text": "..."} 形式,从里面取 text(模拟真实 JSON 响应)
    """
    async def _fake(*, raw, filename, content_type) -> str:
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return payload["text"]
        raise RuntimeError(f"unexpected payload type: {type(payload)}")

    monkeypatch.setattr("app.api.stt._call_minimax_stt", _fake)


# ---------- 用例 ----------

URL = "/stt/upload"


def test_missing_token_returns_401(client):
    """缺 Authorization 头 → 401。"""
    r = client.post(URL, files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")})
    assert r.status_code == 401


def test_invalid_bearer_returns_401(client):
    """无效 token → 401。"""
    r = client.post(
        URL,
        headers={"Authorization": "Bearer not-a-real-jwt"},
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
    )
    assert r.status_code == 401


def test_invalid_mime_returns_400(client, auth_header, monkeypatch):
    """非白名单 MIME(text/plain 等)→ 400 invalid_type。"""
    _stub_minimax_stt(monkeypatch, "should not reach")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert r.json()["msg"] == "invalid_type"


def test_empty_file_returns_400(client, auth_header, monkeypatch):
    """0 字节文件 → 400 empty_file(不要让 STT 自己 422)。"""
    _stub_minimax_stt(monkeypatch, "should not reach")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400
    assert r.json()["msg"] == "empty_file"


def test_oversize_returns_413(client, auth_header, monkeypatch):
    """超过 STT_MAX_UPLOAD_MB → 413 too_large。"""
    _stub_minimax_stt(monkeypatch, "should not reach")
    # 51MB,默认上限 50MB
    big = b"\x00" * (51 * 1024 * 1024)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("big.wav", big, "audio/wav")},
    )
    assert r.status_code == 413
    assert r.json()["msg"] == "too_large"


def test_happy_path_returns_text(client, auth_header, monkeypatch):
    """合法 wav + monkeypatch 上游 → 200,data.text 等于 mock 返回。"""
    _stub_minimax_stt(monkeypatch, "我最近咳嗽有三天了")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("rec.wav", _MOCK_AUDIO, "audio/wav")},
        data={"language": "zh"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["code"] == 200
    assert r.json()["data"]["text"] == "我最近咳嗽有三天了"


def test_response_format_and_model_forwarded(client, auth_header, monkeypatch):
    """MiniMax STT 必传 model,可选 response_format=json;language 不再透传(不在上游 spec)。

    注意:本端点接收的 language Form 字段保留为兼容性参数(API 不破坏),但内部
    _call_minimax_stt 不再向上游发送,避免 MiniMax 400。
    """
    captured = {}

    async def _capture(*, raw, filename, content_type):
        captured["called"] = True
        captured["filename"] = filename
        captured["content_type"] = content_type
        return "测试"

    monkeypatch.setattr("app.api.stt._call_minimax_stt", _capture)

    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
    )
    assert r.status_code == 200
    assert captured["called"] is True
    assert captured["filename"] == "a.wav"
    assert captured["content_type"] == "audio/wav"


def test_upstream_data_has_model_and_response_format(monkeypatch):
    """直接调 _call_minimax_stt 验证发到 MiniMax 的 multipart data 不含 language、含 model + response_format=json。"""
    from app.api.stt import _call_minimax_stt

    captured = {}

    async def _fake_post(self, url, **kwargs):
        class _Resp:
            status_code = 200
            def json(self_inner):
                return {"text": "ok"}
            text = '{"text":"ok"}'
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    text = asyncio.run(_call_minimax_stt(
        raw=b"abc", filename="a.wav", content_type="audio/wav",
    ))
    assert text == "ok"
    files = captured["kwargs"]["files"]
    assert "file" in files and files["file"][0] == "a.wav"
    data = captured["kwargs"]["data"]
    assert data.get("model") == os.environ.get("MINIMAX_STT_MODEL", "")
    assert data.get("response_format") == "json"
    assert "language" not in data  # 上游 spec 不接受,不能发


def test_session_id_is_logged_only(client, auth_header, monkeypatch):
    """session_id 传进来仅记日志,不参与业务(无 DB 写入 / 无 SSE / 无 ChatSession 创建)。"""
    _stub_minimax_stt(monkeypatch, "测试文本")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
        data={"session_id": "fake-sid-12345"},
    )
    assert r.status_code == 200
    # 仅校验 status:若有 DB 写入会触发 SQLite 在 module scope engine 上的兼容问题,
    # 反而容易把简单测试搞复杂


@pytest.mark.parametrize(
    "mime",
    [
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/x-m4a",
        "audio/m4a",
        "audio/aac",
    ],
)
def test_audio_mime_whitelist_passes(client, auth_header, monkeypatch, mime):
    """白名单内每个 MIME 都能走到上游(uniapp 默认 aac 也覆盖)。"""
    _stub_minimax_stt(monkeypatch, "ok")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.audio", _MOCK_AUDIO, mime)},
    )
    assert r.status_code == 200, f"{mime}: {r.text}"


def test_upstream_httpx_error_returns_502(client, auth_header, monkeypatch):
    """MiniMax 网络错误 → 502 upstream_error: <ClassName>。"""
    _stub_minimax_stt(monkeypatch, httpx.ConnectError("connection refused"))
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
    )
    assert r.status_code == 502
    assert "upstream_error" in r.json()["msg"]
    assert "ConnectError" in r.json()["msg"]


def test_upstream_runtime_error_returns_502(client, auth_header, monkeypatch):
    """上游 RuntimeError(配置缺失 / 响应格式异常)→ 502 upstream_error。"""
    _stub_minimax_stt(monkeypatch, RuntimeError("model missing"))
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
    )
    assert r.status_code == 502
    assert "upstream_error" in r.json()["msg"]
    assert "model missing" in r.json()["msg"]


def test_upstream_returns_empty_text_returns_502(client, auth_header, monkeypatch):
    """上游响应里 text 为空字符串(实际 _call_minimax_stt 会 raise)→ 502 upstream_unrecognized。

    mock 直接抛 RuntimeError 模拟 _call_minimax_stt 内部的"text 为空"拦截分支。
    """
    _stub_minimax_stt(monkeypatch, RuntimeError("upstream_unrecognized: empty text field"))
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
    )
    assert r.status_code == 502
    assert "upstream_unrecognized" in r.json()["msg"]


def test_missing_api_key_returns_502(client, auth_header, monkeypatch):
    """MINIMAX_API_KEY 未配置 → 502 upstream_error(路由先于 mock 拦截)。"""
    # 不 stub:让路由检测到 _STT_API_KEY 为空,提前 502
    monkeypatch.setattr("app.api.stt._STT_API_KEY", "")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
    )
    assert r.status_code == 502
    assert "MINIMAX_API_KEY" in r.json()["msg"]


def test_missing_model_returns_502(client, auth_header, monkeypatch):
    """MINIMAX_STT_MODEL 未配置 → 502 upstream_error。"""
    monkeypatch.setattr("app.api.stt._STT_MODEL", "")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
    )
    assert r.status_code == 502
    assert "MINIMAX_STT_MODEL" in r.json()["msg"]


def test_filename_default_fallback_in_logic(client, auth_header, monkeypatch):
    """filename 兜底逻辑通过直接调路由函数验证(绕开 TestClient 的 multipart 限制)。

    httpx 不接受 2-tuple(无 filename)multipart,UploadFile.content_type 没有 setter,
    TestClient 路径无法触发空 filename 分支。改用 PropertyMock + 手工 UploadFile 验证
    ``audio.filename or fallback`` 拼接逻辑。
    """
    from unittest.mock import PropertyMock
    from fastapi import UploadFile
    from io import BytesIO
    import asyncio

    captured = {}

    async def _capture(*, raw, filename, content_type):
        captured["filename"] = filename
        return "ok"

    monkeypatch.setattr("app.api.stt._call_minimax_stt", _capture)

    # 手工构造 filename="" 的 UploadFile + PropertyMock 注入 content_type
    uf = UploadFile(filename="", file=BytesIO(_MOCK_AUDIO))
    monkeypatch.setattr(type(uf), "content_type", PropertyMock(return_value="audio/aac"))

    from app.api.stt import upload_stt
    asyncio.run(upload_stt(audio=uf, language="zh", session_id=None, _userid="test-user-stt"))
    assert captured["filename"] == "audio.aac"


def test_chat_path_unaffected_by_stt_call(client, auth_header, monkeypatch, db_session):
    """STT 路由不写 chat_messages,即便传 session_id 也不创建 ChatSession。

    验证 STT 是纯预处理层,与 chat 完全解耦。
    """
    from app.model.chat_log import ChatMessage, ChatSession

    _stub_minimax_stt(monkeypatch, "测试文本")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"audio": ("a.wav", _MOCK_AUDIO, "audio/wav")},
        data={"session_id": "should-not-persist"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["text"] == "测试文本"

    # STT 不应创建任何 ChatSession / ChatMessage
    assert db_session.query(ChatSession).count() == 0
    assert db_session.query(ChatMessage).count() == 0