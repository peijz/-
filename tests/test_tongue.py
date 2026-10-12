"""POST /tongue/analyze 的测试用例。

运行:  pytest tests/test_tongue.py -v

策略:
  - DB 用内存 SQLite + StaticPool(conftest.py 已设 DB_URL=sqlite:///:memory:)
  - 通过 monkeypatch ``_call_minimax_vl`` 模拟 MiniMax 视觉上游,完全离线可跑
  - 1x1 透明 PNG 字节硬编码进 fixture,不写盘
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Iterator

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# 1x1 透明 PNG(标准 base64 → 解码后 67 字节)
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


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

    # SQLite 不会给 BigInteger PK 自增（只对 INTEGER PRIMARY KEY 生效），
    # 而生产 MySQL 由 AUTO_INCREMENT 处理。给 ChatMessage 加个 before_insert，
    # 让测试里的 add_message 也能跑通，不污染生产逻辑。
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
        # engine 是 module scope（StaticPool in-memory 跨测试共享），test 之间必须清表，
        # 否则前面测试写入的 ChatSession/ChatMessage 会污染后续断言。
        from app.model.chat_log import ChatMessage, ChatSession
        s.query(ChatMessage).delete()
        s.query(ChatSession).delete()
        s.commit()
        s.close()


@pytest.fixture
def client(monkeypatch, db_session):
    """TestClient + monkeypatch 所有 SessionLocal 引用。"""
    from app.model import db as db_mod
    from app.model import chat_log
    from app.api import auth, chat
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(auth, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(chat, "SessionLocal", lambda: db_session)
    # chat_log.py 里 `from app.model.db import SessionLocal` 在模块加载时就
    # 把名字绑到 chat_log 自己命名空间，只 patch db_mod 不够。
    monkeypatch.setattr(chat_log, "SessionLocal", lambda: db_session)

    from app.api.app import app
    return TestClient(app)


@pytest.fixture
def auth_header():
    """签 access_token,模拟已登录。"""
    from app.api.auth import JWT_ALGORITHM, JWT_SECRET

    def _make(userid: str = "test-user-tongue") -> dict[str, str]:
        now = int(time.time())
        token = jwt.encode(
            {"sub": userid, "type": "access", "role": "user",
             "iat": now, "exp": now + 3600},
            JWT_SECRET, algorithm=JWT_ALGORITHM,
        )
        return {"Authorization": f"Bearer {token}"}

    return _make


# ---------- 默认 LLM 返回:合规报告 ----------

_GOOD_REPORT = {
    "tongue_features": {
        "tongue_color": "淡红",
        "coating_color": "白苔",
        "coating_thickness": "薄苔",
        "tongue_shape": ["齿痕"],
        "moisture": "湿润",
        "sublingual_vessels": "正常",
        "spots": [],
        "overall_impression": "气虚",
    },
    "tcm_pattern": {
        "constitution": "气虚",
        "pathogenesis": "脾气不足,运化失司",
        "cold_heat": "平",
        "deficiency_excess": "虚",
        "organ_focus": ["脾", "胃"],
    },
    "regimen": {
        "diet": ["忌生冷", "山药粥"],
        "lifestyle": ["23 点前入睡"],
        "exercise": ["八段锦"],
        "acupoints": ["足三里", "中脘"],
        "tea": ["党参陈皮茶"],
    },
    "classical_formula_hints": ["四君子汤"],
    "summary": "患者舌象以脾气虚为主,建议健脾益气。",
    "confidence_notes": "舌下络脉因角度受限未看清。",
}


# ---------- 用例 ----------

URL = "/tongue/analyze"


def _stub_minimax_vl(monkeypatch, payload):
    """monkeypatch app.api.tongue._call_minimax_vl 为给定返回。"""
    async def _fake(_data_url: str, _symptoms: str) -> str:
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)
    monkeypatch.setattr("app.api.tongue._call_minimax_vl", _fake)


@pytest.fixture(scope="module", autouse=True)
def sandbox_long_memory():
    """把 zygchat 的 long memory 切到独立测试索引,避免污染 dev 数据。

    原因: ``test_save_tongue_report_to_memory_writes_and_dedupes`` 等用例直接调
    ``zygchat.save_tongue_report_to_memory``,会走真实 Redis + dev 索引
    ``agent_long_memory_v2``。每次跑都往里塞 ``tonguemem*`` / ``tongueskip*``
    前缀的事实,长期累积 100+ 条野生数据,污染 dev 长记忆召回。

    实现: module-scope autouse,覆盖本模块全部用例。
      1. 用同样 schema 建 ``agent_long_memory_v2_test`` 索引
      2. monkeypatch ``zygchat.INDEX_NAME`` 和 ``zygchat.index`` 单例,让
         ``add_text`` / ``find_similar_with_score`` / ``similarity_search`` /
         ``delete_all_for_user`` 全部走测试索引
      3. setup 时 clear 一次(清掉上次中断残留),teardown 时 drop 整个索引
    """
    from _pytest.monkeypatch import MonkeyPatch
    from redis import exceptions as redis_exc
    from redisvl.index import SearchIndex
    from redisvl.schema import IndexSchema

    from app.agent import zygchat

    TEST_INDEX = "agent_long_memory_v2_test"
    # ⚠ prefix 必须与 dev 索引不同:
    #   redisvl SearchIndex 用 prefix 给 doc key 拼前缀 (mem:<uuid>);
    #   两个索引用同一个 prefix 会共享同一批 hash 实体,search 谁都能查到对方的数据。
    #   dev 用 ``mem``,测试用 ``mem_test``,物理隔离。
    schema = IndexSchema.from_dict({
        "index": {"name": TEST_INDEX, "prefix": "mem_test"},
        "fields": [
            {"name": "user_id", "type": "tag"},
            {"name": "fact", "type": "text"},
            {"name": "embedding", "type": "vector",
             "attrs": {"algorithm": "flat", "dims": 1536,
                       "distance_metric": "cosine", "datatype": "float32"}},
        ],
    })
    test_index = SearchIndex(schema, zygchat.client)
    try:
        test_index.create()
    except redis_exc.ResponseError as e:
        if "Index already exists" not in str(e):
            raise
    # 清掉上次中断的残留数据(索引结构不变)
    try:
        test_index.clear()
    except Exception:
        pass

    mp = MonkeyPatch()
    mp.setattr(zygchat, "INDEX_NAME", TEST_INDEX)
    mp.setattr(zygchat, "index", test_index)

    yield TEST_INDEX

    # delete(drop=True) 会 FT.DROPINDEX + 把 prefix 下的 hash key 全 unlink,
    # 下次再跑测试也不会有上次残留。注意别用 ``drop``——SearchIndex 没有该方法,
    # AttributeError 会被 except 吞,索引+keys 全部留着。
    try:
        test_index.delete(drop=True)
    except Exception:
        pass
    mp.undo()


def test_missing_token_returns_401(client):
    """缺 Authorization 头 → 401。"""
    r = client.post(URL, files={"image": ("t.png", _PNG_BYTES, "image/png")})
    assert r.status_code == 401


def test_invalid_bearer_returns_401(client):
    """无效 token → 401。"""
    r = client.post(
        URL,
        headers={"Authorization": "Bearer not-a-real-jwt"},
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 401


def test_invalid_mime_returns_400(client, auth_header, monkeypatch):
    """非 jpg/png → 400 invalid_type。"""
    _stub_minimax_vl(monkeypatch, _GOOD_REPORT)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert r.json()["msg"] == "invalid_type"


def test_oversize_returns_413(client, auth_header, monkeypatch):
    """超过 10MB → 413 too_large(用 11MB bytes 探顶)。"""
    _stub_minimax_vl(monkeypatch, _GOOD_REPORT)
    big = b"\x00" * (11 * 1024 * 1024)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("big.png", big, "image/png")},
    )
    assert r.status_code == 413
    assert r.json()["msg"] == "too_large"


def test_happy_path_returns_structured_report(client, auth_header, monkeypatch):
    """1KB 有效 PNG + user_symptoms → 200,响应字段全在 schema 内。"""
    _stub_minimax_vl(monkeypatch, _GOOD_REPORT)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
        data={"user_symptoms": "易疲劳,怕冷"},
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    # 顶层字段都在
    assert set(data.keys()) >= {
        "tongue_features", "tcm_pattern", "regimen",
        "classical_formula_hints", "summary", "confidence_notes",
    }
    # Literal 取值校验
    assert data["tongue_features"]["tongue_color"] == "淡红"
    assert data["tongue_features"]["coating_thickness"] == "薄苔"
    assert data["tongue_features"]["tongue_shape"] == ["齿痕"]
    assert data["tcm_pattern"]["deficiency_excess"] == "虚"


def test_markdown_fence_is_stripped(client, auth_header, monkeypatch):
    """LLM 返回 ```json ... ``` 围栏,endpoint 应能剥掉并解析。"""
    fenced = "```json\n" + json.dumps(_GOOD_REPORT, ensure_ascii=False) + "\n```"
    _stub_minimax_vl(monkeypatch, fenced)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["data"]["summary"] == _GOOD_REPORT["summary"]


def test_chat_model_preamble_is_extracted(client, auth_header, monkeypatch):
    """LLM 在 JSON 前后加了解释/前缀文字(MiniMax-M3 常见),仍应解析成功。

    模拟输出: ``好的,这是分析: {...完整 JSON...} 如有疑问请咨询医生。``
    """
    pure = json.dumps(_GOOD_REPORT, ensure_ascii=False)
    dirty = f"好的,根据您上传的舌象,我分析如下:\n{pure}\n如有疑问请咨询专业中医师。"
    _stub_minimax_vl(monkeypatch, dirty)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["data"]["summary"] == _GOOD_REPORT["summary"]


def test_unknown_field_value_is_accepted(client, auth_header, monkeypatch):
    """LLM 按 system prompt 把看不出的字段填 ``"无法判断"``,应通过校验。

    复现线上 422:sublingual_vessels 写真实看不清时 LLM 会输出 "无法判断",
    schema 之前只接受 4 个候选值 → 全军覆没。
    """
    report = json.loads(json.dumps(_GOOD_REPORT))  # deep copy
    report["tongue_features"]["sublingual_vessels"] = "无法判断"
    report["confidence_notes"] = "舌下络脉因角度受限未看清。"
    _stub_minimax_vl(monkeypatch, report)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["data"]["tongue_features"]["sublingual_vessels"] == "无法判断"


def test_truncated_llm_output_returns_502(client, auth_header, monkeypatch):
    """LLM 输出 JSON 写到一半被截断(常见:max_tokens 撞墙),应返回 502 而非 422。

    422 会让前端以为是用户输入问题,实际是上游服务问题。
    错误信息应明确提示"重新上传"——客户端容易一键retry。
    """
    truncated = json.dumps(_GOOD_REPORT, ensure_ascii=False)[:120]  # 中间被截断
    _stub_minimax_vl(monkeypatch, truncated)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 502, r.text
    assert "upstream_truncated" in r.json()["msg"]


def test_invalid_json_returns_422(client, auth_header, monkeypatch):
    """LLM 返回非 JSON 文本 → 422 invalid_json。"""
    _stub_minimax_vl(monkeypatch, "抱歉,我无法判断,请提供更清晰的图片")
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 422
    assert "invalid_json" in r.json()["msg"]


def test_schema_violation_returns_422(client, auth_header, monkeypatch):
    """LLM 返回 JSON 但 tongue_color 越界 → 422 invalid_json。"""
    bad = dict(_GOOD_REPORT)
    bad["tongue_features"] = dict(_GOOD_REPORT["tongue_features"])
    bad["tongue_features"]["tongue_color"] = "蔚蓝"  # 不在 Literal 里
    _stub_minimax_vl(monkeypatch, bad)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 422


def test_upstream_httpx_error_returns_502(client, auth_header, monkeypatch):
    """MiniMax 网络错误 → 502 upstream_error。"""
    _stub_minimax_vl(monkeypatch, httpx.ConnectError("connection refused"))
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 502
    assert "upstream_error" in r.json()["msg"]


def test_no_symptoms_still_works(client, auth_header, monkeypatch):
    """user_symptoms 缺省时也能跑通(不传 data 也行)。"""
    _stub_minimax_vl(monkeypatch, _GOOD_REPORT)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 200


def test_happy_path_persists_session_and_two_messages(client, auth_header, monkeypatch, db_session):
    """成功路径落库: 新建 1 个 ChatSession + 2 条 ChatMessage(user 主诉 / assistant JSON 报告)。"""
    from app.model.chat_log import ChatMessage, ChatSession

    _stub_minimax_vl(monkeypatch, _GOOD_REPORT)
    symptoms = "最近口苦"
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
        data={"user_symptoms": symptoms},
    )
    assert r.status_code == 200

    sessions = db_session.query(ChatSession).all()
    assert len(sessions) == 1
    assert sessions[0].user_id == "test-user-tongue"
    assert sessions[0].system_prompt is None

    msgs = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == sessions[0].id)
        .order_by(ChatMessage.id.asc())
        .all()
    )
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[0].content == symptoms
    assert msgs[1].role == "assistant"
    report_payload = json.loads(msgs[1].content)
    assert report_payload["type"] == "tongue_report"
    assert report_payload["text"] == _GOOD_REPORT["summary"]
    assert report_payload["report"]["tongue_features"]["tongue_color"] == "淡红"


def test_failure_path_writes_nothing(client, auth_header, monkeypatch, db_session):
    """失败路径(LLM 502)不写库: 不留孤儿 session / message。"""
    from app.model.chat_log import ChatMessage, ChatSession

    _stub_minimax_vl(monkeypatch, httpx.ConnectError("connection refused"))
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
        data={"user_symptoms": "x"},
    )
    assert r.status_code == 502
    assert db_session.query(ChatSession).count() == 0
    assert db_session.query(ChatMessage).count() == 0


def test_tongue_report_visible_text_in_chat_messages(client, auth_header, monkeypatch):
    """舌象报告的 assistant 消息通过 GET /chat/messages 拉出来时,
    PushContent 应是 summary 而不是截断的 JSON。"""
    _stub_minimax_vl(monkeypatch, _GOOD_REPORT)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
        data={"user_symptoms": "最近口苦"},
    )
    assert r.status_code == 200

    r2 = client.post(
        "/chat/messages",
        headers=auth_header(),
        json={"userid": "test-user-tongue"},
    )
    assert r2.status_code == 200
    items = r2.json()["data"]["items"]
    asst = next(i for i in items if i["FromUserName"] == "ai_zhongyi")
    # Content 仍是完整 JSON(前端按 type 分发),但 PushContent 必须是 summary 文本
    parsed = json.loads(asst["Content"])
    assert parsed["type"] == "tongue_report"
    assert asst["PushContent"] == _GOOD_REPORT["summary"]


def test_happy_path_schedules_tongue_long_memory(client, auth_header, monkeypatch):
    """成功路径应在 chat_messages 落库之后,异步触发舌象长期记忆入库。"""
    import threading as _threading
    from app.agent import zygchat

    captured: dict = {}
    event = _threading.Event()

    def spy(user_id, report_dict):
        captured["user_id"] = user_id
        captured["report"] = report_dict
        event.set()

    monkeypatch.setattr(zygchat, "save_tongue_report_to_memory", spy)

    _stub_minimax_vl(monkeypatch, _GOOD_REPORT)
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
        data={"user_symptoms": "最近口苦"},
    )
    assert r.status_code == 200
    # daemon 线程,等它跑完
    assert event.wait(timeout=2.0), "舌象长期记忆 spy 未在 2s 内被调用"

    assert captured["user_id"] == "test-user-tongue"
    assert captured["report"]["tcm_pattern"]["constitution"] == "气虚"
    assert captured["report"]["tongue_features"]["tongue_color"] == "淡红"


def test_failure_path_does_not_schedule_tongue_long_memory(client, auth_header, monkeypatch):
    """LLM 502 路径不应触发长期记忆入库(避免给失败请求写入脏数据)。"""
    import threading as _threading
    from app.agent import zygchat

    call_count = {"n": 0}
    event = _threading.Event()

    def spy(user_id, report_dict):
        call_count["n"] += 1
        event.set()

    monkeypatch.setattr(zygchat, "save_tongue_report_to_memory", spy)

    _stub_minimax_vl(monkeypatch, httpx.ConnectError("connection refused"))
    r = client.post(
        URL,
        headers=auth_header(),
        files={"image": ("t.png", _PNG_BYTES, "image/png")},
        data={"user_symptoms": "x"},
    )
    assert r.status_code == 502
    # 给 daemon 线程一点时间,如果它错误地被调用了,会被 event 抓到
    event.wait(timeout=0.5)
    assert call_count["n"] == 0


def test_save_tongue_report_to_memory_writes_and_dedupes(monkeypatch):
    """直接调 zygchat.save_tongue_report_to_memory:首次按字段拆 5-8 条事实,
    KNN 去重后实际写入 5+ 条;二次写入相同报告全部被去重。
    注意:本测试连真实 Redis,会污染开发库的该 user 记忆,故用一次性 user_id。"""
    from app.agent import zygchat

    # 用一次性 user_id 避免污染任何真实用户。RediSearch TAG 字段对带连字符的
    # 值在某些语法位置会报 "Syntax error near <hyphen>"，所以保持纯 hex/字母。
    test_uid = "tonguemem" + uuid.uuid4().hex[:8]
    # 用语义差异较大的取值,确保拆出的事实 embedding 互不冲突,不被 KNN 误去重
    report = {
        "tongue_features": {
            "tongue_color": "红",
            "coating_color": "黄苔",
            "coating_thickness": "厚苔",
            "moisture": "燥裂",
            "tongue_shape": ["裂纹", "点刺"],
            "sublingual_vessels": "迂曲",
            "overall_impression": "湿热",
        },
        "tcm_pattern": {"constitution": "阴虚"},
    }

    # 第一次:8 条(sublingual_vessels 也是有效值,全写入)。
    # KNN 阈值 0.08 下,字段之间 embedding 距离可能 ≥ 阈值,部分事实可能被合并,
    # 故用下限断言:确保主要维度都落库。
    saved1 = zygchat.save_tongue_report_to_memory(test_uid, report)
    assert saved1 >= 5, f"期望至少 5 条独立事实,实际 {saved1}"

    # 第二次(完全相同输入):全部被去重,saved=0
    saved2 = zygchat.save_tongue_report_to_memory(test_uid, report)
    assert saved2 == 0

    # KNN 召回应当看到所有有意义的字段
    docs = zygchat.long_memory_store.similarity_search(
        query="舌象", filter={"user_id": test_uid}, k=20,
    )
    facts = {d["page_content"] for d in docs}
    assert any("辨证" in f and "阴虚" in f for f in facts)
    assert any("舌色" in f and "红" in f for f in facts)
    assert any("苔色" in f and "黄苔" in f for f in facts)


def test_delete_all_for_user_clears_all_pages(monkeypatch):
    """回归 delete_all_for_user 的"单次 FT.SEARCH 只回一页"缺陷。

    一次 ``FT.SEARCH`` 带 ``@user_id:{tag}`` 即便 ``LIMIT 0 10000`` 也只命中一页;
    修复后必须循环到 ``raw[0]==0`` 才返回。

    连真实 Redis,使用一次性 user_id + sandbox index 隔离(sandbox_memory fixture)。
    """
    from app.agent import zygchat

    test_uid = "deltest" + uuid.uuid4().hex[:12]

    # 一次性写 50 条同 user_id,确保单次 FT.SEARCH 命中 >1 页
    for i in range(50):
        zygchat.long_memory_store.add_text(f"待删事实{i}", {"user_id": test_uid})

    # 修复前: 单次返回 < 50, 剩余 > 0
    # 修复后: 必须全部清空,返回 50
    deleted = zygchat.long_memory_store.delete_all_for_user(test_uid)
    assert deleted == 50, f"期望清空 50 条,实际只删 {deleted}"

    # 二次调用应立即返回 0(无残留)
    assert zygchat.long_memory_store.delete_all_for_user(test_uid) == 0


def test_save_tongue_report_skips_unknown_values(monkeypatch):
    """字段值为 ``"无法判断"`` 或缺失时应跳过,不写入无意义事实。"""
    from app.agent import zygchat

    test_uid = "tongueskip" + uuid.uuid4().hex[:8]
    report = {
        "tongue_features": {
            "tongue_color": "无法判断",
            "coating_color": "",  # 空字符串也跳过
            "moisture": "无法判断",
            "tongue_shape": ["无法判断"],  # list 中含"无法判断"应过滤
            # sublingual_vessels 缺失 → 跳过
        },
        "tcm_pattern": {"constitution": "气虚"},
    }
    saved = zygchat.save_tongue_report_to_memory(test_uid, report)
    # 应当只存 constitution 一条
    assert saved == 1

    docs = zygchat.long_memory_store.similarity_search(
        query="舌象", filter={"user_id": test_uid}, k=20,
    )
    facts = {d["page_content"] for d in docs}
    assert "舌象辨证: 气虚" in facts
    # "无法判断"应被过滤,不进档案
    assert not any("无法判断" in f for f in facts)
