"""MiniMaxChatModel._astream 边界条件回归测试。

修复前：当 MiniMax 把鉴权/参数/限流错误包成 ``HTTP 200 + choices=[] +
非零 base_resp.status_code``，或 SSE 流全部是 ``reasoning_content`` / 空 delta
时，``_astream`` 一个 chunk 都不 yield，langchain-core 会抛
``ValueError: No generations found in stream.`` —— 根因（上游报错）被掩盖。

修复后：``_astream`` 自身必须抛清晰的 ``RuntimeError``，让 SSE 链路把它转成
``{"event": "error", ...}`` 返回给前端。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

import pytest


# ---------- httpx stub ----------


class _FakeResponse:
    """httpx.Response 的极简替身：只暴露 _astream 用到的 status_code / aiter_lines。"""

    def __init__(self, status_code: int, lines: List[str]) -> None:
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


def _patch_stream(monkeypatch, response: _FakeResponse) -> None:
    """让 ``httpx.AsyncClient().stream(...)`` 走我们的 fake response。

    httpx 的 stream 是 ``@asynccontextmanager`` 装饰的方法，本替身同样形态。
    """
    import httpx

    @asynccontextmanager
    async def _fake_stream(self, method, url, **kwargs):
        yield response

    monkeypatch.setattr(httpx.AsyncClient, "stream", _fake_stream)


# ---------- helpers ----------


def _make_model():
    """绕开模块级单例（会被环境变量污染），手工建一个测试用的 model。"""
    from app.agent.zygchat import MiniMaxChatModel

    return MiniMaxChatModel(api_key="x", base_url="https://test.local")


async def _drain(agen):
    """把 async generator 全部消费完。"""
    async for _ in agen:
        pass


def _extract_text(content) -> str:
    """AIMessageChunk.content 可能是 str / list[dict] / None，统一拍平。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else ""
            for b in content
        )
    return ""


# ---------- 用例 ----------


def test_astream_raises_when_base_resp_error_wrapped_in_200(monkeypatch):
    """上游 HTTP 200 + ``base_resp.status_code != 0`` + ``choices=[]``：
    必须立刻抛 RuntimeError，不让 langchain-core 兜底报 "No generations found in stream."。"""
    from langchain.messages import HumanMessage

    resp = _FakeResponse(
        200,
        [
            'data: {"choices":[],"base_resp":{"status_code":2013,"status_msg":"invalid params"}}',
            "data: [DONE]",
        ],
    )
    _patch_stream(monkeypatch, resp)

    model = _make_model()
    with pytest.raises(RuntimeError, match="MiniMax stream API error"):
        asyncio.run(_drain(model._astream([HumanMessage(content="hi")])))


def test_astream_raises_when_only_reasoning_content_emitted(monkeypatch):
    """上游只发 ``reasoning_content``（不暴露给用户但占满流）后断开：
    兜底必须抛 RuntimeError，而不是无声返回 0 chunk。"""
    from langchain.messages import HumanMessage

    resp = _FakeResponse(
        200,
        [
            'data: {"choices":[{"delta":{"reasoning_content":"让我想想"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"再想想"},"finish_reason":null}]}',
            "data: [DONE]",
        ],
    )
    _patch_stream(monkeypatch, resp)

    model = _make_model()
    with pytest.raises(RuntimeError, match="no yieldable chunks"):
        asyncio.run(_drain(model._astream([HumanMessage(content="hi")])))


def test_astream_yields_chunks_normally_on_healthy_stream(monkeypatch):
    """正常路径回归：content + finish_reason stop 两类 chunk 都能正确 yield，
    修复不应破坏 happy path。"""
    from langchain.messages import HumanMessage

    resp = _FakeResponse(
        200,
        [
            'data: {"choices":[{"delta":{"content":"你"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"好"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ],
    )
    _patch_stream(monkeypatch, resp)

    model = _make_model()

    async def _collect():
        chunks = []
        async for c in model._astream([HumanMessage(content="hi")]):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(_collect())
    full = "".join(_extract_text(c.message.content) for c in chunks)
    assert "你好" in full
    # 最后一块应当带 finish_reason
    last_finish = chunks[-1].generation_info.get("finish_reason")
    assert last_finish == "stop"


def test_astream_raises_when_response_is_http_400(monkeypatch):
    """HTTP 4xx 时应当冒 RuntimeError("MiniMax HTTP 400: ...")，
    而不是被 generic httpx 异常吞掉（已有逻辑保留，验证未退化）。"""
    from langchain.messages import HumanMessage

    class _ResponseWithBody(_FakeResponse):
        async def aread(self) -> bytes:
            return b'{"base_resp":{"status_code":1004,"status_msg":"auth failed"}}'

    _patch_stream(monkeypatch, _ResponseWithBody(400, []))

    model = _make_model()
    with pytest.raises(RuntimeError, match="MiniMax HTTP 400"):
        asyncio.run(_drain(model._astream([HumanMessage(content="hi")])))
