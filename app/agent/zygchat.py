from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# 必须在 import app.model 之前加载 .env，否则 db.py 读不到 DB_URL
from dotenv import load_dotenv

load_dotenv()

import json
import logging
import os
import re
import threading
import uuid
import httpx
import requests
import numpy as np
from typing import Annotated, Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain.tools import tool
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.query import VectorQuery
from redisvl.exceptions import RedisSearchError
from langchain.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

from app.model.db import init_db
from app.model import chat_log as db_log

# 1.1 初始化数据库表（fail-fast：DB 不可用就在 stream 之前崩）
init_db()

# 模块 logger（提前定义，给 checkpointer 初始化等启动期日志使用）
logger = logging.getLogger("app.agent")


# =====================================================================
# 1. 模型初始化（LLM 客户端单例）
#    - 模型选型：MiniMax-M3，温度 0.6
#    - 走 MiniMax 原生 chat 端点 /text/chatcompletion_v2（不走 OpenAI 兼容网关）
#    - 网关 /v1/chat/completions 对当前 key 返 401（1004），原生端点 200
#    - 与 embedding 走原生一致（避开网关的格式转换坑）
# =====================================================================

class MiniMaxChatModel(BaseChatModel):
    """MiniMax 原生 chat 端点（/text/chatcompletion_v2）的 LangChain 包装。

    与 ChatOpenAI(base_url=...) 的关键差异：
      - 网关 /v1/chat/completions 对当前 key 返 1004 鉴权失败；原生 200。
      - 流式协议：SSE data: {json}\\n\\n
          - 中间 chunk：delta.content 是用户可见文本；delta.reasoning_content 是
            thinking，本类不暴露（否则 SSE 会泄露思路给前端）。
          - 末 chunk：含 message（全文 + tool_calls）和 finish_reason。
      - tool_calls 原生协议与 OpenAI 兼容（list[{"id","type","function":{"name","arguments"}}]），
        bind_tools 注入的格式直接复用。
    """

    model_name: str = "MiniMax-M3"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.6
    timeout: float = 60.0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "MiniMax-native"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        """绑定 tools 到模型。

        基类 ``BaseChatModel.bind_tools`` 直接 ``raise NotImplementedError``，
        这里把 tool 工具（BaseTool / @tool callable / dict）转成 OpenAI 函数
        格式（``{"type":"function","function":{...}}``），M3 原生协议接受。
        """
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """LangChain messages → MiniMax 原生格式。

        规则：
          - 文本走 str；多模态 list blocks 原样保留。
          - ToolMessage 走 ``tool`` role + ``tool_call_id``（M3 原生协议字段）。
        """
        out: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
                # 必须保留 tool_calls；否则第二轮调用 MiniMax 找不到上轮的工具调用
                # 会报 invalid params (status_code=2013)
                m_tool_calls = getattr(m, "tool_calls", None) or []
                if m_tool_calls:
                    converted_calls = []
                    for tc in m_tool_calls:
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                        tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                        if isinstance(tc_args, dict):
                            tc_args_str = json.dumps(tc_args, ensure_ascii=False)
                        else:
                            tc_args_str = tc_args or "{}"
                        converted_calls.append({
                            "id": tc_id or "",
                            "type": "function",
                            "function": {"name": tc_name or "", "arguments": tc_args_str},
                        })
                    out.append({
                        "role": "assistant",
                        "content": m.content if isinstance(m.content, str) else "",
                        "tool_calls": converted_calls,
                    })
                    continue
            elif isinstance(m, ToolMessage):
                entry: Dict[str, Any] = {
                    "role": "tool",
                    "content": m.content if isinstance(m.content, str) else str(m.content),
                }
                if getattr(m, "tool_call_id", None):
                    entry["tool_call_id"] = m.tool_call_id
                out.append(entry)
                continue
            else:
                role = "user"

            content = m.content
            if isinstance(content, str):
                out.append({"role": role, "content": content})
            elif isinstance(content, list):
                out.append({"role": role, "content": content})
            else:
                out.append({"role": role, "content": str(content)})
        return out

    def _build_payload(
        self, messages: List[BaseMessage], *, stream: bool, **kwargs: Any
    ) -> Dict[str, Any]:
        """组装 MiniMax 请求体。

        tools 是 LangChain ``bind_tools`` 注入的 OpenAI 格式 list（type=function），
        M3 原生协议直接接受，无需转换。
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "stream": stream,
        }
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")
        for k in ("max_tokens", "top_p", "stop"):
            v = kwargs.get(k)
            if v is not None:
                payload[k] = v
        return payload

    @staticmethod
    def _normalize_tool_calls(raw: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """MiniMax 原生 tool_calls → LangChain tool_calls 字段格式。

        M3: ``{"id","type","function":{"name","arguments"(JSON str)}}``
        LC : ``{"id","name","args"(dict),"type"}``
        """
        if not raw:
            return []
        out: List[Dict[str, Any]] = []
        for tc in raw:
            func = tc.get("function") or {}
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except Exception:
                args = {}
            out.append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "args": args if isinstance(args, dict) else {},
                "type": tc.get("type", "tool_call"),
            })
        return out

    def _message_from_choice(self, choice: Dict[str, Any]) -> AIMessage:
        """非流式 choice → AIMessage。"""
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        tool_calls = self._normalize_tool_calls(msg.get("tool_calls"))
        kwargs: Dict[str, Any] = {"name": msg.get("name")}
        if tool_calls:
            kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, **kwargs)

    # ---------- sync（_summarize_and_save 走 model.invoke）----------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stream=False, **kwargs)
        resp = requests.post(
            f"{self.base_url}/text/chatcompletion_v2",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"MiniMax HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        base_resp = data.get("base_resp") or {}
        if base_resp.get("status_code", 0) not in (0, None):
            raise RuntimeError(f"MiniMax API error: {base_resp}")
        choice = data["choices"][0]
        gen_info: Dict[str, Any] = {
            "finish_reason": choice.get("finish_reason"),
            "model": data.get("model"),
        }
        return ChatResult(
            generations=[ChatGeneration(message=self._message_from_choice(choice), generation_info=gen_info)]
        )

    # ---------- async（agent.astream / agent.ainvoke）----------

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stream=False, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers=self._headers(),
                json=payload,
            )
        if r.status_code >= 400:
            raise RuntimeError(f"MiniMax HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        base_resp = data.get("base_resp") or {}
        if base_resp.get("status_code", 0) not in (0, None):
            raise RuntimeError(f"MiniMax API error: {base_resp}")
        choice = data["choices"][0]
        gen_info: Dict[str, Any] = {
            "finish_reason": choice.get("finish_reason"),
            "model": data.get("model"),
        }
        return ChatResult(
            generations=[ChatGeneration(message=self._message_from_choice(choice), generation_info=gen_info)]
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """流式 chat 补全：逐 chunk 解析原生 SSE 协议，yield ChatGenerationChunk。
        """
        payload = self._build_payload(messages, stream=True, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/text/chatcompletion_v2",
                headers=self._headers(),
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise RuntimeError(
                        f"MiniMax HTTP {resp.status_code}: {body[:500].decode('utf-8', 'ignore')}"
                    )
                # 跟踪已 yield 过的 tool_call id：M3 流式协议会在 delta 和
                # 末 chunk 的 message 里各 emit 一份相同的 tool_calls，AIMessageChunk
                # 合并会得到重复列表，LangChain 多次执行同一 tool 后第二轮 MiniMax
                # 会报 "duplicate tool_call id" (2013)。
                seen_tool_ids: set = set()
                chunks_yielded = 0
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if not raw or raw.strip() == "[DONE]":
                        continue
                    try:
                        d = json.loads(raw)
                    except Exception:
                        continue
                    # MiniMax 把鉴权 / 参数 / 限流等错误包成 HTTP 200 + 空 choices +
                    # 非零 base_resp.status_code。若不在这里显式拦截，每个 SSE 行都会
                    # 被下面的 choice/content 过滤吃掉，导致 _astream 一个 chunk 都不
                    # yield，最后由 langchain-core 抛 "No generations found in stream."
                    # ——根因被掩盖。
                    base_resp = d.get("base_resp") if isinstance(d, dict) else None
                    if (
                        isinstance(base_resp, dict)
                        and base_resp.get("status_code", 0) not in (0, None)
                    ):
                        raise RuntimeError(
                            "MiniMax stream API error: "
                            f"status_code={base_resp.get('status_code')}, "
                            f"status_msg={base_resp.get('status_msg')!r}"
                        )
                    choice = (d.get("choices") or [{}])[0]
                    if not choice:
                        continue

                    delta = choice.get("delta") or {}
                    msg = choice.get("message")
                    finish_reason = choice.get("finish_reason")

                    content = delta.get("content") or ""
                    tool_calls = self._normalize_tool_calls(delta.get("tool_calls"))
                    if msg and not tool_calls:
                        tool_calls = self._normalize_tool_calls(msg.get("tool_calls"))

                    # 按 id 去重：同一 tool_call 只在第一次出现时 yield
                    if tool_calls:
                        tool_calls = [tc for tc in tool_calls if tc.get("id") and tc["id"] not in seen_tool_ids and not seen_tool_ids.add(tc["id"])]

                    if not content and not tool_calls and not finish_reason:
                        continue

                    ai_kwargs: Dict[str, Any] = {}
                    if finish_reason:
                        ai_kwargs["chunk_position"] = "last"
                    if tool_calls:
                        ai_kwargs["tool_calls"] = tool_calls

                    message = AIMessageChunk(content=content, **ai_kwargs)
                    gen_info: Optional[Dict[str, Any]] = (
                        {"finish_reason": finish_reason} if finish_reason else None
                    )
                    yield ChatGenerationChunk(message=message, generation_info=gen_info)
                    chunks_yielded += 1

                if chunks_yielded == 0:
                    # 兜底：SSE 行全部被过滤（只发了 reasoning_content / 全是空 delta /
                    # 上游提前关流）。此时不要让 langchain 抛 "No generations found in
                    # stream."，否则定位不到根因。
                    raise RuntimeError(
                        "MiniMax stream returned no yieldable chunks "
                        "(可能全部是 reasoning_content 或上游异常关闭)"
                    )


model = MiniMaxChatModel(
    model_name=os.getenv("MINIMAX_CHAT_MODEL", "MiniMax-M3"),
    api_key=os.getenv("MINIMAX_API_KEY") or "",
    base_url=os.getenv("MINIMAX_BASE_URL") or "",
    temperature=float(os.getenv("MINIMAX_TEMPERATURE", "0.6")),
)


# =====================================================================
# 2. 长期记忆模块
#    - 用 RedisVL 自建一个向量索引（替代 RedisVectorStore，规避 LangChain 0.x 旧 API）
#    - 每个用户独立隔离（TAG 字段 user_id），跨会话记忆
#    - 维度固定 1536（MiniMax embo-01 不可配置）
# =====================================================================

# MiniMax 的 OpenAI 兼容网关对 embedding 转换格式不可靠（与之前 DashScope 同样的坑），
# 直接走 MiniMax 原生 /v1/embeddings 端点，避开网关格式转换。
MINIMAX_EMBED_URL = os.getenv(
    "MINIMAX_EMBED_URL",
    "https://api.minimax.chat/v1/embeddings",
)
MINIMAX_EMBED_MODEL = os.getenv("MINIMAX_EMBED_MODEL", "embo-01")
MINIMAX_EMBED_DIMS = 1536  # embo-01 固定维度
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class MiniMaxEmbeddings:
    """MiniMax 原生 Embedding API（embo-01）的轻量封装。

    为什么不走 OpenAI 兼容网关：
      - 与 DashScope 类似，网关在 /v1/embeddings 上的格式转换不可靠，
        直接 POST 原生端点最稳。

    维度：embo-01 固定 1536 维，不可配置（与历史 DashScope text-embedding-v3 的 1024 不兼容）。

    type 区分 query / db，MiniMax 内部会用不同的检索编码，
    推荐对召回侧文本用 "query"、入库侧用 "db"，提升召回质量。
    """

    def __init__(
        self,
        model: str = MINIMAX_EMBED_MODEL,
        api_key: Optional[str] = None,
        url: str = MINIMAX_EMBED_URL,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.url = url

    def _embed(self, texts: List[str], text_type: str) -> List[List[float]]:
        """批量调用 MiniMax embedding 接口，返回二维向量列表。

        :param texts: 待编码的文本列表（一次不超过 100 条，建议）。
        :param text_type: "query" 或 "db"，影响检索侧编码。
        :raises requests.HTTPError: HTTP 非 2xx 时抛出。
        """
        resp = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "type": text_type,
                "texts": texts,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # 原生响应: {"vectors": [[...]], "base_resp": {...}}
        # 兼容网关可能包成 {"data":[{"embedding":[...]}]}，这里两种都认
        if isinstance(data, dict) and "vectors" in data:
            return data["vectors"]
        if isinstance(data, dict) and "data" in data:
            return [item["embedding"] for item in data["data"]]
        # 错误响应通常是 {"base_resp": {"status_code": xxx, "status_msg": "..."}}
        base_resp = data.get("base_resp") if isinstance(data, dict) else None
        detail = base_resp if isinstance(base_resp, dict) else data
        raise ValueError(f"MiniMax embeddings 响应格式无法识别: {detail}")

    def embed_query(self, text: str) -> List[float]:
        """编码单条 query 文本。LangChain VectorStore 召回时会调用。"""
        return self._embed([text], "query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码入库文本。LangChain VectorStore 写入时调用。"""
        return self._embed(texts, "db")


# 模块级单例：Embedding 与索引名都只创建一次，避免每次请求重新初始化
EMBEDDINGS = MiniMaxEmbeddings()
INDEX_NAME = "agent_long_memory_v2"

# 索引 schema：3 个字段
# - user_id (TAG): 用于按用户过滤，DIALECT=3 下做精确隔离
# - fact    (TEXT): 事实正文，做全文 + 向量混合召回时可被检索
# - embedding (VECTOR, 1536 维, COSINE): MiniMax embo-01 固定维度
schema = IndexSchema.from_dict({
    "index": {"name": INDEX_NAME, "prefix": "mem"},
    "fields": [
        {"name": "user_id", "type": "tag"},
        {"name": "fact", "type": "text"},
        {"name": "embedding", "type": "vector",
         "attrs": {"algorithm": "flat", "dims": MINIMAX_EMBED_DIMS, "distance_metric": "cosine", "datatype": "float32"}},
    ],
})
client = RedisConnectionFactory.get_redis_connection(REDIS_URL)
index = SearchIndex(schema, client)
# 多 worker 并发加载模块时,exists() 与 create_index() 之间存在竞态窗口,
# 两个进程都看到索引不存在就会后一个失败。直接调 create() 并吞掉 "Index already exists"。
try:
    index.create()
except RedisSearchError as e:
    if "Index already exists" not in str(e):
        raise


class RedisVLMemoryStore:
    """长期记忆向量存储的轻量封装（替代 LangChain 旧 RedisVectorStore）。

    数据模型：每个 user 一组 (fact, embedding) 文档，存到 RedisVL HashStorage。
    召回：KNN 向量相似度 + TAG 过滤（DIALECT=3 严格隔离）。

    为什么不直接用 LangChain RedisVectorStore：
      - LangChain 0.x 那套依赖 redis 包的 Redis 模块，跟当前 redisvl 的接口不兼容；
      - 改用 redisvl 直发 FT.SEARCH，可控性更强，能精确控制 DIALECT/PARAMS 等参数。
    """

    @staticmethod
    def _to_bytes(vec: List[float]) -> bytes:
        """float32 列表 → little-endian 二进制。RediSearch vector 字段要求 float32 bytes。"""
        return np.array(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _from_bytes(buf) -> List[float]:
        """二进制 → float32 列表（用于调试场景，目前未启用）。"""
        return np.frombuffer(buf, dtype=np.float32).tolist()

    @staticmethod
    def add_text(text: str, metdata: dict) -> None:
        """写入一条事实到向量索引。

        :param text: 事实正文（建议 ≤ 30 字短句，便于后续被 LLM 综合引用）。
        :param metdata: 必须包含 "user_id" 字段，作为隔离键。
        """
        emb = EMBEDDINGS.embed_query(text)
        doc = {
            "user_id": metdata["user_id"],
            "fact": text,
            "embedding": RedisVLMemoryStore._to_bytes(emb),
        }
        # SearchIndex.load 会走 HashStorage，vector 字段必须传 float32 bytes
        index.load([doc])

    @staticmethod
    def find_similar_with_score(
        text: str,
        user_id: str,
        threshold: float = 0.08,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """在该 user 范围内找与 text 向量相似的已有记忆，返回带 score 的列表。

        与 similarity_search 的区别：返回 cosine **距离**（越接近 0 越相似），
        用于 save_long_memory 去重。

        :param text: 待比对的事实文本。
        :param user_id: 业务 userid（TAG 字段严格隔离）。
        :param threshold: 距离阈值；score ≤ threshold 视为重复。0.08 ≈ 相似度 0.92。
        :param k: KNN 候选数。
        :return: [{"id", "fact", "score", "similarity"}]，已按 distance ASC 排好，
                 只保留 score ≤ threshold 的命中。
        """
        emb = EMBEDDINGS.embed_query(text)
        emb_bytes = np.array(emb, dtype=np.float32).tobytes()
        # 把 user_id 过滤写进 KNN query 本身（DIALECT=3 下 TAG 精确匹配）
        q_str = f"@user_id:{{{user_id}}}=>[KNN {k} @embedding $vec]"
        from redis.commands.search.query import Query
        q = (
            Query(q_str)
            .sort_by("__embedding_score")  # KNN 默认距离列名
            .return_fields("fact", "__embedding_score")
            .paging(0, k)
            .dialect(3)
        )
        raw = client.ft(INDEX_NAME).search(q, query_params={"vec": emb_bytes})
        out: List[Dict[str, Any]] = []
        for d in raw.docs:
            # __embedding_score 是 redis-py 解析 KNN 结果时挂到 Document 上的属性。
            # 不能直接写 d.__embedding_score —— Python name mangling 会把它改写成
            # d._RedisVLMemoryStore__embedding_score，触发 AttributeError 被 except 吃掉。
            score_attr = getattr(d, "__embedding_score", None)
            if score_attr is None:
                continue
            try:
                score = float(score_attr)
            except ValueError:
                continue
            if score > threshold:
                continue  # 不够相似，丢掉
            out.append({
                "id": d.id,
                "fact": d.fact,
                "score": score,
                "similarity": max(0.0, 1.0 - score),
            })
        return out

    @staticmethod
    def similarity_search(query: str, filter: dict, k: int = 3) -> List[Dict]:
        """按 user 隔离的 KNN 向量召回。

        :param query: 自然语言查询，会先调 embed_query 得到向量。
        :param filter: 必须包含 "user_id"；本方法会用 DIALECT=3 严格按 TAG 匹配。
        :param k: 返回 top-k 条。
        :return: [{"page_content": <fact>, "id": <doc_id>}, ...]

        实现要点（踩坑记录）：
          - 不能用 client.ft().search(query, query_params=params)，redis-py 的
            query_params kwarg 不会生成 RediSearch 的 PARAMS N name value 段。
          - 必须走 execute_command 手工拼 FT.SEARCH。
          - PARAMS 段的 <count> 是参数总个数（name+value 各算一个），不是"对数"。
          - DIALECT 3 必须显式声明在 PARAMS 段之后；否则 @user_id:{tag} 按
            DIALECT=2 宽松文本匹配，产生跨用户越权读取。
        """
        emb = EMBEDDINGS.embed_query(query)
        user_tag = filter["user_id"]
        query_obj = VectorQuery(
            vector=emb,
            vector_field_name="embedding",
            return_fields=["fact"],
            filter_expression=f"@user_id:{{{user_tag}}}",
            num_results=k,
            dialect=3,  # RediSearch 2.10+ 解析 KNN 的 AS 子句需要 DIALECT=3
        )
        params = query_obj.params  # {'vector': <bytes>}
        # redis-py packer 不收 tuple，所以 PARAMS 的 (name, value) 必须展平成平铺参数
        flat_params = []
        for name, value in params.items():
            flat_params.extend([name, value])
        raw = client.execute_command(
            "FT.SEARCH", INDEX_NAME,
            query_obj.query_string(),
            # 重要：len(params)*2，不是 len(params)；否则报 "PARAM VALUE pairs"
            "PARAMS", str(len(params) * 2), *flat_params,
            "RETURN", "1", "fact",
            "DIALECT", "3",
            "LIMIT", "0", str(k),
        )
        # raw 形如 [total, id1, [field, value, ...], id2, ...]
        # 注意：fields 子列表里 field/value 交替出现
        docs = []
        i = 1
        while i < len(raw):
            doc_id = raw[i].decode() if isinstance(raw[i], bytes) else raw[i]
            fields = raw[i + 1] if i + 1 < len(raw) else []
            fact = ""
            if isinstance(fields, list):
                for j in range(0, len(fields) - 1, 2):
                    fname = fields[j].decode() if isinstance(fields[j], bytes) else fields[j]
                    if fname == "fact":
                        fval = fields[j + 1]
                        fact = fval.decode() if isinstance(fval, bytes) else fval
                        break
            docs.append({"page_content": fact, "id": doc_id})
            i += 2
        return docs

    @staticmethod
    def delete_by_ids(ids: List[str]) -> None:
        """按 doc id 批量删除记忆；空列表直接跳过。"""
        if ids:
            client.delete(*ids)


# 模块级单例，便于各模块直接调用
long_memory_store = RedisVLMemoryStore()


# =====================================================================
# 3. 记忆 Tool：给 Agent 自主调用
#    - recall_long_memory / save_long_memory / delete_long_memory
#    - user_id 通过 RunnableConfig 注入（每次 invoke 自动带入）
# =====================================================================

def _get_user_id(config: RunnableConfig) -> str:
    """从 LangChain RunnableConfig 抽取 user_id。

    :raises ValueError: 调用方没在 configurable 里塞 user_id 时报错。
    """
    user_id = config.get("configurable", {}).get("user_id") if config else None
    if not user_id:
        raise ValueError("user_id 未注入：调用 agent 时需传 config={'configurable': {'user_id': ...}}")
    return user_id


@tool
def recall_long_memory(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """查询长期记忆，用于调取用户历史对话消息。

    :param query: 需要检索的信息描述（自然语言）。
    :return: 拼接好的事实列表（每行 `- <fact>`），无记忆时返回 "暂无相关记忆"。
    """
    user_id = _get_user_id(config)
    docs = long_memory_store.similarity_search(
        query=query,
        filter={"user_id": user_id},
        k=3,
    )
    if not docs:
        return "暂无相关记忆"

    return "\n".join([f"- {doc['page_content']}" for doc in docs])


@tool
def save_long_memory(
    fact: str,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """写入一条稳定健康信号到用户长期记忆。

    需保存(满足任一即调本工具):
      - 体质判定(如阳虚质、痰湿质、气虚)
      - 主诉与持续时长(如失眠3个月、咳嗽1周)
      - 关键症状(寒热偏好、出汗、二便、舌象、脉象)
      - 作息/饮食/情志稳定模式(长期熬夜、常饮冷、易怒)
      - 用药/调理反馈(如某方喝了3天胃痛缓解)

    不存:问候、寒暄、追问选项、临时情绪、单次偶发事件。

    :param fact: 精简事实短句（建议 ≤ 30 字），一条一个点。
    :return: 写入结果提示（含保存的原文）；若已有相似记忆则跳过并返回已有事实。

    去重策略：写入前先在该 user 范围内做 KNN 召回（cosine 距离 ≤ 0.08，
    即相似度 ≥ 0.92）视为重复，跳过保存并把已有事实反馈给 Agent。
    若 Agent 想覆盖（事实更新/细化），先调 delete_long_memory 再调本工具。
    """
    user_id = _get_user_id(config)
    clean = fact.strip()
    if not clean or len(clean) < 4:
        return "无效事实，未保存"

    similar = long_memory_store.find_similar_with_score(
        text=clean, user_id=user_id, threshold=0.08, k=5,
    )
    if similar:
        top = similar[0]
        return (
            f"已存在相似记忆（相似度 {top['similarity']:.2f}），未重复保存："
            f"{top['fact']!r}。如需覆盖请先调 delete_long_memory 再调 save_long_memory。"
        )

    long_memory_store.add_text(text=clean, metdata={"user_id": user_id})
    return f"记忆已保存: {clean}"


@tool
def delete_long_memory(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """从长期记忆中删除一条或多条已记录的事实。

    :param query: 自然语言描述要删除的事实，按相似度取 top-5 后整批删除。
    :return: 实际删除条数提示；未匹配时返回 "未匹配到可删除记忆"。
    """
    user_id = _get_user_id(config)
    docs = long_memory_store.similarity_search(query=query, filter={"user_id": user_id}, k=5)
    if not docs:
        return "未匹配到可删除记忆"
    ids = [doc["id"] for doc in docs if doc["id"]]
    long_memory_store.delete_by_ids(ids)
    return f"已删除{len(ids)}条记忆"


# Agent 可用的工具列表（create_agent 时传入）
tools = [recall_long_memory, save_long_memory, delete_long_memory]


# =====================================================================
# 4. 智能体定义（system prompt + LangChain create_agent）
#    - system prompt 是中医辨证 SOP + 合规红线 + 长期记忆规则 + 结构化追问格式
#    - checkpointer 走 AsyncRedisSaver（多 worker 共享、重启不丢）
#    - 用环境变量 APP_CHECKPOINTER 切换：redis | memory（默认 redis）
# =====================================================================

def _build_checkpointer():
    """构建 checkpointer。

    APP_CHECKPOINTER=memory → InMemorySaver（开发/单进程）
    默认 → AsyncRedisSaver（生产，多 worker 共享 state.messages）

    为什么用 AsyncRedisSaver 而不是 RedisSaver：
      - SSE 走 agent.astream（async 路径），LangGraph 会调 checkpointer 的
        aget_tuple / aput 等异步方法。同步版 RedisSaver 没实现这些方法，
        走基类直接 raise NotImplementedError（无消息），导致 SSE 整条链路
        立刻挂掉。AsyncRedisSaver 的同步 get_tuple / put 是基于
        run_coroutine_threadsafe 的包装，所以非流式 agent.invoke 也能用。
    """
    backend = os.getenv("APP_CHECKPOINTER", "redis").lower()
    if backend == "memory":
        logger.info("[checkpointer] using InMemorySaver (single-process)")
        return InMemorySaver()
    saver = AsyncRedisSaver(redis_url=REDIS_URL)
    logger.info("[checkpointer] using AsyncRedisSaver @ %s (setup deferred)", REDIS_URL)
    return saver


CHECKPOINTER = _build_checkpointer()
_CHECKPOINTER_READY = False


async def ensure_checkpointer_setup() -> None:
    """幂等初始化 AsyncRedisSaver：asetup() 必须跑在 event loop 里。

    由 FastAPI 启动钩子调用一次；后续 chat_stream / chat 不再重复。
    asetup() 内部用 create(overwrite=False) 建索引，重复调用安全。
    """
    global _CHECKPOINTER_READY
    if _CHECKPOINTER_READY:
        return
    if not isinstance(CHECKPOINTER, AsyncRedisSaver):
        _CHECKPOINTER_READY = True  # InMemorySaver 无需 setup
        return
    await CHECKPOINTER.asetup()
    _CHECKPOINTER_READY = True
    logger.info("[checkpointer] AsyncRedisSaver ready")

# Agent 与摘要共用同一个 model（顶部定义）；_summarize_and_save 通过 model.invoke 复用
# v2: 拆分精简版。1856 → ~690 token,节省 ~62%。关键约束(合规红线/6 段结构/长期记忆强制/
# 追问格式)保留;删了原版八(服务准则)和四(问诊流程,与核心能力合并)等冗余复述。
systemp_prompt = """你是"小二大夫",合规的中医 AI 助手,做中医辨证咨询与居家养生指导。**不开西药、不做西医诊断、不替代线下诊疗、不承诺疗效、不涉玄学命理**。

## 核心能力
按四诊逻辑收集主诉/症状/寒热/二便/情志/舌脉,信息不全时主动追问,不全不辨证;依四诊判定证型病机;输出辨证结论、体质辨识、饮食/作息/穴位等可落地的居家养生方案。

## 回答结构(6 段,不可乱序/省略)
1.症状体质汇总(当前+历史记忆) 2.中医辨证(证型+病机) 3.核心调理方向(治则)
4.生活方案(饮食/作息/运动/情志) 5.穴位养生(穴位+操作+频次) 6.禁忌提醒
语言通俗温和,无玄学夸大臆断。

## 长期记忆(必调,不可凭感觉跳过)
- **开场**:立即调 recall_long_memory(query=用户主诉)纳入辨证
- **新信号入库**:先调 save_long_memory 入库再作答(详则见该 tool description)
- **按需**:用户要求遗忘 → delete_long_memory

## 结构化追问(仅寒热/舌苔/便质/睡眠等可枚举四诊要素时)
末尾追加一个 [OPTIONS]...[/OPTIONS] JSON 块,例如:
[OPTIONS]
{"question": "您平时怕冷吗?", "options": ["很怕冷","有点怕冷","不怕冷","怕热"]}
[/OPTIONS]
每次 1 个追问,2~6 个选项,块外不要重复追问文本。开放性问题/辨证结论/调理方案不生成追问。
"""

# Agent 本身（带 checkpointer，支持多轮会话 state.messages 累计）
# state.messages 按 configurable.thread_id（即 session_id）隔离，跨 worker 一致。
agent = create_agent(
    model=model,
    system_prompt=systemp_prompt,
    tools=tools,
    checkpointer=CHECKPOINTER,
)

# 公开常量，便于上层（如脚本）直接拿到 prompt
SYSTEM_PROMPT: str = systemp_prompt


# =====================================================================
# 5. 内部 helpers（消息序列化、内容提取、记忆注入、摘要入库）
# =====================================================================


def _flatten_content(content) -> str:
    """把 LangChain Message.content 统一拍平成纯文本字符串。

    LangChain 的 content 可能是 str（普通文本）、list[dict]（多模态 block）、
    或 None。本函数的目标：尽量只取出 text 块，丢掉 image_url 等非文本块。

    :return: 拼接后的纯文本；无内容时返回 ""。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _recall_context_for(user_id: str, query: str, k: int = 3) -> str:
    """程序侧预调向量库，跳过 LLM 自觉调用 recall_long_memory。

    背景：让 LLM 自己决定要不要查记忆，它会偷懒（尤其在首次问诊时），
    所以这里强行先查一次，把命中事实作为 SystemMessage 注入到对话开头。

    :param user_id: 业务 user_id。
    :param query: 当前用户消息文本（用于 embedding 召回）。
    :param k: 召回条数，默认 3。
    :return: 拼接好的 SystemMessage 提示文本；无记忆或查询失败时返回 ""。
    """
    if not query or not query.strip():
        return ""
    try:
        docs = long_memory_store.similarity_search(
            query=query, filter={"user_id": user_id}, k=k,
        )
    except Exception as e:
        # 召回失败不能阻断对话，仅记日志
        logger.warning("[long_memory] recall failed: %s", e)
        return ""
    if not docs:
        return ""
    bullets = "\n".join(f"- {d['page_content']}" for d in docs)
    return (
        "以下是关于该用户的长期健康记忆（来自历史对话），请综合参考：\n"
        f"{bullets}"
    )


# 匹配 AI 回复末尾的 [OPTIONS]...[/OPTIONS] 追问块（re.DOTALL 让 . 跨行）
_OPTIONS_RE = re.compile(r"\[OPTIONS\]\s*(\{.*?\})\s*\[/OPTIONS\]", re.DOTALL)


def _parse_follow_up(reply_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """从 AI 回复中剥离 [OPTIONS]...[/OPTIONS] 块，提取结构化追问。

    :param reply_text: AI 完整回复文本（流式拼接后的 full_text）。
    :return: (剩余可见文本, follow_up dict 或 None)
        - 可见文本：去掉 OPTIONS 块后的纯文本（去掉首尾空白）。
        - follow_up：{"question": str, "options": [str, ...]} 或 None。

    容错策略：OPTIONS 块 JSON 错误 / 字段缺失 / 选项数越界（2~6）/
    含非字符串选项 → 视为无追问，原文保留，记 warning。
    """
    if not reply_text:
        return "", None
    match = _OPTIONS_RE.search(reply_text)
    if not match:
        return reply_text.strip(), None
    raw_json = match.group(1)
    visible = (reply_text[: match.start()] + reply_text[match.end() :]).strip()
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("[follow_up] invalid JSON in OPTIONS block: %r", raw_json[:120])
        return reply_text.strip(), None
    if not isinstance(data, dict):
        logger.warning("[follow_up] OPTIONS block is not a JSON object: %r", raw_json[:120])
        return reply_text.strip(), None
    question = data.get("question")
    options = data.get("options")
    if not isinstance(question, str) or not question.strip():
        logger.warning("[follow_up] missing/empty question: %r", raw_json[:120])
        return reply_text.strip(), None
    if not isinstance(options, list) or not (2 <= len(options) <= 6):
        logger.warning("[follow_up] options count out of range [2,6]: %r", raw_json[:120])
        return reply_text.strip(), None
    if not all(isinstance(o, str) and o.strip() for o in options):
        logger.warning("[follow_up] non-string/empty option: %r", raw_json[:120])
        return reply_text.strip(), None
    return visible, {
        "question": question.strip(),
        "options": [o.strip() for o in options],
    }


# 摘要 prompt：把本轮对话压成 1~3 条事实短句，由 LLM 自行抽取
_SUMMARY_PROMPT = """你是中医健康档案助手。请把以下一轮用户与中医 AI 的对话，压缩成 1~3 条"第三人称事实短句"，用于后续检索长期记忆。

要求：
- 仅抽取：体质、症状变化、作息/饮食调整、用药/养生反馈、既往病史关键节点
- 每条不超过 30 字，第三人称（"用户 ..."）
- 一条一行，纯短句，无编号、无标点堆砌
- 若本轮无有效健康信息，**只输出** NONE

对话内容：
用户：{user_text}
AI：{ai_text}

事实："""


def _summarize_and_save(user_id: str, user_text: str, ai_text: str) -> None:
    """调用 LLM 提取本轮事实并写入长期记忆（异步线程调用，失败仅记日志不抛）。

    :param user_id: 业务 user_id。
    :param user_text: 本轮用户消息文本（可空，会用占位符）。
    :param ai_text: 本轮 AI 可见文本（可空，会用占位符）。

    实现要点：
      - 拆行 + lstrip 数字 / 符号，处理 "1. xxx" 这类带编号的回答。
      - 单条长度 < 4 字直接跳过（无意义）。
      - 单条 add_text 失败不影响整体（继续处理下一条）。
    """
    if not user_text and not ai_text:
        return
    try:
        resp = model.invoke(
            _SUMMARY_PROMPT.format(
                user_text=user_text or "(无文本)",
                ai_text=ai_text or "(无回复)",
            )
        )
        facts_text = _flatten_content(resp.content).strip()
        if not facts_text or facts_text.upper() == "NONE":
            return
        saved = 0
        for line in facts_text.splitlines():
            # 去掉行首常见前缀: 编号 / 列表符号 / 中文项目符号
            fact = line.strip().lstrip("•-・1234567890. ").strip()
            if len(fact) < 4:
                continue
            try:
                long_memory_store.add_text(text=fact, metdata={"user_id": user_id})
                saved += 1
            except Exception as e:
                logger.warning("[long_memory] save failed for fact=%r: %s", fact, e)
        if saved:
            logger.info("[long_memory] auto-saved %d facts for user=%s", saved, user_id)
    except Exception as e:
        logger.warning("[long_memory] summarize failed: %s", e)


def save_tongue_report_to_memory(user_id: str, report: Dict[str, Any]) -> int:
    """舌象报告 → 长期记忆(结构化路径,绕开 LLM 抽取)。

    为什么绕开 ``_summarize_and_save``:
      - 舌象报告是结构化 JSON,直接构造事实短句更准,比让 LLM 二次抽取
        更稳(避免二次幻觉)。
      - 与 chat 路径的 summarize 并行存在:chat 走 LLM 抽自然语言事实,
        tongue 走结构化字段。两者互不重复。

    :param user_id: 业务 user_id(用于 TAG 隔离)。
    :param report: ``report.model_dump()`` 出来的 dict,需含
        ``tongue_features.*`` 与 ``tcm_pattern.constitution``。
    :return: 实际写入条数(已去重 / 已长度过滤)。

    写入策略:
      - 每个有意义的字段拆为独立短事实(constitution / 舌色 / 苔色 /
        苔厚 / 润燥 / 舌形 / 舌下络脉 / 整体印象),便于后续召回时按
        维度精准命中(例:用户说"最近舌苔厚不厚",能直接命中"苔厚"事实)。
      - 值是 ``"无法判断"`` 或字段缺失 → 跳过(避免在档案里污染无意义事实)。
      - 长度 < 4 字的条目跳过(与 ``save_long_memory`` tool 一致)。
      - 先 KNN 召回,相似度 ≥ 0.92(距离 ≤ 0.08)视为重复,跳过写入。
      - 单条 add_text 失败不影响其它条目。
    """
    UNKNOWN = "无法判断"

    def _skip_unknown(v: Any) -> bool:
        """值为 None / 空 / ``"无法判断"`` 时跳过该事实。"""
        if not v:
            return True
        if isinstance(v, str) and v.strip() == UNKNOWN:
            return True
        return False

    saved = 0
    try:
        features = (report.get("tongue_features") or {}) if isinstance(report, dict) else {}
        pattern = (report.get("tcm_pattern") or {}) if isinstance(report, dict) else {}

        facts: List[str] = []

        constitution = pattern.get("constitution") if isinstance(pattern, dict) else None
        if not _skip_unknown(constitution):
            facts.append(f"舌象辨证: {constitution}")

        tongue_color = features.get("tongue_color") if isinstance(features, dict) else None
        if not _skip_unknown(tongue_color):
            facts.append(f"舌色: {tongue_color}")

        coating_color = features.get("coating_color") if isinstance(features, dict) else None
        if not _skip_unknown(coating_color):
            facts.append(f"苔色: {coating_color}")

        coating_thickness = features.get("coating_thickness") if isinstance(features, dict) else None
        if not _skip_unknown(coating_thickness):
            facts.append(f"苔厚: {coating_thickness}")

        moisture = features.get("moisture") if isinstance(features, dict) else None
        if not _skip_unknown(moisture):
            facts.append(f"舌质润燥: {moisture}")

        tongue_shape = features.get("tongue_shape") if isinstance(features, dict) else None
        if isinstance(tongue_shape, list) and tongue_shape:
            valid_shape = [s for s in tongue_shape if isinstance(s, str) and s != UNKNOWN]
            if valid_shape:
                facts.append(f"舌形: {'+'.join(valid_shape)}")

        sublingual_vessels = features.get("sublingual_vessels") if isinstance(features, dict) else None
        if not _skip_unknown(sublingual_vessels):
            facts.append(f"舌下络脉: {sublingual_vessels}")

        overall_impression = features.get("overall_impression") if isinstance(features, dict) else None
        if not _skip_unknown(overall_impression):
            facts.append(f"舌象整体印象: {overall_impression}")

        for fact in facts:
            if len(fact) < 4:
                continue
            try:
                similar = long_memory_store.find_similar_with_score(
                    text=fact, user_id=user_id, threshold=0.08, k=5,
                )
                if similar:
                    logger.info(
                        "[long_memory] skip duplicate tongue fact for user=%s: %r (similar=%s)",
                        user_id, fact, f"{similar[0]['similarity']:.2f}",
                    )
                    continue
                long_memory_store.add_text(text=fact, metdata={"user_id": user_id})
                saved += 1
            except Exception as e:
                logger.warning("[long_memory] tongue save failed for fact=%r: %s", fact, e)
        if saved:
            logger.info("[long_memory] tongue auto-saved %d facts for user=%s", saved, user_id)
    except Exception as e:
        logger.warning("[long_memory] tongue summarize failed: %s", e)
    return saved


# LangChain 消息类 → 入库 role 的映射
_ROLE_MAP = {
    HumanMessage: "user",
    AIMessage: "assistant",
    ToolMessage: "tool",
    SystemMessage: "system",
}


def _persist_message(session_id: str, msg) -> Dict[str, Any]:
    """把一条 LangChain 消息落库，返回入库后的完整 row dict。

    :param session_id: 目标会话 id。
    :param msg: HumanMessage / AIMessage / ToolMessage / SystemMessage 任一。
    :return: db_log.add_message 返回的 dict（含 id / created_at 等）。
    """
    role = _ROLE_MAP.get(type(msg), "unknown")
    return db_log.add_message(
        session_id=session_id,
        role=role,
        content=_flatten_content(msg.content),
        tool_call_id=getattr(msg, "tool_call_id", None),
        tool_calls=(msg.tool_calls if getattr(msg, "tool_calls", None) else None),
        name=getattr(msg, "name", None),
    )


def _serialize_user_payload(
    text: Optional[str], images: List[str], videos: List[str]
) -> str:
    """用户消息入库格式：纯文本走纯字符串，含媒体走 JSON 便于前端还原。

    :return: 入库字符串（纯文本或 JSON 包装）。
    """
    if not images and not videos:
        return text or ""
    payload = {
        "type": "media",
        "text": text or "",
        "images": images or [],
        "videos": videos or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_human_content(
    text: Optional[str], images: List[str], videos: List[str]
) -> Union[str, List[Dict[str, Any]]]:
    """组装 LangChain 多模态 HumanMessage.content。

    当前 LLM（MiniMax-M3）不识别 video block，所以视频统一转成 `[视频] <url>` 文本块，
    让 LLM 知道有这个媒体但不会去解析。

    :return: 单文本时返回字符串（简化），多模态时返回 block 列表。
    """
    blocks: List[Dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for url in images or []:
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    for url in videos or []:
        blocks.append({"type": "text", "text": f"[视频] {url}"})
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return blocks[0]["text"]
    return blocks


# =====================================================================
# 6. 对外 API：chat (非流式) + chat_stream (SSE)
# =====================================================================


def chat(
    user_id: str,
    *,
    text: Optional[str] = None,
    images: Optional[List[str]] = None,
    videos: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], str, Optional[Dict[str, Any]]]:
    """调用智能体处理一条用户消息并把整轮对话入库（非流式）。

    完整流程：
      1) 会话管理（新建 / 校验归属）
      2) 用户消息入库
      3) 程序侧预调长期记忆 + 构造 invoke payload
      4) agent.invoke 拿到 state.messages，把 AI/tool 消息落库
      5) 解析 [OPTIONS] 块；带追问时改写 content 为 JSON 包装
      6) 异步线程跑摘要 + 入长期记忆（不阻塞响应）

    :param user_id: 业务 userid（来自 JWT sub 或外部传入）。
    :param text: 文本内容（可与 images/videos 组合，也可单独为空，但三者至少有一项）。
    :param images: 图片 URL 列表。
    :param videos: 视频 URL 列表（MiniMax-M3 不解析视频，转为文本标记传入 agent）。
    :param session_id: 已存在的会话 id；为空则自动新建。
    :return: (session_id, user_msg_row, ai_msg_row, ai_reply_text, follow_up)
        - session_id: 实际使用的会话 id。
        - user_msg_row: 入库后的用户消息 row。
        - ai_msg_row: 入库后的 AI 消息 row；agent 暂时无回复时为 {}。
        - ai_reply_text: AI 纯文本回复（追问块已剥离）。
        - follow_up: 结构化追问 {question, options} 或 None。
    :raises ValueError: text/images/videos 全为空；或 session_id 不存在/不属于该用户。
    """
    images = images or []
    videos = videos or []
    if not text and not images and not videos:
        raise ValueError("text/images/videos 至少需要一项")

    # 1. 会话：未传则新建；传入则要求它存在并归属 user_id
    if session_id is None:
        session_id = str(uuid.uuid4())
        db_log.create_session(user_id, session_id, system_prompt=SYSTEM_PROMPT)
    else:
        if db_log.get_session(session_id, user_id) is None:
            raise ValueError(f"session {session_id} 不存在或不属于用户 {user_id}")

    # thread_id 用 session_id，让 checkpointer 按会话隔离 state.messages
    cfg = {"configurable": {"user_id": user_id, "thread_id": session_id}}

    # 2. 先把用户消息入库（多模态用 JSON 序列化）
    user_payload = _serialize_user_payload(text, images, videos)
    user_row = db_log.add_message(
        session_id=session_id, role="user", content=user_payload
    )

    # 3. 调 agent：invoke 会把 HumanMessage 写进 checkpointer，state.messages 累计
    human_content = _build_human_content(text, images, videos)
    extra_context = _recall_context_for(user_id, text or "用户最近健康状况")
    invoke_messages: List[Any] = []
    if extra_context:
        invoke_messages.append(SystemMessage(content=extra_context))
    invoke_messages.append(HumanMessage(content=human_content))
    result = agent.invoke(
        {"messages": invoke_messages},
        config=cfg,
    )
    state_messages = result.get("messages", []) or []

    # 4. 把本次新增的 AI/tool 消息入库（跳过已经手动入库的用户消息）
    ai_row: Dict[str, Any] = {}
    ai_reply_raw = ""
    for msg in state_messages:
        if isinstance(msg, HumanMessage):
            continue
        row = _persist_message(session_id, msg)
        if isinstance(msg, AIMessage):
            text_content = _flatten_content(msg.content)
            if text_content:
                ai_reply_raw = text_content
                ai_row = row

    # 5. 解析 [OPTIONS]...[/OPTIONS] 块 → 结构化追问
    visible_reply, follow_up = _parse_follow_up(ai_reply_raw)
    ai_reply = visible_reply or ai_reply_raw

    # 6. 若带追问，把 AI 消息 content 改为 JSON 包装形式（便于历史回放还原）
    if follow_up and ai_row:
        wrapped = json.dumps(
            {"type": "ai_followup", "text": visible_reply, "follow_up": follow_up},
            ensure_ascii=False,
        )
        with db_log.SessionLocal() as _db:
            m = _db.query(db_log.ChatMessage).filter(
                db_log.ChatMessage.id == ai_row["id"]
            ).one()
            m.content = wrapped
            _db.commit()
        ai_row["content"] = wrapped
        ai_row["follow_up"] = follow_up

    # 7. 自动归档：本轮对话摘要 → 长期记忆（异步，不阻塞响应）
    threading.Thread(
        target=_summarize_and_save,
        args=(user_id, text, visible_reply),
        daemon=True,
    ).start()

    return session_id, user_row, ai_row, ai_reply, follow_up


def _extract_delta(content) -> str:
    """从流式 chunk.content 提取增量文本（保持 token 原序，不再插换行）。

    与 _flatten_content 的区别：本函数用于 SSE 增量，必须保持 token 原序，
    所以多 block 用空串拼接而不是 '\\n'。

    :return: 增量字符串；无文本时返回 ""。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _error_event(message: str, session_id: str = "") -> Dict[str, Any]:
    """统一的 error 事件：同时带 message / content (alias) / session_id。

    为什么三份冗余：
    - message：标准字段名（前端可能用这个解析）
    - content：旧版前端 SSE 解析逻辑用的是 content（这是事故现场多份冗余的由来）
    - session_id：让前端在 error 时也能拿到 sid，便于后续 loadLatest 重建关联
    """
    return {
        "event": "error",
        "message": message,
        "content": message,
        "session_id": session_id,
    }


async def chat_stream(
    user_id: str,
    *,
    text: Optional[str] = None,
    images: Optional[List[str]] = None,
    videos: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """SSE 流式调用智能体。

    事件序列（成功路径）：
      1) {"event": "start",  "session_id": ..., "user_message": {...}}
      2) {"event": "chunk",  "content": "..."}    多次（每个 token 增量）
      3) {"event": "end",    "ai_message":  {...}, "follow_up": ...}

    异常路径（任意阶段）：
      {"event": "error", "message": "...", "content": "...", "session_id": "..."}

    实现要点：
      - 立即派发 start（含 session_id），让前端在等首字前就能拿到 sid。
      - 只透传 AIMessage 的文本增量；tool_calls / 非 AIMessage 静默丢弃。
      - 流式收齐 full_text 后再解析 [OPTIONS]（不边流边解析，避免边界问题）。
      - AI 入库时若带 follow_up，content 同样改成 JSON 包装。
      - 末尾再异步入库长期记忆（不阻塞流的关闭）。

    :yield: dict 形式的事件 payload，由上层 json.dumps + SSE 序列化。
    """
    images = images or []
    videos = videos or []
    if not text and not images and not videos:
        yield _error_event("text/images/videos 至少需要一项")
        return

    # 1. 会话：未传则新建；传入则校验归属
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())
            db_log.create_session(user_id, session_id, system_prompt=SYSTEM_PROMPT)
        else:
            if db_log.get_session(session_id, user_id) is None:
                yield _error_event(
                    f"session {session_id} 不存在或不属于用户 {user_id}",
                    session_id=session_id,
                )
                return
    except Exception as e:
        yield _error_event(f"会话初始化失败: {type(e).__name__}: {e}")
        return

    cfg = {"configurable": {"user_id": user_id, "thread_id": session_id}}

    # 2. 用户消息先入库
    user_payload = _serialize_user_payload(text, images, videos)
    user_row = db_log.add_message(
        session_id=session_id, role="user", content=user_payload
    )

    # 3. 立刻派发 start，前端握手成功，避免等首字超时
    yield {"event": "start", "session_id": session_id, "user_message": user_row}

    # 4. 流式调 agent：只把 AIMessage 的文本增量透传给前端；tool 调用对用户静默
    accumulated: List[str] = []
    human_content = _build_human_content(text, images, videos)
    extra_context = _recall_context_for(user_id, text or "用户最近健康状况")
    invoke_messages: List[Any] = []
    if extra_context:
        invoke_messages.append(SystemMessage(content=extra_context))
    invoke_messages.append(HumanMessage(content=human_content))
    try:
        async for chunk, _metadata in agent.astream(
            {"messages": invoke_messages},
            config=cfg,
            stream_mode="messages",
        ):
            if not isinstance(chunk, AIMessage):
                continue
            if getattr(chunk, "tool_calls", None):
                # tool 调用阶段不暴露给前端，避免 UI 抖动
                continue
            delta = _extract_delta(chunk.content)
            if not delta:
                continue
            accumulated.append(delta)
            yield {"event": "chunk", "content": delta}
    except Exception as e:
        yield _error_event(f"{type(e).__name__}: {e}", session_id=session_id)
        return

    # 5. AI 整段入库（若含 [OPTIONS] 块，content 改为 JSON 包装）
    full_text = "".join(accumulated)
    visible_text, follow_up = _parse_follow_up(full_text)
    if follow_up:
        ai_content = json.dumps(
            {"type": "ai_followup", "text": visible_text, "follow_up": follow_up},
            ensure_ascii=False,
        )
    else:
        ai_content = full_text
    ai_row = db_log.add_message(
        session_id=session_id, role="assistant", content=ai_content
    )
    if follow_up:
        ai_row["follow_up"] = follow_up
    yield {
        "event": "end",
        "ai_message": ai_row,
        "follow_up": follow_up,
    }

    # 6. 自动归档：本轮对话摘要 → 长期记忆（异步，不阻塞流）
    save_text = visible_text or full_text
    threading.Thread(
        target=_summarize_and_save,
        args=(user_id, text, save_text),
        daemon=True,
    ).start()