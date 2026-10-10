"""舌象分析接口：上传舌照，多模态 LLM 出结构化辨证报告。

最小闭环（v1）：
  前端拍照 → multipart 上传 → base64 → Qwen-VL → JSON → Pydantic 校验 → 返回 + 落库。

不做：
  - 不存图（bytes 仅在请求生命周期内持有）
  - 不引入流式输出

落库策略（与 chat 复用同一张表）：
  - 每次成功调用新建一个 ChatSession，写 2 条 ChatMessage：
    - role=user,      content=主诉文本（空则 ""）
    - role=assistant, content={"type":"tongue_report","text":summary,"report":{...}}
  - LLM 上游错 / schema 校验失败不写库，聊天历史保持干净。

后续接视觉专用模型 / RAG / 报告渲染时,只换 ``_call_qwen_vl`` 内部即可，
路由层 schema 不变。
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import uuid
from typing import List, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ValidationError

from app.agent import zygchat
from app.api.auth import get_current_userid
from app.model import chat_log as db_log


router = APIRouter(prefix="/tongue", tags=["tongue"])


# =====================================================================
# Pydantic 输出 schema（与 system prompt 中的 JSON 描述一一对应）
# =====================================================================

class TongueFeatures(BaseModel):
    """舌象八维客观特征。Literal 锁死候选词表,LLM 想越界会被拒。"""
    tongue_color: Literal["淡白", "淡红", "红", "绛红", "青紫", "瘀斑"]
    coating_color: Literal["白苔", "黄苔", "灰黑苔"]
    coating_thickness: Literal["无苔", "薄苔", "厚苔", "腻苔"]
    tongue_shape: List[Literal["胖大", "瘦薄", "齿痕", "裂纹", "点刺"]]
    moisture: Literal["湿润", "偏干", "燥裂", "水滑"]
    sublingual_vessels: Literal["正常", "粗大", "迂曲", "瘀血"]
    spots: List[Literal["瘀点", "红点", "芒刺"]]
    overall_impression: Literal["气虚", "湿热", "寒湿", "阴虚", "阳虚", "痰湿"]


class TcmPattern(BaseModel):
    """中医辨证结论。"""
    constitution: Literal["气虚", "湿热", "寒湿", "阴虚", "阳虚", "痰湿"]
    pathogenesis: str
    cold_heat: Literal["寒", "热", "平"]
    deficiency_excess: Literal["虚", "实", "虚实夹杂"]
    organ_focus: List[str]


class Regimen(BaseModel):
    """调理方案。每条建议一行,前端直接渲染。"""
    diet: List[str]
    lifestyle: List[str]
    exercise: List[str]
    acupoints: List[str]
    tea: List[str]


class TongueReport(BaseModel):
    """完整舌诊报告。"""
    tongue_features: TongueFeatures
    tcm_pattern: TcmPattern
    regimen: Regimen
    classical_formula_hints: List[str]
    summary: str
    confidence_notes: str


# =====================================================================
# System prompt：约束 LLM 输出纯 JSON,无 markdown/前后缀
# =====================================================================

_SYSTEM_PROMPT = """你是资深中医师,正通过患者上传的舌象照片进行舌诊分析。

输出必须是且仅是一个 JSON 对象,严格按下面的字段名和取值,不要新增/省略字段、不要使用任何未列出的取值:

{
  "tongue_features": {
    "tongue_color": "淡白|淡红|红|绛红|青紫|瘀斑",
    "coating_color": "白苔|黄苔|灰黑苔",
    "coating_thickness": "无苔|薄苔|厚苔|腻苔",
    "tongue_shape": ["胖大|瘦薄|齿痕|裂纹|点刺", ...],
    "moisture": "湿润|偏干|燥裂|水滑",
    "sublingual_vessels": "正常|粗大|迂曲|瘀血",
    "spots": ["瘀点|红点|芒刺", ...],
    "overall_impression": "气虚|湿热|寒湿|阴虚|阳虚|痰湿"
  },
  "tcm_pattern": {
    "constitution": "气虚|湿热|寒湿|阴虚|阳虚|痰湿",
    "pathogenesis": "病机自由文本",
    "cold_heat": "寒|热|平",
    "deficiency_excess": "虚|实|虚实夹杂",
    "organ_focus": ["心|肝|脾|肺|肾|胃|胆", ...]
  },
  "regimen": {
    "diet":      ["建议1", "建议2"],
    "lifestyle": ["建议1"],
    "exercise":  ["建议1"],
    "acupoints": ["穴位1", "穴位2"],
    "tea":       ["茶饮1"]
  },
  "classical_formula_hints": ["方剂1", "方剂2"],
  "summary": "一段话总结,≤200字",
  "confidence_notes": "无法判断的字段填'无法判断'并在此说明"
}

硬性要求:
1. 输出且仅输出上面这个 JSON 对象,不要 Markdown 代码块、不要任何前后缀文字、不要第二段。
2. 若图像质量差/角度异常/确实看不出某项,该字段填字符串 "无法判断" 并在 confidence_notes 里说明。
3. 本结果仅作养生参考,不是医学诊断,不可替代面诊。
"""


# =====================================================================
# DashScope Qwen-VL 调用（与现有 MiniMaxChatModel 解耦,便于后续替换）
# =====================================================================

_DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
_VL_MODEL = os.getenv("DASHSCOPE_VL_MODEL", "qwen-vl-plus")
_MAX_UPLOAD_MB = int(os.getenv("TONGUE_MAX_UPLOAD_MB", "10"))
_ALLOWED_MIME = {"image/jpeg", "image/png"}


async def _call_qwen_vl(data_url: str, user_symptoms: str) -> str:
    """POST 到 DashScope OpenAI 兼容 chat/completions,返回 LLM 原始文本。

    测试里 monkeypatch 这个名字即可模拟上游;不依赖具体 httpx 写法。
    """
    if not _DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置,无法调用 Qwen-VL")

    user_content: List[dict] = []
    if user_symptoms:
        user_content.append({"type": "text", "text": f"患者主诉:{user_symptoms}"})
    user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    payload = {
        "model": _VL_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {_DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=45.0, write=10.0, pool=5.0),
    ) as client:
        resp = await client.post(
            f"{_DASHSCOPE_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """剥掉 LLM 偶尔输出的 ```json ... ``` 围栏,容忍围栏前后有空白。"""
    return _FENCE_RE.sub("", text.strip()).strip()


# =====================================================================
# 路由
# =====================================================================

@router.post("/analyze", summary="上传舌象照片,返回结构化中医辨证报告")
async def analyze_tongue(
    image: UploadFile = File(..., description="舌象照片,jpg/png"),
    user_symptoms: str = Form("", description="可选,患者主诉文本"),
    _userid: str = Depends(get_current_userid),
) -> dict:
    """接收 multipart 上传 → base64 → Qwen-VL → JSON 校验 → 返回结构化报告。

    错误码:
      400  invalid_type   - 文件类型不是 jpg/png
      413  too_large      - 超过 TONGUE_MAX_UPLOAD_MB
      401  鉴权失败        - 由 get_current_userid 抛出
      422  invalid_json   - LLM 输出不是合法 JSON 或不符合 schema
      502  upstream_error - DashScope 失败 / 网络错误
    """
    if image.content_type not in _ALLOWED_MIME:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_type")

    raw = await image.read()
    max_bytes = _MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail="too_large")

    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{image.content_type};base64,{b64}"

    try:
        raw_text = await _call_qwen_vl(data_url, user_symptoms)
    except httpx.HTTPError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"upstream_error: {e.__class__.__name__}")
    except RuntimeError as e:
        # 配置缺失等本侧错误,直接 502 也合理（属于上游不可用）
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"upstream_error: {e}")

    cleaned = _strip_code_fence(raw_text)
    try:
        report = TongueReport.model_validate_json(cleaned)
    except ValidationError as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"invalid_json: {e.error_count()} 处错误",
        )

    # 落库：复用 ChatSession/ChatMessage，前端 GET /chat/messages 直接能拉到。
    # 必须等 LLM + schema 都成功才写，避免失败路径在聊天历史里留孤儿 user 消息。
    session_id = str(uuid.uuid4())
    db_log.create_session(_userid, session_id, system_prompt=None)
    db_log.add_message(session_id, "user", user_symptoms or "")
    report_dict = report.model_dump()
    report_payload = json.dumps(
        {
            "type": "tongue_report",
            "text": report.summary or "舌象辨证报告",
            "report": report_dict,
        },
        ensure_ascii=False,
    )
    db_log.add_message(session_id, "assistant", report_payload)

    # 长期记忆：把舌象报告里的稳定信号（体质、舌色+苔色）入库，便于后续
    # chat 召回时 LLM 能拿到历次舌象佐证症状。走 zygchat 的结构化路径（绕开
    # LLM 抽取，避免二次幻觉）。daemon 线程不阻塞响应。
    threading.Thread(
        target=zygchat.save_tongue_report_to_memory,
        args=(_userid, report_dict),
        daemon=True,
    ).start()

    return report_dict
