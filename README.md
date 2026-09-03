# zhongyiguan · 中医 AI 聊天助手

基于 LangChain + LangGraph 的中医辨证聊天助手，支持：

- 流式 SSE 输出（中医辨证 + 结构化追问）
- 长期健康记忆（跨会话、按用户隔离，向量召回）
- JWT 鉴权 + argon2 密码哈希
- MySQL 持久化聊天历史 + Redis 向量记忆 / Agent checkpointer

## 技术栈

| 层 | 选型 |
|---|---|
| API | FastAPI 0.141 + uvicorn |
| Agent | LangChain 1.3 + LangGraph 1.2 |
| LLM | MiniMax-M3（走原生 chatcompletion_v2 端点） |
| Embedding | MiniMax embo-01（1536 维） |
| 向量库 | RedisVL（RediSearch HNSW/flat + COSINE） |
| Checkpointer | langgraph-checkpoint-redis（AsyncRedisSaver） |
| 数据库 | MySQL 8 + SQLAlchemy 2 |
| 鉴权 | JWT (HS256) + argon2-cffi |

## 项目结构

```
zhongyiguan/
├── main.py                # uvicorn 入口
├── pyproject.toml
├── requirements.txt
├── .env.example           # 配置模板（不含真实 key）
├── app/
│   ├── main.py            # re-export FastAPI app
│   ├── api/
│   │   ├── app.py         # create_app() + middleware + startup
│   │   ├── auth.py        # /auth/*  注册/登录/token
│   │   ├── chat.py        # /chat/*  流式 SSE + 历史
│   │   ├── health.py      # /health /health/live
│   │   ├── logging_middleware.py
│   │   └── response.py    # 统一响应包络 {code, msg, data}
│   ├── agent/
│   │   └── zygchat.py     # LLM client + tools + agent + memory store
│   └── model/
│       ├── db.py          # SQLAlchemy engine + SessionLocal
│       ├── users.py       # users 表
│       └── chat_log.py    # chat_sessions / chat_messages 表
└── tests/
```

## 本地开发

```bash
# 1. 准备依赖服务
#   - MySQL 8.x（建库 zhongyiguan）
#   - Redis Stack（含 RediSearch 模块，apt: redis-stack-server）

# 2. 配置
cp .env.example .env
# 编辑 .env，至少填好：
#   DB_URL=mysql+pymysql://user:pwd@127.0.0.1:3306/zhongyiguan?charset=utf8mb4
#   REDIS_URL=redis://localhost:6379/0
#   MINIMAX_API_KEY=<your-key>

# 3. 安装 + 启动
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
APP_ENV=development python main.py   # 开发模式 reload=True

# 健康检查
curl http://127.0.0.1:8005/health/live
curl http://127.0.0.1:8005/health
```

## 生产部署

依赖：MySQL 8 + Redis Stack + Nginx + Python 3.14。

核心步骤（详见内部运维手册）：
1. `git clone` 到 `/opt/zhongyiguan`
2. `python3.14 -m venv .venv && pip install -r requirements.txt`
3. 写 `.env`（含 `JWT_SECRET`，`openssl rand -hex 32` 生成）
4. 配 systemd 单元，监听 `127.0.0.1:8005`
5. Nginx 反代终结 HTTPS，开启 SSE 友好配置（`proxy_buffering off`、超时放大）

## 配置项速查

| 变量 | 必填 | 说明 |
|---|---|---|
| `APP_ENV` | ❌（默认 production） | `production` 下 JWT_SECRET 必填且 ≥ 32 字符 |
| `APP_WORKERS` | ❌（默认 2） | uvicorn worker 数 |
| `APP_CHECKPOINTER` | ❌（默认 redis） | `redis` / `memory` |
| `DB_URL` | ✅ | SQLAlchemy DSN |
| `REDIS_URL` | ✅ | 含密码时 `redis://:pwd@host:6379/0` |
| `MINIMAX_API_KEY` | ✅ | LLM + Embedding 共用 |
| `MINIMAX_BASE_URL` | ❌ | 默认 `https://api.minimax.chat/v1` |
| `MINIMAX_CHAT_MODEL` | ❌ | 默认 `MiniMax-M3` |
| `JWT_SECRET` | 生产必填 | `openssl rand -hex 32` |

## API 速览

| 路径 | 方法 | 说明 |
|---|---|---|
| `/health/live` | GET | 进程存活 |
| `/health` | GET | 依赖就绪（DB + Redis），任一挂掉 503 |
| `/auth/register` | POST | 手机号注册 |
| `/auth/login` | POST | 登录返回 access/refresh token |
| `/auth/refresh` | POST | 刷新 access token |
| `/chat/send` | POST | 非流式调用 |
| `/chat/stream` | POST (SSE) | 流式调用，事件：`start` / `chunk` / `end` / `error` |
| `/chat/history` | GET | 历史消息列表 |
| `/chat/sessions` | GET | 会话列表 |

所有业务 JSON 响应统一包络：
```json
{ "code": 200, "msg": "成功", "data": { ... } }
```

## 长期记忆机制

- 存储：RedisVL 向量索引 `agent_long_memory_v2`，维度 1536，COSINE
- 隔离：每条记忆带 `user_id` TAG，DIALECT=3 严格按用户过滤（防越权）
- 写入：LLM 自主调 `save_long_memory` 工具；写入前 KNN 召回（距离 ≤ 0.08 ≈ 相似度 ≥ 0.92）判定重复，命中则跳过并提示已有事实
- 读取：双向 ——
  - **程序侧**（`_recall_context_for`）：每轮先做一次 KNN 召回，把命中事实作为 SystemMessage 注入，跳过 LLM 偷懒
  - **LLM 侧**：仍可自主调 `recall_long_memory` 取更针对性的查询
