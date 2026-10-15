"""测试全局配置：必须在任何 app import 之前设置环境变量。

  - load_dotenv: 必须先加载 .env,否则 test_stt.py 里 setdefault("MINIMAX_API_KEY", ...)
                 会在 zygchat 模块加载之前把 EMBEDDINGS.api_key 锁成假 key,
                 导致 test_tongue.py 调真实 MiniMax embeddings 全部 1004 鉴权失败。
  - DB_URL:        db.py 模块级就要读,默认 sqlite:///:memory: 就够用
  - APP_ENV:       默认 production 会强制要求 JWT_SECRET,测试里切到 development
  - JWT_SECRET:    dev 环境下 auth.py 也会兜底生成进程内随机 secret,
                   但同一个进程内签发/校验用的是同一个 secret,显式设一个固定的更稳
"""

import os

from dotenv import load_dotenv

# 先加载真实 .env,确保 MINIMAX_API_KEY 等用真 key;后续 test 里的 setdefault 不会覆盖。
load_dotenv()

os.environ.setdefault("DB_URL", "sqlite:///:memory:")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault(
    "JWT_SECRET",
    "test-only-jwt-secret-please-do-not-use-in-production-32+chars",
)
