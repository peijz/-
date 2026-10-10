"""测试全局配置：必须在任何 app import 之前设置环境变量。

  - DB_URL:        db.py 模块级就要读,默认 sqlite:///:memory: 就够用
  - APP_ENV:       默认 production 会强制要求 JWT_SECRET,测试里切到 development
  - JWT_SECRET:    dev 环境下 auth.py 也会兜底生成进程内随机 secret,
                   但同一个进程内签发/校验用的是同一个 secret,显式设一个固定的更稳
"""

import os

os.environ.setdefault("DB_URL", "sqlite:///:memory:")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault(
    "JWT_SECRET",
    "test-only-jwt-secret-please-do-not-use-in-production-32+chars",
)
