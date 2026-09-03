"""测试全局配置：必须在任何 app import 之前设置 DB_URL，避免 db.py 读不到。"""

import os

os.environ.setdefault("DB_URL", "sqlite:///:memory:")
