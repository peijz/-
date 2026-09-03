import os

import uvicorn

# 启动命令:
#   开发: APP_ENV=development python main.py
#   生产: APP_ENV=production APP_WORKERS=4 python main.py
#   或直接: uvicorn app.main:app --host 0.0.0.0 --port 8005 --workers 4
ENV = os.getenv("APP_ENV", "production").lower()
HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("APP_PORT", "8005"))
WORKERS = int(os.getenv("APP_WORKERS", "2"))
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "info").lower()

IS_DEV = ENV == "development"
RELOAD = IS_DEV
# uvicorn 不允许 reload=True 与 workers>1 同时使用
WORKERS = 1 if RELOAD else max(WORKERS, 1)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        workers=WORKERS,
        log_level=LOG_LEVEL,
    )
