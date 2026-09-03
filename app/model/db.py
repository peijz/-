import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL = os.environ["DB_URL"]

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

Base = declarative_base()


def init_db() -> None:
    """创建所有未存在的表。失败时直接抛 OperationalError，便于上游 fail-fast。"""
    from app.model import chat_log, users  # noqa: F401,E402  确保模型已注册到 Base.metadata

    Base.metadata.create_all(engine)
