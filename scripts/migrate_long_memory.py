"""把 1024 维的旧 agent_long_memory 索引迁移到 1536 维的 agent_long_memory_v2。

用法:
    python scripts/migrate_long_memory.py --dry-run          # 只统计
    python scripts/migrate_long_memory.py --batch-size 50    # 实际迁移
    python scripts/migrate_long_memory.py --user u_001       # 只迁某个用户

为什么需要这个脚本:
    MiniMax embo-01 固定 1536 维,与历史 DashScope text-embedding-v3 的 1024 维不兼容。
    v2 索引用新维度建好后,旧 1024 维的向量不能直接搬过去,只能重新嵌入。

实现要点:
    - 用 FT.SEARCH 直接读旧索引(只要 user_id / fact / id 三个字段,不要 embedding)
    - 用 MiniMaxEmbeddings 重新编码
    - 用确定性 doc id ``mem:{user_id}:{sha1(fact)[:16]}`` 写入 v2,天然幂等
    - 旧索引不存在 → 直接退出 (no-op)
    - 失败单条不阻塞整体,记 warning 继续
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

# 让脚本可以从项目根直接运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 在 import app.agent.zygchat 之前 stub init_db,避免脚本依赖 MySQL
from app.model import db as _db_module  # noqa: E402
_db_module.init_db = lambda: None  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from redisvl.index import SearchIndex  # noqa: E402
from redisvl.redis.connection import RedisConnectionFactory  # noqa: E402
from redisvl.schema import IndexSchema  # noqa: E402

from app.agent.zygchat import (  # noqa: E402
    EMBEDDINGS,
    REDIS_URL,
    INDEX_NAME as NEW_INDEX_NAME,
)

OLD_INDEX_NAME = "agent_long_memory"
OLD_INDEX_DIMS = 1024
BATCH_SIZE_DEFAULT = 50

logger = logging.getLogger("migrate_long_memory")


def _build_old_index(client) -> SearchIndex:
    """旧索引的 schema(只用于读取,所以 embedding 维度按历史 1024 声明)。"""
    schema = IndexSchema.from_dict({
        "index": {"name": OLD_INDEX_NAME, "prefix": "mem"},
        "fields": [
            {"name": "user_id", "type": "tag"},
            {"name": "fact", "type": "text"},
            {"name": "embedding", "type": "vector",
             "attrs": {"algorithm": "flat", "dims": OLD_INDEX_DIMS,
                       "distance_metric": "cosine", "datatype": "float32"}},
        ],
    })
    return SearchIndex(schema, client)


def _iter_old_docs(client, user_filter: str | None):
    """分页拉取旧索引全部文档。

    :param user_filter: 限定单个 user_id(可选)。
    :yield: (doc_id, user_id, fact) 三元组
    """
    page_size = 1000
    offset = 0
    while True:
        args = [
            "FT.SEARCH", OLD_INDEX_NAME,
            f"@user_id:{{{user_filter}}}" if user_filter else "*",
            "RETURN", "2", "user_id", "fact",
            "LIMIT", str(offset), str(page_size),
        ]
        raw = client.execute_command(*args)
        if not isinstance(raw, list) or len(raw) == 0:
            break
        total = raw[0] if isinstance(raw[0], int) else int(raw[0])
        if total == 0 or len(raw) == 1:
            break

        returned_count = (len(raw) - 1) // 2
        for i in range(1, len(raw), 2):
            doc_id = raw[i].decode() if isinstance(raw[i], bytes) else raw[i]
            fields = raw[i + 1] if i + 1 < len(raw) else []
            user_id, fact = "", ""
            if isinstance(fields, list):
                for j in range(0, len(fields) - 1, 2):
                    fname = fields[j].decode() if isinstance(fields[j], bytes) else fields[j]
                    fval = fields[j + 1].decode() if isinstance(fields[j + 1], bytes) else fields[j + 1]
                    if fname == "user_id":
                        user_id = fval
                    elif fname == "fact":
                        fact = fval
            if fact:
                yield doc_id, user_id, fact

        if returned_count < page_size:
            break
        offset += page_size


def _new_doc_id(user_id: str, fact: str) -> str:
    """确定性 doc id,让 v2 写入天然幂等。"""
    h = hashlib.sha1(f"{user_id}\x00{fact}".encode("utf-8")).hexdigest()[:16]
    return f"mem:{user_id}:{h}"


def migrate(dry_run: bool, batch_size: int, user_filter: str | None) -> None:
    client = RedisConnectionFactory.get_redis_connection(REDIS_URL)
    old_index = _build_old_index(client)

    if not old_index.exists():
        logger.info("旧索引 %s 不存在,无需迁移,退出。", OLD_INDEX_NAME)
        return

    # 准备新索引(zygchat.py 已经在启动时建好了,这里确保存在)
    try:
        client.execute_command("FT.INFO", NEW_INDEX_NAME)
    except Exception as e:
        logger.error("新索引 %s 不存在: %s\n请先启动一次 app 让 zygchat 把它建好,再跑迁移。", NEW_INDEX_NAME, e)
        sys.exit(1)

    logger.info("开始迁移: 旧=%s(1024d) → 新=%s(1536d), dry_run=%s, batch=%d",
                OLD_INDEX_NAME, NEW_INDEX_NAME, dry_run, batch_size)

    total = 0
    written = 0
    skipped = 0
    failed = 0

    batch_facts: list[tuple[str, str, str]] = []  # (doc_id_new, user_id, fact)
    for old_id, user_id, fact in _iter_old_docs(client, user_filter):
        total += 1
        new_id = _new_doc_id(user_id, fact)
        batch_facts.append((new_id, user_id, fact))

        if len(batch_facts) >= batch_size:
            n_w, n_s, n_f = _flush_batch(client, batch_facts, dry_run)
            written += n_w
            skipped += n_s
            failed += n_f
            batch_facts.clear()
            logger.info("进度: total=%d written=%d skipped=%d failed=%d", total, written, skipped, failed)

    if batch_facts:
        n_w, n_s, n_f = _flush_batch(client, batch_facts, dry_run)
        written += n_w
        skipped += n_s
        failed += n_f

    logger.info("迁移完成: total=%d written=%d skipped=%d failed=%d (dry_run=%s)",
                total, written, skipped, failed, dry_run)


def _flush_batch(client, batch: list[tuple[str, str, str]], dry_run: bool) -> tuple[int, int, int]:
    """处理一批:嵌入 + 写入(带去重)。

    :return: (written, skipped, failed) 三元组
    """
    written = skipped = failed = 0

    # 1) 先检查哪些 new_id 已存在,避免重复嵌入
    new_ids = [item[0] for item in batch]
    existing = set()
    try:
        # HEXISTS 批量查(用 pipeline)
        pipe = client.pipeline()
        for nid in new_ids:
            pipe.exists(nid)
        results = pipe.execute()
        existing = {nid for nid, exists in zip(new_ids, results) if exists}
    except Exception as e:
        logger.warning("检查 existing 失败,降级为全部重新嵌入: %s", e)

    todo = [item for item in batch if item[0] not in existing]
    skipped += len(batch) - len(todo)

    if not todo:
        return written, skipped, failed

    # 2) 批量嵌入
    facts = [item[2] for item in todo]
    try:
        vectors = EMBEDDINGS.embed_documents(facts)
    except Exception as e:
        logger.error("批量嵌入失败(batch=%d): %s", len(facts), e)
        return written, skipped, failed + len(facts)

    if len(vectors) != len(facts):
        logger.error("嵌入返回数量不匹配: 请求=%d, 返回=%d", len(facts), len(vectors))
        return written, skipped, failed + len(facts)

    if dry_run:
        return written + len(facts), skipped, failed

    # 3) 逐条写入(失败不阻塞其他)
    import numpy as np
    for (new_id, user_id, fact), vec in zip(todo, vectors):
        try:
            embedding_bytes = np.array(vec, dtype=np.float32).tobytes()
            client.hset(new_id, mapping={
                "user_id": user_id,
                "fact": fact,
                "embedding": embedding_bytes,
            })
            written += 1
        except Exception as e:
            logger.warning("写入失败 doc=%s fact=%r: %s", new_id, fact[:30], e)
            failed += 1

    return written, skipped, failed


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="迁移长期记忆索引 1024d → 1536d")
    parser.add_argument("--dry-run", action="store_true", help="只统计,不写入")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="每批嵌入条数")
    parser.add_argument("--user", type=str, default=None, help="只迁移某个 user_id")
    args = parser.parse_args()

    migrate(dry_run=args.dry_run, batch_size=args.batch_size, user_filter=args.user)


if __name__ == "__main__":
    main()
