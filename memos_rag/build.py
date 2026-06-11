#!/usr/bin/env python3

import sqlite3
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
from . import environment
from .utils import encode_vector, embed_texts

# ── Config ────────────────────────────────────────────────────────────────────


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def init_rag_db(rag: sqlite3.Connection, embed_dim: int):
    """Create tables if they don't exist. Uses a plain BLOB column for vectors
    since sqlite-vec may not be available; swap the similarity query if you
    install it as an extension."""
    rag.executescript(f"""
        CREATE TABLE IF NOT EXISTS memo_embeddings (
            memo_id     INTEGER PRIMARY KEY,
            uid         TEXT NOT NULL,
            created_ts  BIGINT NOT NULL,
            updated_ts  BIGINT NOT NULL,
            content_hash TEXT NOT NULL,
            content     TEXT NOT NULL,
            embedding   BLOB NOT NULL          -- {embed_dim} × float32 (IEEE-754)
        );
    """)
    rag.commit()


# ── Core indexing ─────────────────────────────────────────────────────────────


def fetch_memos_since(memos: sqlite3.Connection, since_ts: int) -> list[dict]:
    rows = memos.execute(
        """
        SELECT id, uid, created_ts, updated_ts, content
        FROM   memo
        WHERE  updated_ts > ?
          AND  row_status = 'NORMAL'
          AND  visibility != 'ARCHIVED'
        ORDER  BY updated_ts ASC
    """,
        (since_ts,),
    ).fetchall()

    return [
        {
            "id": r[0],
            "uid": r[1],
            "created_ts": r[2],
            "updated_ts": r[3],
            "content": r[4],
        }
        for r in rows
    ]


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def fetch_memos_needing_index(
    memos: sqlite3.Connection, rag: sqlite3.Connection
) -> list[dict]:
    # Get all current memos
    all_memos = memos.execute("""
        SELECT id, uid, created_ts, updated_ts, content
        FROM   memo
        WHERE  row_status = 'NORMAL'
    """).fetchall()

    # Get existing hashes from RAG DB in one query
    existing = {
        row[0]: row[1]
        for row in rag.execute("SELECT memo_id, content_hash FROM memo_embeddings")
    }

    to_index = []
    for id, uid, created_ts, updated_ts, content in all_memos:
        hash = content_hash(content)
        if id not in existing:
            to_index.append(
                {
                    "id": id,
                    "uid": uid,
                    "created_ts": created_ts,
                    "updated_ts": updated_ts,
                    "content": content,
                    "content_hash": hash,
                    "reason": "new",
                }
            )
        elif existing[id] != hash:
            to_index.append(
                {
                    "id": id,
                    "uid": uid,
                    "created_ts": created_ts,
                    "updated_ts": updated_ts,
                    "content": content,
                    "content_hash": hash,
                    "reason": "changed",
                }
            )

    return to_index


def fetch_all_memos(memos: sqlite3.Connection) -> list[dict]:
    rows = memos.execute(
        """
        SELECT id, uid, created_ts, updated_ts, content
        FROM   memo
        WHERE  row_status = 'NORMAL'
          AND  visibility != 'ARCHIVED'
        ORDER  BY updated_ts ASC
    """,
    ).fetchall()

    return [
        {
            "id": r[0],
            "uid": r[1],
            "created_ts": r[2],
            "updated_ts": r[3],
            "content": r[4],
            "content_hash": content_hash(r[4]),
            "reason": "full_reindex",
        }
        for r in rows
    ]


def index_memos(
    memos_list: list[dict],
    rag: sqlite3.Connection,
    batch_size: int,
    ollama_url: str,
    embed_model: str,
):
    """Embed and upsert a list of memo dicts."""
    if not memos_list:
        return

    total = len(memos_list)
    log.info(f"Indexing {total} memo(s)…")

    for batch_start in range(0, total, batch_size):
        batch = memos_list[batch_start : batch_start + batch_size]
        texts = [m["content"] for m in batch]

        log.info(f"  Embedding batch {batch_start}–{batch_start + len(batch) - 1}")
        embeddings = embed_texts(texts, ollama_url, embed_model)

        rag.executemany(
            """
            INSERT OR REPLACE INTO memo_embeddings
                (memo_id, uid, created_ts, updated_ts, content, content_hash, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                (
                    m["id"],
                    m["uid"],
                    m["created_ts"],
                    m["updated_ts"],
                    m["content"],
                    m["content_hash"],
                    encode_vector(emb),
                )
                for m, emb in zip(batch, embeddings)
            ],
        )
        rag.commit()

    log.info("Batch done.")


def do_build():
    MEMOS_DB = environment.Environment().memos_db
    RAG_DB = environment.Environment().rag_db
    OLLAMA_URL = environment.Environment().ollama_url
    BATCH_SIZE = environment.Environment().batch_size
    EMBED_DIM = environment.Environment().embed_dim
    EMBED_MODEL = environment.Environment().embed_model

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="Force a full reindex (ignores watermark)"
    )
    args = parser.parse_args()

    # Ensure RAG directory exists
    Path(RAG_DB).parent.mkdir(parents=True, exist_ok=True)

    memos = sqlite3.connect(f"file:{MEMOS_DB}?mode=ro", uri=True)  # read-only
    rag = sqlite3.connect(RAG_DB)

    init_rag_db(rag, EMBED_DIM)

    memos_list = (
        fetch_memos_needing_index(memos, rag)
        if not args.full
        else fetch_all_memos(memos)
    )
    log.info(f"Found {len(memos_list)} memo(s) to index")

    if memos_list:
        index_memos(memos_list, rag, BATCH_SIZE, OLLAMA_URL, EMBED_MODEL)
    else:
        log.info("Nothing to index.")

    memos.close()
    rag.close()


@retry(stop=stop_after_attempt(10), wait=wait_exponential(min=10, max=300))
def do_build_with_retry():
    log.info("Running rebuild...")
    do_build()


def start_build_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        do_build,
        "interval",
        hours=1,
        next_run_time=datetime.now() + timedelta(minutes=2),
    )
    scheduler.start()
