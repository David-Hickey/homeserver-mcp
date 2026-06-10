#!/usr/bin/env python3

#!/usr/bin/env python3
"""
build_rag.py — Build and incrementally update a RAG index for memos.

Usage:
    python build_rag.py --full        # Full reindex (first run)
    python build_rag.py               # Incremental update only

Paths (edit to match your setup):
    MEMOS_DB  = /srv/ssd/memos/data/memos_prod.db
    RAG_DB    = /srv/ssd/memos/rag/rag.db
    OLLAMA_URL = http://localhost:11434
"""

import sqlite3
import requests
import struct
import argparse
import logging
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

MEMOS_DB   = "/srv/ssd/memos/data/memos_prod.db"
RAG_DB     = "/srv/ssd/memos/rag/rag.db"
OLLAMA_URL = "http://localhost:11434"

# nomic-embed-text produces 768-dim vectors and is lightweight.
# Pull it once with: ollama pull nomic-embed-text
EMBED_MODEL   = "nomic-embed-text"
EMBED_DIM     = 768

BATCH_SIZE    = 32      # Memos to embed per Ollama call (adjust for your RAM)
WATERMARK_KEY = "last_indexed_ts"

REWRITE_MODEL = "qwen2.5:7b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Vector helpers (sqlite-vec stores raw IEEE-754 blobs) ─────────────────────

def encode_vector(floats: list[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)

def decode_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call Ollama's /api/embed endpoint for a batch of texts."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]

# ── RAG DB setup ──────────────────────────────────────────────────────────────

def init_rag_db(rag: sqlite3.Connection):
    """Create tables if they don't exist. Uses a plain BLOB column for vectors
    since sqlite-vec may not be available; swap the similarity query if you
    install it as an extension."""
    rag.executescript(f"""
        CREATE TABLE IF NOT EXISTS memo_embeddings (
            memo_id     INTEGER PRIMARY KEY,
            uid         TEXT NOT NULL,
            created_ts  BIGINT NOT NULL,
            updated_ts  BIGINT NOT NULL,
            content     TEXT NOT NULL,
            embedding   BLOB NOT NULL          -- {EMBED_DIM} × float32 (IEEE-754)
        );

        CREATE INDEX IF NOT EXISTS idx_me_updated ON memo_embeddings(updated_ts);

        -- Watermark table so incremental runs know where to resume
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    rag.commit()

# ── Watermark ─────────────────────────────────────────────────────────────────

def get_watermark(rag: sqlite3.Connection) -> int:
    row = rag.execute(
        "SELECT value FROM meta WHERE key = ?", (WATERMARK_KEY,)
    ).fetchone()
    return int(row[0]) if row else 0

def save_watermark(rag: sqlite3.Connection, ts: int):
    rag.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (WATERMARK_KEY, str(ts)),
    )
    rag.commit()

# ── Core indexing ─────────────────────────────────────────────────────────────

def fetch_memos_since(memos: sqlite3.Connection, since_ts: int) -> list[dict]:
    rows = memos.execute("""
        SELECT id, uid, created_ts, updated_ts, content
        FROM   memo
        WHERE  updated_ts > ?
          AND  row_status = 'NORMAL'
          AND  visibility != 'ARCHIVED'
        ORDER  BY updated_ts ASC
    """, (since_ts,)).fetchall()

    return [
        {"id": r[0], "uid": r[1], "created_ts": r[2], "updated_ts": r[3], "content": r[4]}
        for r in rows
    ]

def index_memos(memos_list: list[dict], rag: sqlite3.Connection):
    """Embed and upsert a list of memo dicts."""
    if not memos_list:
        return

    total = len(memos_list)
    log.info(f"Indexing {total} memo(s)…")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = memos_list[batch_start : batch_start + BATCH_SIZE]
        texts = [m["content"] for m in batch]

        log.info(f"  Embedding batch {batch_start}–{batch_start + len(batch) - 1}")
        embeddings = embed_texts(texts)

        rag.executemany("""
            INSERT OR REPLACE INTO memo_embeddings
                (memo_id, uid, created_ts, updated_ts, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (
                m["id"], m["uid"], m["created_ts"], m["updated_ts"],
                m["content"], encode_vector(emb),
            )
            for m, emb in zip(batch, embeddings)
        ])
        rag.commit()

    log.info("Batch done.")

# ── Similarity search (pure-Python cosine, no extension needed) ───────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def rewrite_query(query: str) -> list[str]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": REWRITE_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at search query expansion. Expand the given query with synonyms and alternate phrasings. Output exactly 4 lines, one per line, no numbering, no bullets, nothing else."
                },
                {
                    "role": "user",
                    "content": f"Expand this query for better search: {query}"
                }
            ]
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]

    rewrites = [line.strip() for line in raw.splitlines() if line.strip()]

    if not rewrites:
        log.warning(f"Query rewriting produced no output, falling back to original")
        return [query]

    log.debug(f"Query rewrites: {rewrites}")
    return [query] + rewrites


def average_vectors(vecs: list[list[float]]) -> list[float]:
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]


def search(query: str, rag: sqlite3.Connection, top_k: int = 5) -> list[dict]:
    queries = rewrite_query(query)
    print(f"Searching with {len(queries)} queries: {queries}")  # helpful for debugging

    embeddings = embed_texts(queries)
    query_vec = average_vectors(embeddings)

    rows = rag.execute(
        "SELECT memo_id, uid, created_ts, content, embedding FROM memo_embeddings"
    ).fetchall()

    scored = [
        {
            "memo_id":    r[0],
            "uid":        r[1],
            "created_ts": r[2],
            "content":    r[3],
            "score":      cosine_similarity(query_vec, decode_vector(r[4])),
        }
        for r in rows
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Force a full reindex (ignores watermark)")
    args = parser.parse_args()

    # Ensure RAG directory exists
    Path(RAG_DB).parent.mkdir(parents=True, exist_ok=True)

    memos = sqlite3.connect(f"file:{MEMOS_DB}?mode=ro", uri=True)  # read-only
    rag   = sqlite3.connect(RAG_DB)

    init_rag_db(rag)

    since = 0 if args.full else get_watermark(rag)
    since_human = datetime.fromtimestamp(since).isoformat() if since else "beginning"
    log.info(f"Fetching memos updated since {since_human}")

    memos_list = fetch_memos_since(memos, since)
    log.info(f"Found {len(memos_list)} memo(s) to index")

    if memos_list:
        index_memos(memos_list, rag)
        new_watermark = max(m["updated_ts"] for m in memos_list)
        save_watermark(rag, new_watermark)
        log.info(f"Watermark updated to {datetime.fromtimestamp(new_watermark).isoformat()}")
    else:
        log.info("Nothing to index.")

    memos.close()
    rag.close()


if __name__ == "__main__":
    main()

