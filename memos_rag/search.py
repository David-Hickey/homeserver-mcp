#!/usr/bin/env python3

import sqlite3
import requests
import logging
from . import environment
from .utils import decode_vector, embed_texts, cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def rewrite_query(query: str, ollama_url: str, rewrite_model: str) -> list[str]:
    resp = requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": rewrite_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at search query expansion. Expand the given query with synonyms and alternate phrasings. Output exactly 4 lines, one per line, no numbering, no bullets, nothing else.",
                },
                {
                    "role": "user",
                    "content": f"Expand this query for better search: {query}",
                },
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]

    rewrites = [line.strip() for line in raw.splitlines() if line.strip()]

    if not rewrites:
        log.warning("Query rewriting produced no output, falling back to original")
        return [query]

    log.debug(f"Query rewrites: {rewrites}")
    return [query] + rewrites


def average_vectors(vecs: list[list[float]]) -> list[float]:
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]


def search(query: str, rag: sqlite3.Connection = None, top_k: int = 5) -> list[dict]:
    OLLAMA_URL = environment.Environment().ollama_url
    REWRITE_MODEL = environment.Environment().rewrite_model
    EMBED_MODEL = environment.Environment().embed_model

    if rag is None:
        rag = sqlite3.connect(environment.Environment().rag_db)
        close_after = True
    else:
        close_after = False

    try:
        queries = rewrite_query(query, OLLAMA_URL, REWRITE_MODEL)
        print(
            f"Searching with {len(queries)} queries: {queries}"
        )  # helpful for debugging

        embeddings = embed_texts(queries, OLLAMA_URL, EMBED_MODEL)
        query_vec = average_vectors(embeddings)

        rows = rag.execute(
            "SELECT memo_id, uid, created_ts, content, embedding FROM memo_embeddings"
        ).fetchall()

        scored = [
            {
                "memo_id": r[0],
                "uid": r[1],
                "created_ts": r[2],
                "content": r[3],
                "score": cosine_similarity(query_vec, decode_vector(r[4])),
            }
            for r in rows
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    finally:
        if close_after:
            rag.close()
