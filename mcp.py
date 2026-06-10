#!/usr/bin/env python3
"""
mcp.py — MCP server exposing a journal_rag tool.

Install:
    pip install fastmcp

Run:
    python mcp.py

The server listens on HTTP/SSE at http://0.0.0.0:8000/sse
"""

import sqlite3
import logging
from fastmcp import FastMCP
import memos_rag  # your existing module with search()

# ── Config ────────────────────────────────────────────────────────────────────

RAG_DB = "/srv/ssd/memos/rag/rag.db"
HOST   = "0.0.0.0"
PORT   = 8000

# ── Setup ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

mcp = FastMCP(
    name="memos-journal",
    instructions="Search the user's personal journal using semantic search.",
)

# ── Tool ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def journal_rag(query: str, top_k: int = 5) -> str:
    """
    Search the personal journal for entries semantically related to a query.

    Use this when the user asks about past experiences, feelings, events,
    places they've visited, people they've mentioned, or anything that might
    appear in a personal journal. Accepts natural language queries.

    Args:
        query: What to search for, in natural language.
        top_k: Number of results to return (default 5, max 20).
    """
    top_k = min(top_k, 20)  # sanity cap

    conn = sqlite3.connect(RAG_DB)
    try:
        results = memos_rag.search(query, conn, top_k=top_k)
    finally:
        conn.close()

    if not results:
        return "No journal entries found matching that query."

    # Format results for the LLM — include date and content, skip low scores
    entries = []
    for r in results:
        if r["score"] < 0.3:   # tune this threshold to taste
            continue
        from datetime import datetime
        date = datetime.fromtimestamp(r["created_ts"]).strftime("%Y-%m-%d")
        entries.append(f"[{date}] (score: {r['score']:.2f})\n{r['content']}")

    if not entries:
        return "No sufficiently relevant journal entries found."

    return "\n\n---\n\n".join(entries)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="sse", host=HOST, port=PORT)
