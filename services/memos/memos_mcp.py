#!/usr/bin/env python3

import logging
from fastmcp import FastMCP

import build
import search

# ── Config ────────────────────────────────────────────────────────────────────

HOST = "0.0.0.0"
PORT = 8000

# ── Setup ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

mcp = FastMCP(
    name="memos-rag-mcp",
)


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
    top_k = min(top_k, 20)
    results = search.search(query, top_k=top_k)

    if not results:
        return "No journal entries found matching that query."

    entries = []
    for r in results:
        if r["score"] < 0.3:
            continue
        from datetime import datetime

        date = datetime.fromtimestamp(r["created_ts"]).strftime("%Y-%m-%d")
        entries.append(f"[{date}] (score: {r['score']:.2f})\n{r['content']}")

    if not entries:
        return "No sufficiently relevant journal entries found."

    return "\n\n---\n\n".join(entries)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build.start_build_scheduler()
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
