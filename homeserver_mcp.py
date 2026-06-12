#!/usr/bin/env python3

import os
import logging
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from memos_rag import build as memos_build, search as memos_search

# ── Config ────────────────────────────────────────────────────────────────────

HOST = "0.0.0.0"
PORT = int(os.getenv("MCP_PORT", "8000"))

# ── Setup ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


auth_token = os.environ.get("MCP_AUTH_TOKEN")

mcp = FastMCP(
    name="homeserver-mcp",
    auth=(
        StaticTokenVerifier(
            tokens={auth_token: {"client_id": "openwebui", "scopes": ["read"]}}
        )
        if auth_token
        else None
    ),
)

if auth_token:
    logging.info(f"AUTH_TOKEN is set to {auth_token[:5] + ('*' * (len(auth_token) - 5))}, using StaticTokenVerifier")
else:
    logging.warning("No AUTH_TOKEN set, running without authentication!")

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

    results = memos_search.search(query, top_k=top_k)

    if not results:
        return "No journal entries found matching that query."

    # Format results for the LLM — include date and content, skip low scores
    entries = []
    for r in results:
        if r["score"] < 0.3:  # tune this threshold to taste
            continue
        from datetime import datetime

        date = datetime.fromtimestamp(r["created_ts"]).strftime("%Y-%m-%d")
        entries.append(f"[{date}] (score: {r['score']:.2f})\n{r['content']}")

    if not entries:
        return "No sufficiently relevant journal entries found."

    return "\n\n---\n\n".join(entries)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    memos_build.start_build_scheduler()
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
