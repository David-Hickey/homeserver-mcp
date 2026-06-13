#!/usr/bin/env python3

import os
import logging
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.tools import Tool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import asyncio

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
    logging.info(
        f"AUTH_TOKEN is set to {auth_token[:5] + ('*' * (len(auth_token) - 5))}, using StaticTokenVerifier"
    )
else:
    logging.warning("No AUTH_TOKEN set, running without authentication!")


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.getLogger(
        f"mount.{retry_state.args[0]}"
    ).info(f"Attempt {retry_state.attempt_number} failed — retrying in 2s"),
)
async def try_mount_service(entry: str):
    name, _, port = entry.partition(":")
    name = name.strip()
    port = port.strip()
    url = f"http://{name}:{port}/mcp"
    svc_log = logging.getLogger(f"mount.{name}")

    client = Client(url)
    tools: list[Tool] = []
    async with client:
        await client.ping()
        tools = await client.list_tools()
    mcp.mount(client, namespace=name.replace("-", "_"), as_proxy=True)
    svc_log.info(f"Mounted successfully from {url}")

    if tools:
        svc_log.info(f"Available tools: {', '.join([tool.name for tool in tools])}")
    else:
        svc_log.warning("No tools available.")


async def mount_services():
    services_env = os.environ.get("MCP_SERVICES", "")
    if not services_env:
        logging.info("No MCP_SERVICES defined, starting with no sub-services")
        return

    entries = [e.strip() for e in services_env.split(",") if e.strip()]
    await asyncio.gather(*[try_mount_service(e) for e in entries])


if __name__ == "__main__":
    asyncio.run(mount_services())
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
