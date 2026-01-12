#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mcp_server.py

Runs a FastMCP server that exposes a web-search tool powered by jarvis_search_v2.
Implements robust, cross-platform logging (no external dependencies), structured error handling,
and correlation IDs for distributed tracing.

Author:   Distinguished Software Architect
License:  MIT
"""

from __future__ import annotations

import os
import sys
import uuid
import logging
from datetime import datetime
from logging import DEBUG
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

# --------------------------------------------------------------------------- #
# Logging Configuration — Cross-Platform ISO 8601 with Microsecond Precision
# --------------------------------------------------------------------------- #

LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "DEBUG").upper()

class ISO8601Formatter(logging.Formatter):
    """
    Custom logging formatter that emits timestamps in ISO 8601 format with microsecond precision.
    Uses datetime.isoformat() instead of time.strftime() to ensure compatibility on Windows.

    Output example: 2025-04-05T16:38:22.123456Z
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """
        Override to use datetime.isoformat() for full ISO 8601 support including microseconds.
        This avoids the %f strftime limitation on Windows.
        """
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            return ct.strftime(datefmt)
        # Use microsecond precision (6 digits) and append 'Z' to indicate UTC
        return ct.isoformat(timespec='microseconds') + 'Z'


# Define message format with correlation ID support
MSG_FMT = "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

# Create formatter instance
formatter = ISO8601Formatter(fmt=MSG_FMT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "mcp_debug.log")
# File handler: rotates at 5MB, keeps 5 backups
file_handler = RotatingFileHandler(
    filename=LOG_FILE_PATH,  # <-- ИЗМЕНЕНИЕ ЗДЕСЬ
    maxBytes=5_000_000,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

# Optional console output — comment out in production
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)  # <-- Remove this line in production

logger: logging.Logger = root_logger.getChild(__name__)

# --------------------------------------------------------------------------- #
# FastMCP Imports — Graceful Failure Handling
# --------------------------------------------------------------------------- #

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except ImportError as exc:
    logger.critical("Failed to import FastMCP. Is 'python-mcp' installed? Error: %s", exc, exc_info=True)
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Search Module Imports — Graceful Failure Handling
# --------------------------------------------------------------------------- #

try:
    from jarvis_search import smart_search, SearchConfig  # type: ignore
except ImportError as exc:
    logger.critical("jarvis_search module not found. Ensure 'jarvis_search.py' is in PYTHONPATH. Error: %s", exc)
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Tool Definition — Web Search via SearXNG
# --------------------------------------------------------------------------- #

mcp = FastMCP("JarvisSearch")


@mcp.tool()
async def search_web(query: str) -> str:
    """
    Execute a web search through SearXNG with caching, deduplication, and result filtering.

    This tool is designed to be called by LLM agents or external services via the MCP protocol.
    It uses an internal async search engine (jarvis_search_v2) backed by a local SearXNG instance.

    Parameters
    ----------
    query : str
        The natural language search query. Must not be empty.
        Example: "latest AI breakthroughs 2025"

    Returns
    -------
    str
        A formatted, human-readable string of up to 10 results.
        Each result includes title, URL, and snippet.
        If no results or an error occurs, returns a user-friendly error message with correlation ID.

    Raises
    ------
    Exception
        Any underlying exception from smart_search is caught and wrapped in a user-safe response.
    """

    # Generate unique correlation ID for tracing across logs and systems
    corr_id = uuid.uuid4().hex

    logger.info("search_web called", extra={"query": query, "corr_id": corr_id})

    config = SearchConfig(
        base_url="http://localhost:8080",
        max_concurrent=3,
        max_retries=2,
    )

    try:
        # smart_search expects a list of queries
        results = await smart_search([query], max_sources=10, config=config)
    except Exception as exc:
        logger.exception(
            "smart_search failed during execution",
            extra={"query": query, "corr_id": corr_id},
        )
        return (
            f"❌ Ошибка при поиске. Попробуйте позже.\n"
            f"[ID: {corr_id}]"
        )

    if not results:
        logger.warning("No search results found", extra={"query": query, "corr_id": corr_id})
        return "Поиск не дал результатов."

    # Build human-readable output
    output_lines = []
    for result in results:
        title = result.get("title", "[Без названия]")
        url = result.get("url", "[Без URL]")
        snippet = result.get("content", "[Нет сниппета]")

        # Sanitize newlines in snippets to avoid breaking output format
        snippet = snippet.replace('\n', ' ').replace('\r', ' ')
        output_lines.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}\n"
            f"---"
        )

    logger.debug(
        "search_web completed successfully",
        extra={"result_count": len(results), "corr_id": corr_id},
    )
    return "\n".join(output_lines)


# --------------------------------------------------------------------------- #
# Entrypoint — Start the MCP Server
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        logger.info("Starting FastMCP server 'JarvisSearch'")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
    except Exception as exc:
        logger.critical("Unhandled exception in server loop", exc_info=True)
        sys.exit(1)
