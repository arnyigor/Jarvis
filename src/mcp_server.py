#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mcp_server.py

Runs a FastMCP server that exposes web-search tools powered by jarvis_search_v2 AND Yandex Selenium.
Implements robust, cross-platform logging (no external dependencies), structured error handling,
and correlation IDs for distributed tracing.
"""

from __future__ import annotations

import os
import sys
import uuid
import logging
import asyncio
import contextlib
from datetime import datetime
from logging import DEBUG
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Logging Configuration — Cross-Platform ISO 8601 with Microsecond Precision
# --------------------------------------------------------------------------- #

LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "DEBUG").upper()

class ISO8601Formatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.isoformat(timespec='microseconds') + 'Z'

MSG_FMT = "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
formatter = ISO8601Formatter(fmt=MSG_FMT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "mcp_debug.log")

file_handler = RotatingFileHandler(
    filename=LOG_FILE_PATH,
    maxBytes=5_000_000,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stderr) # ВАЖНО: Логи только в stderr, stdout занят MCP
console_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger: logging.Logger = root_logger.getChild(__name__)

# --------------------------------------------------------------------------- #
# FastMCP Imports
# --------------------------------------------------------------------------- #

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    logger.critical("Failed to import FastMCP. Error: %s", exc, exc_info=True)
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Search Module Imports
# --------------------------------------------------------------------------- #

# 1. Jarvis / SearXNG
try:
    from jarvis_search import smart_search, SearchConfig
except ImportError:
    logger.warning("jarvis_search module not found. 'search_web' tool will fail if called.")

# 2. Yandex Selenium Scraper
try:
    # Предполагаем, что файл называется yandex_search.py
    from search_links_cli import get_links_from_yandex
except ImportError:
    logger.warning("yandex_search module not found. 'search_yandex_selenium' tool will fail if called.")

# --------------------------------------------------------------------------- #
# Helper: Redirect Stdout Protection
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def protect_stdout():
    """
    Context manager to prevent external libraries (like Selenium/Drivers)
    from writing to stdout, which would break the MCP JSON-RPC protocol.
    Redirects stdout to stderr temporarily.
    """
    original_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        yield
    finally:
        sys.stdout = original_stdout

# --------------------------------------------------------------------------- #
# Tool Definition
# --------------------------------------------------------------------------- #

mcp = FastMCP("JarvisSearch")

@mcp.tool()
async def search_web(query: str) -> str:
    """
    [FAST] Execute a web search through SearXNG (Jarvis).
    Best for quick lookups, facts, and API-friendly scraping.
    """
    corr_id = uuid.uuid4().hex
    logger.info("search_web called", extra={"query": query, "corr_id": corr_id})

    config = SearchConfig(base_url="http://localhost:8080", max_concurrent=3, max_retries=2)

    try:
        results = await smart_search([query], max_sources=10, config=config)
    except Exception as exc:
        logger.exception("smart_search failed", extra={"corr_id": corr_id})
        return f"❌ SearXNG Error. ID: {corr_id}"

    if not results:
        return "Поиск SearXNG не дал результатов."

    output_lines = []
    for result in results:
        output_lines.append(
            f"Title: {result.get('title')}\nURL: {result.get('url')}\nSnippet: {result.get('content')}\n---"
        )
    return "\n".join(output_lines)


@mcp.tool()
async def search_yandex_selenium(query: str ) -> str:
    """
    [SLOW/DEEP] Execute a real browser search via Yandex using Selenium.

    Use this tool ONLY when:
    1. 'search_web' (SearXNG) failed or returned irrelevant results.
    2. You need specific Yandex RU results that other engines miss.
    3. The target site requires a real browser fingerprint.

    WARNING: This is slower (5-10s) than search_web.

    Parameters
    ----------
    query : str
        Search query.
    """
    corr_id = uuid.uuid4().hex
    logger.info("search_yandex_selenium called", extra={"query": query, "headless": False, "corr_id": corr_id})

    try:
        # Selenium is blocking, so we run it in a separate thread to keep MCP responsive
        # We also protect stdout so the driver doesn't break the MCP pipe

        def safe_selenium_execution():
            with protect_stdout():
                # get_links_from_yandex ожидает список запросов
                return get_links_from_yandex([query], headless=False, limit=5, global_unique=False)

        # Выполняем в потоке
        results_dict = await asyncio.to_thread(safe_selenium_execution)

        # Результат приходит в виде {"query": ["url1", "url2"]}
        urls = results_dict.get(query, [])

    except Exception as exc:
        logger.exception(
            "Selenium search failed",
            extra={"query": query, "corr_id": corr_id},
        )
        return f"❌ Ошибка Yandex Selenium. См. логи. [ID: {corr_id}]"

    if not urls:
        logger.warning("No Yandex results found", extra={"query": query, "corr_id": corr_id})
        return "Yandex поиск не дал результатов (возможно, капча или пустая выдача)."

    # Форматирование вывода
    output_lines = [f"Found {len(urls)} links via Yandex for: '{query}'\n"]
    for i, url in enumerate(urls, 1):
        output_lines.append(f"{i}. {url}")

    # Подсказка для LLM
    output_lines.append("\n(Note: These are direct URLs. Use 'fetch_url' or similar tool to read their content if needed.)")

    logger.debug(
        "search_yandex_selenium completed",
        extra={"url_count": len(urls), "corr_id": corr_id},
    )
    return "\n".join(output_lines)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        # ВАЖНО: убеждаемся, что логи идут в stderr перед запуском
        # FastMCP использует stdout для общения с клиентом (Claude/LM Studio)
        logger.info("Starting FastMCP server 'JarvisSearch' with Yandex Support")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as exc:
        logger.critical("Unhandled exception", exc_info=True)
        sys.exit(1)
