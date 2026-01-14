#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mcp_server.py

FastMCP server:
- search_yandex_selenium (Selenium/Yandex)
- execute_python_code (secure-ish sandbox executor)

IMPORTANT:
- MCP stdio transport uses stdout for JSON-RPC. Any accidental stdout breaks the protocol.
- Keep logs strictly on stderr.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import sys
import uuid
import ast
import time

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List

# --------------------------------------------------------------------------- #
# Logging Configuration — Cross-Platform ISO 8601 with Microsecond Precision
# --------------------------------------------------------------------------- #

LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "DEBUG").upper()


class ISO8601Formatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        # Use UTC with explicit tz; keep 'Z'
        ct = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.isoformat(timespec="microseconds").replace("+00:00", "Z")


class EnsureExtrasFilter(logging.Filter):
    """Ensure optional LogRecord attributes exist so formatter won't crash."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "corr_id"):
            record.corr_id = "-"
        if not hasattr(record, "query"):
            record.query = "-"
        return True


# Include corr_id in logs (safe due to filter above)
MSG_FMT = "%(asctime)s %(levelname)-8s [corr_id=%(corr_id)s] [%(filename)s:%(lineno)d] %(message)s"
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
file_handler.addFilter(EnsureExtrasFilter())

console_handler = logging.StreamHandler(sys.stderr)  # IMPORTANT: logs ONLY to stderr
console_handler.setFormatter(formatter)
console_handler.addFilter(EnsureExtrasFilter())

root_logger = logging.getLogger()
root_logger.handlers.clear()

numeric_level = getattr(logging, LOG_LEVEL, logging.DEBUG)
root_logger.setLevel(numeric_level)
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

try:
    from jarvis_search import smart_search, SearchConfig
except ImportError:
    logger.warning("jarvis_search module not found. 'search_web' tool will fail if called.")

try:
    from search_links_cli import get_links_from_yandex
except ImportError:
    logger.warning("search_links_cli module not found. 'search_yandex_selenium' tool will fail if called.")

# --------------------------------------------------------------------------- #
# Helper: Redirect Stdout Protection
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def protect_stdout():
    """
    Prevent external libs from writing to stdout (MCP uses stdout for JSON-RPC).
    Redirect stdout to stderr temporarily.
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
async def search_yandex_selenium(query: str) -> str:
    """
    [SLOW/DEEP] Execute a real browser search via Yandex using Selenium.

    Use ONLY when:
    1) search_web (SearXNG) failed/irrelevant,
    2) RU-specific Yandex results needed,
    3) target requires real browser fingerprint.

    WARNING: slower (5–10s).
    """
    corr_id = uuid.uuid4().hex
    logger.info("search_yandex_selenium called", extra={"query": query, "headless": False, "corr_id": corr_id})

    try:
        def safe_selenium_execution():
            with protect_stdout():
                return get_links_from_yandex([query], headless=False, limit=5, global_unique=False)

        results_dict = await asyncio.to_thread(safe_selenium_execution)
        urls = results_dict.get(query, [])

    except Exception:
        logger.exception("Selenium search failed", extra={"query": query, "corr_id": corr_id})
        return f"❌ Ошибка Yandex Selenium. См. логи. [ID: {corr_id}]"

    if not urls:
        logger.warning("No Yandex results found", extra={"query": query, "corr_id": corr_id})
        return "Yandex поиск не дал результатов (возможно, капча или пустая выдача)."

    output_lines = [f"Found {len(urls)} links via Yandex for: '{query}'\n"]
    for i, url in enumerate(urls, 1):
        output_lines.append(f"{i}. {url}")

    output_lines.append("\n(Note: These are direct URLs. Use 'fetch_url' or similar tool to read their content if needed.)")

    logger.debug("search_yandex_selenium completed", extra={"url_count": len(urls), "corr_id": corr_id})
    return "\n".join(output_lines)

# --------------------------------------------------------------------------- #
# 3) execute_python_code – sandbox executor
# --------------------------------------------------------------------------- #

def _safe_exec(code: str) -> Dict[str, Any]:
    """
    Execute code in a restricted globals/builtins environment.

    Returns:
      output: value of _result_ if set (or expression result),
      stdout/stderr: captured outputs,
      error: exception string if any
    """

    # Allowed imports (stdlib + optionally present external)
    allowed_modules = {
        # stdlib
        "math", "threading", "json",
        "urllib", "urllib.parse", "urllib.request",
        "datetime", "time", "os", "sys", "re", "collections", "typing", "io",
        "uuid", "base64", "hashlib", "random", "string", "csv",
        "xml", "xml.etree", "xml.etree.ElementTree",
        "html", "html.parser",
        "unicodedata",
        "functools", "itertools", "operator",
        "ast",
        "contextlib",
        # NOTE: requests is 3rd-party; allow if installed
        "requests",
    }

    # Dynamically allow external deps if installed
    for module in ("numpy", "mpmath"):
        try:
            __import__(module)
            allowed_modules.add(module)
        except ImportError:
            pass

    def _is_allowed(module_name: str) -> bool:
        if module_name in allowed_modules:
            return True
        # allow submodules of an allowed package/module (prefix-based)
        for m in allowed_modules:
            if module_name.startswith(m + "."):
                return True
        # allow base package if explicitly allowed
        base = module_name.split(".", 1)[0]
        return base in allowed_modules

    def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if not _is_allowed(name):
            raise ImportError(f"Module {name!r} is not allowed")
        return __import__(name, globals, locals, fromlist, level)

    # Minimal-but-practical builtins set
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "str": str,
        "sum": sum,
        "print": print,
        "set": set,
        "tuple": tuple,
        "isinstance": isinstance,
        "type": type,
        "Exception": Exception,
        "__import__": _safe_import,
        # keep exec available for normal Python execution in sandbox
        "exec": exec,
    }

    sandbox: Dict[str, Any] = {"__builtins__": safe_builtins}

    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                parsed = ast.parse(code, mode="exec")
                # If it's a single expression (e.g., "2 + 3"), capture value into _result_
                if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Expr):
                    expr_source = code.strip()
                    exec_code = f"_result_ = ({expr_source})"
                else:
                    exec_code = code

                exec(exec_code, sandbox)
            except SyntaxError:
                # Fallback if AST manipulation fails (rare)
                exec(code, sandbox)

        return {
            "output": sandbox.get("_result_", None),
            "stdout": stdout.getvalue().strip(),
            "stderr": stderr.getvalue().strip(),
            "error": None,
        }

    except Exception as exc:
        return {
            "output": None,
            "stdout": stdout.getvalue().strip(),
            "stderr": stderr.getvalue().strip(),
            "error": f"{type(exc).__name__}: {exc}",
        }


@mcp.tool()
def execute_python_code(code: str) -> str:
    """
    Executes Python code in a secure, isolated sandbox.

    Use this tool to perform calculations, data transformation, or logic verification
    that cannot be handled by simple text generation.

    Parameters:
    - code (str): A valid Python script or expression to execute.

    Format Requirement:
    Input MUST be a JSON object with the "code" key.

    Example:
    {
      "code": "import math\\nprint(f'Result: {math.sqrt(25)}')"
    }

    Returns:
    stdout, stderr, and the final expression result (if any).
    """
    # Генерация correlation ID для сквозной трассировки
    corr_id = uuid.uuid4().hex

    # =====================================================================
    # [STAGE 1] Прием запроса
    # =====================================================================
    logger.info(
        "[STAGE 1/5] execute_python_code invoked",
        extra={
            "corr_id": corr_id,
            "code_length": len(code) if code else 0,
            "code_first_80": (code or '')[:80].replace('\n', '\\n'),
        }
    )

    # Защита от пустого кода
    if not code:
        logger.warning(
            "[STAGE 1/5] Empty code received",
            extra={"corr_id": corr_id}
        )
        return "❌ Error: 'code' argument is missing or empty. Usage: {\"code\": \"...\"}"

    # =====================================================================
    # [STAGE 2] Попытка распарсить AST
    # =====================================================================
    try:
        parsed = ast.parse(code, mode="exec")
        is_single_expr = len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Expr)

        logger.debug(
            "[STAGE 2/5] AST parsing successful",
            extra={
                "corr_id": corr_id,
                "ast_body_count": len(parsed.body),
                "is_single_expression": is_single_expr
            }
        )
    except SyntaxError as exc:
        logger.error(
            "[STAGE 2/5] AST parsing failed (SyntaxError)",
            extra={
                "corr_id": corr_id,
                "error_line": exc.lineno,
                "error_offset": exc.offset,
                "error_text": exc.text
            },
            exc_info=True  # Полный traceback
        )
        return f"❌ Syntax Error at line {exc.lineno}: {exc.msg}"

    # =====================================================================
    # [STAGE 3] Выполнение в sandbox
    # =====================================================================
    start_time = time.perf_counter()
    logger.info(
        "[STAGE 3/5] Calling _safe_exec",
        extra={"corr_id": corr_id}
    )

    res = _safe_exec(code)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # =====================================================================
    # [STAGE 4] Анализ результата выполнения
    # =====================================================================
    logger.info(
        "[STAGE 4/5] Execution completed",
        extra={
            "corr_id": corr_id,
            "execution_time_ms": round(elapsed_ms, 2),
            "has_error": res["error"] is not None,
            "error_type": res["error"].split(':')[0] if res["error"] else None,
            "stdout_length": len(res["stdout"]),
            "stderr_length": len(res["stderr"]),
            "has_output": res["output"] is not None,
            "output_type": type(res["output"]).__name__ if res["output"] is not None else None,
        }
    )

    # Если была ошибка в песочнице, логируем подробно
    if res["error"]:
        logger.error(
            "[STAGE 4/5] Sandbox execution error",
            extra={
                "corr_id": corr_id,
                "error_full": res["error"],
                "stderr_content": res["stderr"][:500]  # Первые 500 символов stderr
            }
        )

    # =====================================================================
    # [STAGE 5] Формирование ответа клиенту
    # =====================================================================
    parts: List[str] = []

    if res["stdout"]:
        parts.append(f"[stdout]\n{res['stdout']}")
    if res["stderr"]:
        parts.append(f"[stderr]\n{res['stderr']}")
    if res["error"]:
        parts.append(f"[error] {res['error']}")
    if res["output"] is not None:
        parts.append(f"[result] {res['output']}")

    response = "\n\n".join(parts) or "✅ Code executed – no output."

    logger.info(
        "[STAGE 5/5] Response prepared",
        extra={
            "corr_id": corr_id,
            "response_length": len(response),
            "response_first_80": response[:80].replace('\n', '\\n')
        }
    )

    return response

# --------------------------------------------------------------------------- #
# Entrypoint with Warmup
# --------------------------------------------------------------------------- #

def warmup_sandbox():
    """Preload common modules to avoid first-call latency."""
    logger.info("Warming up sandbox...")
    warmup_code = """
import math
import json
import datetime
import re
import os
import io
import uuid
x = math.sqrt(16)
"""
    try:
        _safe_exec(warmup_code)
        logger.info("Sandbox warmup completed")
    except Exception as exc:
        logger.warning(f"Warmup failed (non-critical): {exc}")

if __name__ == "__main__":
    try:
        logger.info("Starting FastMCP server 'JarvisSearch' with Yandex Support")
        warmup_sandbox()
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception:
        logger.critical("Unhandled exception", exc_info=True)
        sys.exit(1)
