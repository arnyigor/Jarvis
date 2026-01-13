# tools/python_exec.py
import asyncio
import logging
import os
import tempfile
import textwrap

from src.tool import Tool

logger = logging.getLogger(__name__)


class PythonExecTool(Tool):
    name = "python_exec"

    async def call(self, params: dict) -> dict:
        code = params.get("code")
        timeout = int(params.get("timeout", 5))
        if not code:
            return {"error": "Missing 'code'"}

        # Sandbox – write to temp file, run with restricted env
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".py", delete=False
            ) as tmp:
                tmp.write(textwrap.dedent(code))
                tmp_path = tmp.name

            env = os.environ.copy()
            env.update({
                "PYTHONPATH": "",  # no imports from host
                "PATH": "/usr/bin:/bin",
            })
            process = await asyncio.create_subprocess_exec(
                "python3", "-u", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,  # prevent signal propagation
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                return {"error": f"Execution timed out after {timeout}s"}

            result = {
                "stdout": stdout.decode()[:2000],
                "stderr": stderr.decode()[:2000],
                "exit_code": process.returncode,
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {"result": result}
