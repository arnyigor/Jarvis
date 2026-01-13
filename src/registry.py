# registry.py
from typing import Dict

from src.tool import Tool


class ToolRegistry:
    def __init__(self):
        self._registry: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("Только подклассы Tool могут быть зарегистрированы")
        self._registry[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise ValueError(f"Инструмент '{name}' не найден") from exc
