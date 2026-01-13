# tool.py
from abc import ABC, abstractmethod
from typing import Dict, Any


class Tool(ABC):
    name: str = ""

    @abstractmethod
    async def call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет инструмент и возвращает структурированный JSON."""
