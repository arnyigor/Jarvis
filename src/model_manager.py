import logging
from typing import List, Optional

import lmstudio as lms
from src.lmstudio_client import LMStudioClient
from .config import MAX_RESULTS_FOR_LLM
# Prefer the SDK's own exception if available; otherwise fall back to Exception.
try:
    from lmstudio.exceptions import LmStudioError  # type: ignore[assignment]
except ImportError:  # pragma: no cover – older SDK releases
    LmStudioError = Exception  # fallback


logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, default_model: str):
        self.default_model = default_model
        self.llm: Optional[lms.LLM] = None
        self.current_model_id: Optional[str] = None
        # Композиция: ModelManager "has-a" LMStudioClient
        self._client: Optional[LMStudioClient] = None

    @property
    def client(self) -> LMStudioClient:
        """Lazy initialization of client."""
        if self._client is None:
            self._client = LMStudioClient(self.current_model_id or self.default_model)
        return self._client

    async def load(self, model_id: str) -> None:
        self.llm.unload()
        self.llm = lms.llm(model_id)
        self.current_model_id = model_id
        # Обновляем клиент при смене модели
        self._client = LMStudioClient(model_id)
