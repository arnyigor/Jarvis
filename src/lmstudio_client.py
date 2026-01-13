# src/lmstudio_client.py
import json
import re
from typing import List, Dict, Any

import lmstudio as lms
from lmstudio._sdk_models import Temperature
import logging  # новый импорт

from .config import LMSTUDIO_MODEL_NAME, SearchResult, TOP_K, MAX_RESULTS_FOR_LLM

# Локальный логгер
logger = logging.getLogger(__name__)

class LMStudioClient:
    """Клиент для LLM через lmstudio SDK."""

    def __init__(self, model_name: str = LMSTUDIO_MODEL_NAME):
        self.model = lms.llm(model_name)
        logger.info(f"LM Studio LLM инициализирован: {model_name}")

    def _extract_final_response(self, text: str) -> str:
        """Извлекает ответ после маркера <|channel|>final<|message|>"""
        separator = "<|end|><|start|>assistant<|channel|>final<|message|>"
        if separator in text:
            return text.split(separator, 1)[-1].strip()
        return text

    def _run_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Стриминг чата с фильтрацией"""
        chat = lms.Chat(system_prompt)
        chat.add_user_message(user_prompt)

        # Temp 0.1 для детерминизма
        config = lms.LlmPredictionConfig(temperature=Temperature(0.1))

        try:
            stream = self.model.respond_stream(chat, config=config)
            full_content = ""

            # Если нужно видеть процесс в консоли (для дебага)
            # print("🤖 LLM Stream: ", end="", flush=True)

            for chunk in stream:
                if chunk.content:
                    # print(chunk.content, end="", flush=True)
                    full_content += chunk.content

            # print()
            return self._extract_final_response(full_content)

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return ""

    def _extract_json_ids(self, text: str) -> List[int]:
        """Надежный парсер JSON-списка из текста."""

        # 1. Очистка от markdown (если есть)
        if "```" in text:
            text = text.replace("``````", "")

        # 2. Ищем паттерн (ВНЕ блока if!)
        matches = re.findall(r'\[([\d,\s]+)\]', text)

        candidates = []
        for match in matches:
            try:
                json_str = f"[{match}]"
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                    candidates.append(parsed)
            except json.JSONDecodeError:
                continue

        # Теперь candidates точно существует (пустой список или найденные)
        return candidates[-1] if candidates else []

    def select_relevant_ids(self, query: str, results: List[SearchResult], top_k: int = TOP_K) -> List[int]:
        """Фильтрация результатов через LLM."""
        if not results:
            return []

        candidates = results[:MAX_RESULTS_FOR_LLM]
        items_text = ""
        for r in candidates:
            items_text += f"[{r.id}] {r.title} | {r.snippet[:120]}...\n"

        system_prompt = (
            "You are a precise search result filter. "
            "Your task is to select only results that are truly helpful for answering the user query. "
            "Do not explain. Do not analyze. "
            "Return output strictly as a JSON list of integers."
        )
        user_prompt = f"QUERY: {query}\n\nCANDIDATES:\n{items_text}\nTASK: Select up to {top_k} most relevant IDs.\nOUTPUT FORMAT: JSON list.\nYOUR OUTPUT:"

        raw_response = self._run_chat(system_prompt, user_prompt)
        ids = self._extract_json_ids(raw_response)

        # Валидация ID
        valid_ids = {r.id for r in candidates}
        filtered = [i for i in ids if i in valid_ids]

        if not filtered and ids:
            logger.warning(f"LLM вернула несуществующие ID: {ids}")

        return filtered

    def answer_with_context(self, query: str, context_blocks: List[Dict[str, Any]]) -> str:
        """Генерация финального ответа по источникам."""
        sources_text = ""
        for i, doc in enumerate(context_blocks, 1):
            sources_text += f"---\nSource {i}: {doc['title']}\nURL: {doc['url']}\n\n{doc['text']}\n\n"

        system_prompt = (
            "You are Jarvis – a privacy-first multimodal assistant.\n"
            "You have access to the following tools:\n"
            "- python_exec: Execute Python code safely in sandbox. Use only for math, data manipulation, or automation.\n"
            "When you need to perform calculations, analyze data, or run code, use 'python_exec' with a JSON object containing 'code'.\n"
            "Always think step-by-step before using a tool."
        )
        user_prompt = f"USER QUESTION:\n{query}\n\nSOURCES:\n{sources_text}\n***\nProvide a detailed answer in Russian."

        return self._run_chat(system_prompt, user_prompt)
