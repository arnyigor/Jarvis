# smart_search_pipeline.py
# Система: JarvisSearchV2 (SearXNG) → LLM-фильтр (GPT-OSS 20B) → Async Read → LLM Answer.

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .config import LMSTUDIO_MODEL_NAME, TOP_K, MAX_RESULTS_FOR_LLM, SearchResult
from .lmstudio_client import LMStudioClient
import logging
import aiohttp
import trafilatura

# ИМПОРТИРУЕМ ВАШ ПРОДВИНУТЫЙ КЛИЕНТ
try:
    from jarvis_search import smart_search, SearchConfig
except ImportError:
    print("❌ Ошибка: Не найден файл jarvis_search.py. Убедитесь, что он в той же папке.")
    exit(1)

# -----------------------
# Конфигурация
# -----------------------

LMSTUDIO_MODEL_NAME = "gpt-oss-20b"
MAX_RESULTS_FOR_LLM = 20  # Максимум кандидатов для LLM
TOP_K = 3  # Сколько выбрать для чтения
MAX_CHARS_PER_DOC = 4000  # Обрезка текста
FETCH_TIMEOUT = 5.0  # Таймаут чтения страницы

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("SmartSearchPipeline")


@dataclass
class SearchResult:
    """Единый формат результата для пайплайна"""
    id: int
    title: str
    url: str
    snippet: str



# -----------------------
# Чтение веб-страниц
# -----------------------

class WebPageReader:
    """Асинхронный загрузчик контента."""

    def __init__(self, timeout: float = FETCH_TIMEOUT, max_chars: int = MAX_CHARS_PER_DOC):
        self.timeout = timeout
        self.max_chars = max_chars
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
        }

    async def _fetch_html(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            async with session.get(url, headers=self._headers, timeout=self.timeout) as resp:
                if resp.status != 200:
                    return None
                return await resp.text(errors='ignore')
        except Exception:
            return None

    def _extract_text(self, html: str) -> Optional[str]:
        try:
            text = trafilatura.extract(html, include_comments=False, include_tables=False, include_links=False)
            if not text:
                return None
            text = " ".join(text.split())
            if len(text) > self.max_chars:
                text = text[:self.max_chars] + "... [truncated]"
            return text
        except Exception:
            return None

    async def fetch_and_extract_many(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Параллельное чтение."""
        extracted_docs: List[Dict[str, Any]] = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_html(session, r.url) for r in results]
            html_pages = await asyncio.gather(*tasks, return_exceptions=True)

        for r, html in zip(results, html_pages):
            if isinstance(html, Exception) or not html:
                continue
            text = self._extract_text(html)
            if text and len(text) > 100:
                extracted_docs.append({"url": r.url, "title": r.title, "text": text})

        return extracted_docs


# -----------------------
# Основной Пайплайн
# -----------------------

class SmartSearchPipeline:
    def __init__(self):
        # Используем ваш JarvisSearchV2 конфиг
        self.search_config = SearchConfig(
            base_url="http://localhost:8080",
            max_concurrent=5,
            cache_ttl=3600
        )
        self.llm = LMStudioClient()
        self.reader = WebPageReader()

    async def answer_question(self, query: str) -> str:
        logger.info(f"🚀 Старт обработки: {query}")

        # 1. ПОИСК (Используем ваш jarvis_search_v2)
        # smart_search принимает список запросов, поэтому оборачиваем в list
        # Она сама делает rate limit, cache, deduplication
        raw_results_dicts = await smart_search(
            queries=[query],
            max_sources=20,  # Берем с запасом для фильтрации
            config=self.search_config
        )

        if not raw_results_dicts:
            return "Поиск не дал результатов."

        # Конвертируем словари в объекты SearchResult
        search_results = []
        for idx, r in enumerate(raw_results_dicts):
            search_results.append(SearchResult(
                id=idx,
                title=r.get('title', 'No title'),
                url=r.get('url', ''),
                snippet=r.get('content', '')
            ))

        logger.info(f"🔍 SearXNG вернул {len(search_results)} результатов.")

        # 2. ФИЛЬТРАЦИЯ (LLM)
        logger.info("🧠 LLM анализирует релевантность...")
        relevant_ids = await asyncio.to_thread(
            self.llm.select_relevant_ids, query, search_results, TOP_K
        )

        if not relevant_ids:
            return "LLM не нашла релевантных ссылок среди результатов поиска."

        selected = [r for r in search_results if r.id in relevant_ids]
        logger.info(f"✅ Выбрано {len(selected)} ссылок: {[r.url for r in selected]}")

        # 3. ЧТЕНИЕ (Async)
        logger.info("🌐 Скачивание контента...")
        docs = await self.reader.fetch_and_extract_many(selected)

        if not docs:
            return "Не удалось скачать текст выбранных статей."

        # 4. ОТВЕТ (LLM)
        logger.info("📝 Генерация финального ответа...")
        answer = await asyncio.to_thread(
            self.llm.answer_with_context, query, docs
        )

        return answer


# -----------------------
# CLI Точка входа
# -----------------------

if __name__ == "__main__":
    async def main():
        pipeline = SmartSearchPipeline()
        print("\n🤖 Jarvis Search Pipeline V2 Ready.")
        print("Используется jarvis_search_v2 для поиска и GPT-OSS 20B для анализа.\n")

        while True:
            try:
                q = input("Вопрос (или 'exit'): ")
                if q.lower() in ['exit', 'quit']:
                    break

                print("-" * 50)
                answer = await pipeline.answer_question(q)
                print("\n=== ОТВЕТ ===\n")
                print(answer)
                print("-" * 50 + "\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Ошибка: {e}")


    asyncio.run(main())
