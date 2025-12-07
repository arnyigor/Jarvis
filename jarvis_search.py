#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jarvis_search_v2.py – Production-ready SearXNG client для Jarvis.

Особенности:
- Adaptive rate limiting (умные задержки)
- Exponential backoff retry
- Engine fallback (если Google банит → переключаемся на DDG)
- Result caching (disk-based с UTF-8)
- Health monitoring (отслеживание проблемных движков)
- Domain diversity enforcement
- Query simplification при пустых результатах
- Unicode support (исправлена ошибка encoding)

Usage:
    python jarvis_search_v2.py "query 1" "query 2" "query 3"
    python jarvis_search_v2.py --verbose --max-sources 15 "deep learning"
"""

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

import aiohttp

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class SearchConfig:
    """Конфигурация клиента"""
    base_url: str = "http://localhost:8080"
    timeout: int = 10
    max_retries: int = 3
    cache_dir: Path = field(default_factory=lambda: Path(".cache/searxng"))
    cache_ttl: int = 3600  # 1 час

    # Rate limiting
    min_delay: float = 0.5  # Минимальная задержка между запросами
    max_delay: float = 2.0  # Максимальная задержка
    jitter: bool = True  # Случайное варьирование задержек

    # Parallelism
    max_concurrent: int = 5  # Макс. одновременных запросов

    # Engine fallback
    preferred_engines: List[str] = field(default_factory=lambda: [
        "duckduckgo", "brave", "qwant", "wikipedia"
    ])
    banned_engines: Set[str] = field(default_factory=set)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# =====================================================================
# CACHING
# =====================================================================

class SearchCache:
    """Дисковый кэш для результатов поиска с поддержкой UTF-8"""

    def __init__(self, cache_dir: Path, ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str, engines: Optional[List[str]] = None) -> str:
        """Генерирует уникальный ключ для запроса"""
        key_data = f"{query}:{','.join(sorted(engines or []))}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()

    def get(self, query: str, engines: Optional[List[str]] = None) -> Optional[Dict]:
        """Получает результат из кэша с robust error handling"""
        cache_key = self._get_cache_key(query, engines)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Проверяем возраст кэша
        try:
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > self.ttl:
                cache_file.unlink()  # Удаляем устаревший
                return None
        except Exception as e:
            logger.debug(f"Cache stat error: {e}")
            return None

        # Читаем с fallback на разные кодировки
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                text = cache_file.read_text(encoding=encoding)

                # Проверяем, что файл не пустой
                if not text or not text.strip():
                    logger.debug(f"Empty cache file: {cache_file.name}")
                    cache_file.unlink()  # Удаляем пустой файл
                    return None

                data = json.loads(text)
                logger.info(f"💾 Cache HIT: {query[:50]}... (age: {int(cache_age)}s)")
                return data

            except json.JSONDecodeError as e:
                if encoding == 'latin-1':  # Последняя попытка
                    logger.warning(f"Cache corrupted, deleting: {cache_file.name}")
                    cache_file.unlink()  # Удаляем поломанный файл
                    return None
                # Пробуем следующую кодировку
                continue

            except UnicodeDecodeError:
                if encoding == 'latin-1':  # Последняя попытка
                    logger.warning(f"Cache encoding error, deleting: {cache_file.name}")
                    cache_file.unlink()
                    return None
                continue

            except Exception as e:
                logger.warning(f"Cache read error ({encoding}): {e}")
                if encoding == 'latin-1':
                    cache_file.unlink()
                    return None
                continue

        return None

    def set(self, query: str, data: Dict, engines: Optional[List[str]] = None):
        """Сохраняет результат в кэш с атомарной записью"""
        cache_key = self._get_cache_key(query, engines)
        cache_file = self.cache_dir / f"{cache_key}.json"
        temp_file = self.cache_dir / f"{cache_key}.tmp"

        try:
            # Сначала пишем во временный файл
            temp_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

            # Атомарная замена (предотвращает частично записанные файлы)
            temp_file.replace(cache_file)

            logger.debug(f"💾 Cache WRITE: {query[:50]}... → {cache_file.name}")

        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            # Очищаем временный файл, если он остался
            if temp_file.exists():
                temp_file.unlink()

    def repair(self):
        """Удаляет все поломанные кэш файлы"""
        repaired = 0
        total = 0

        for cache_file in self.cache_dir.glob("*.json"):
            total += 1
            try:
                # Пробуем прочитать
                text = cache_file.read_text(encoding='utf-8')
                json.loads(text)
            except Exception:
                # Поломан — удаляем
                cache_file.unlink()
                repaired += 1
                logger.info(f"🔧 Repaired: {cache_file.name}")

        logger.info(f"✅ Cache repair complete: {repaired}/{total} files removed")

    def clear(self):
        """Очищает весь кэш"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("🗑️ Cache cleared")


# =====================================================================
# RATE LIMITER
# =====================================================================

class AdaptiveRateLimiter:
    """Умный rate limiter с адаптивными задержками"""

    def __init__(self, min_delay: float = 0.5, max_delay: float = 2.0, jitter: bool = True):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.last_request_time = 0
        self.consecutive_errors = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Ждёт перед следующим запросом"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time

            # Вычисляем задержку (увеличиваем при ошибках)
            base_delay = self.min_delay * (1.5 ** self.consecutive_errors)
            base_delay = min(base_delay, self.max_delay)

            # Добавляем jitter (случайное варьирование)
            if self.jitter:
                delay = base_delay + random.uniform(0, base_delay * 0.5)
            else:
                delay = base_delay

            # Ждём, если нужно
            if elapsed < delay:
                wait_time = delay - elapsed
                logger.debug(f"⏱️ Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()

    def report_success(self):
        """Сбрасываем счётчик ошибок при успехе"""
        self.consecutive_errors = max(0, self.consecutive_errors - 1)

    def report_error(self):
        """Увеличиваем задержки при ошибках"""
        self.consecutive_errors = min(self.consecutive_errors + 1, 5)
        if self.consecutive_errors > 0:
            logger.warning(f"⚠️ Consecutive errors: {self.consecutive_errors} (delays increased)")


# =====================================================================
# MAIN CLIENT
# =====================================================================

class JarvisSearchClient:
    """Production-ready SearXNG клиент"""

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.cache = SearchCache(self.config.cache_dir, self.config.cache_ttl)
        self.rate_limiter = AdaptiveRateLimiter(
            min_delay=self.config.min_delay,
            max_delay=self.config.max_delay,
            jitter=self.config.jitter
        )
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Статистика (для мониторинга)
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "engine_errors": defaultdict(int),
            "total_results": 0,
            "simplified_queries": 0
        }

    def _simplify_query(self, query: str) -> str:
        """
        Упрощает запрос, убирая специфичные термины и даты

        Examples:
            "machine learning RAG 2024" → "machine learning retrieval augmented generation"
            "deep learning transformers 2025" → "deep learning transformers"
        """
        # Убираем годы (1900-2099)
        simplified = re.sub(r'\b(19|20)\d{2}\b', '', query)

        # Раскрываем аббревиатуры
        abbreviation_map = {
            'RAG': 'retrieval augmented generation',
            'LLM': 'large language model',
            'NLP': 'natural language processing',
            'CV': 'computer vision',
            'ML': 'machine learning',
            'AI': 'artificial intelligence',
            'DL': 'deep learning',
            'RL': 'reinforcement learning',
            'GAN': 'generative adversarial network',
            'CNN': 'convolutional neural network',
            'RNN': 'recurrent neural network'
        }

        for abbr, full in abbreviation_map.items():
            # Case-insensitive замена
            pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
            if pattern.search(simplified):
                simplified = pattern.sub(full, simplified, count=1)
                break  # Заменяем только первую аббревиатуру

        # Убираем лишние пробелы
        simplified = ' '.join(simplified.split())

        return simplified.strip()

    def _get_fallback_engines(self, current_engines: List[str]) -> List[str]:
        """Выбирает альтернативные движки при неудаче"""
        all_safe = ["duckduckgo", "brave", "qwant", "wikipedia"]

        # Пробуем те, которые ещё не использовали
        unused = [e for e in all_safe
                  if e not in current_engines
                  and e not in self.config.banned_engines]

        if unused:
            return unused[:2]  # Берём 2 новых
        else:
            # Fallback на самый надёжный
            return ["duckduckgo"]

    async def search(
            self,
            query: str,
            engines: Optional[List[str]] = None,
            category: Optional[str] = None,
            max_results: Optional[int] = None,
            simplify_on_failure: bool = True
    ) -> Dict:
        """
        Выполняет поиск с retry logic и fallback

        Args:
            query: Поисковый запрос
            engines: Список движков (None = использовать preferred_engines)
            category: Категория ('general', 'science', 'it', 'news')
            max_results: Ограничение количества результатов
            simplify_on_failure: Упрощать запрос при пустых результатах

        Returns:
            Dict с ключами: results, query, engines_used, from_cache
        """
        self.stats["total_queries"] += 1
        original_query = query  # Сохраняем оригинал

        # Используем preferred engines, если не указаны
        if engines is None:
            engines = [e for e in self.config.preferred_engines
                       if e not in self.config.banned_engines]

        # Проверяем кэш
        cached = self.cache.get(original_query, engines)
        if cached:
            self.stats["cache_hits"] += 1
            cached["from_cache"] = True
            return cached

        self.stats["cache_misses"] += 1

        # Выполняем поиск с retry
        for attempt in range(self.config.max_retries):
            try:
                result = await self._search_with_rate_limit(
                    query, engines, category, max_results
                )

                if result.get("results"):
                    # Успех - кэшируем и возвращаем
                    self.rate_limiter.report_success()
                    result["from_cache"] = False
                    result["original_query"] = original_query
                    result["final_query"] = query
                    result["simplified"] = (query != original_query)

                    # Кэшируем под оригинальным запросом
                    self.cache.set(original_query, result, engines)
                    self.stats["total_results"] += len(result["results"])

                    return result

                else:
                    # Пустой результат - возможно бан или нет контента
                    logger.warning(
                        f"Empty result for '{query}' "
                        f"(attempt {attempt + 1}/{self.config.max_retries})"
                    )
                    self.rate_limiter.report_error()

                    if attempt < self.config.max_retries - 1:
                        # Strategy 1: Try fallback engines
                        new_engines = self._get_fallback_engines(engines)

                        # Strategy 2: Simplify query на последней попытке
                        if (attempt == self.config.max_retries - 2
                                and simplify_on_failure
                                and query == original_query):  # Ещё не упрощали

                            simplified = self._simplify_query(query)
                            if simplified != query:
                                query = simplified
                                self.stats["simplified_queries"] += 1
                                logger.info(f"💡 Query simplified: '{original_query}' → '{query}'")

                        backoff = 2 ** attempt
                        logger.info(
                            f"Retrying with engines={new_engines} after {backoff}s..."
                        )
                        engines = new_engines
                        await asyncio.sleep(backoff)

            except Exception as e:
                logger.error(
                    f"Search error on attempt {attempt + 1}: "
                    f"{type(e).__name__}: {e}"
                )
                self.rate_limiter.report_error()

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        # Все попытки неудачны
        logger.error(f"❌ Search failed after {self.config.max_retries} retries: {original_query}")

        simplified_suggestion = self._simplify_query(original_query)

        return {
            "results": [],
            "query": original_query,
            "final_query": query,
            "error": "Max retries exceeded",
            "suggestion": (
                f"Try: '{simplified_suggestion}'"
                if simplified_suggestion != original_query
                else None
            )
        }

    async def _search_with_rate_limit(
            self,
            query: str,
            engines: List[str],
            category: Optional[str],
            max_results: Optional[int]
    ) -> Dict:
        """Внутренний метод с rate limiting"""

        async with self.semaphore:
            await self.rate_limiter.acquire()

            # Строим параметры
            params = {
                "q": query,
                "format": "json",
                "engines": ",".join(engines)
            }

            if category:
                params["categories"] = category

            if max_results:
                params["pageno"] = 1  # Ограничиваем одной страницей

            url = f"{self.config.base_url}/search?{urlencode(params)}"
            logger.debug(f"🔗 Request: {url}")

            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:

                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])

                        # Обрезаем до max_results
                        if max_results:
                            results = results[:max_results]

                        # Статистика по движкам (для debug)
                        if logger.isEnabledFor(logging.DEBUG):
                            engine_counts = {}
                            for r in results:
                                engine = r.get("engine", "unknown")
                                engine_counts[engine] = engine_counts.get(engine, 0) + 1
                            logger.debug(f"Engine breakdown: {engine_counts}")

                        logger.info(
                            f"✅ '{query[:50]}...' → {len(results)} results "
                            f"(engines: {engines})"
                        )

                        return {
                            "results": results,
                            "query": query,
                            "engines_used": engines
                        }

                    elif response.status == 429:
                        # Rate limit от SearXNG
                        logger.warning(f"⚠️ 429 Rate Limit from SearXNG")
                        raise aiohttp.ClientError("Rate limited by SearXNG")

                    else:
                        error_text = await response.text()
                        logger.error(
                            f"❌ HTTP {response.status}: {error_text[:200]}"
                        )
                        raise aiohttp.ClientError(f"HTTP {response.status}")

    async def parallel_search(self, queries: List[str]) -> List[Dict]:
        """Параллельный поиск с семафором"""

        logger.info(f"🔍 Starting parallel search for {len(queries)} queries...")

        tasks = [self.search(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обрабатываем исключения
        clean_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i + 1} failed: {result}")
                clean_results.append({
                    "results": [],
                    "error": str(result),
                    "query": queries[i]
                })
            else:
                clean_results.append(result)

        return clean_results

    def print_stats(self):
        """Выводит статистику работы"""
        print("\n" + "=" * 70)
        print("📊 JARVIS SEARCH STATISTICS")
        print("=" * 70)
        print(f"Total queries:      {self.stats['total_queries']}")

        if self.stats['total_queries'] > 0:
            hit_rate = self.stats['cache_hits'] / self.stats['total_queries'] * 100
            print(f"Cache hits:         {self.stats['cache_hits']} ({hit_rate:.1f}%)")
        else:
            print(f"Cache hits:         {self.stats['cache_hits']}")

        print(f"Cache misses:       {self.stats['cache_misses']}")
        print(f"Total results:      {self.stats['total_results']}")
        print(f"Simplified queries: {self.stats['simplified_queries']}")

        if self.stats['engine_errors']:
            print("\nEngine errors:")
            for engine, count in self.stats['engine_errors'].items():
                print(f"  {engine}: {count}")

        print("=" * 70 + "\n")

    async def health_check(self) -> Dict:
        """Проверяет доступность SearXNG"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.config.base_url}/search?q=test&format=json",
                        timeout=aiohttp.ClientTimeout(total=5)
                ) as response:

                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])

                        # Статистика по движкам
                        engine_counts = {}
                        for r in results:
                            engine = r.get("engine", "unknown")
                            engine_counts[engine] = engine_counts.get(engine, 0) + 1

                        return {
                            "healthy": True,
                            "total_results": len(results),
                            "active_engines": list(engine_counts.keys()),
                            "engine_breakdown": engine_counts
                        }
                    else:
                        return {
                            "healthy": False,
                            "error": f"HTTP {response.status}"
                        }

        except Exception as e:
            return {"healthy": False, "error": str(e)}


# =====================================================================
# URL DEDUPLICATION
# =====================================================================

class URLDeduplicator:
    """Дедупликация и Domain Diversity"""

    def __init__(self, max_per_domain: int = 2):
        self.seen_urls: Set[str] = set()
        self.domain_counts: Dict[str, int] = defaultdict(int)
        self.max_per_domain = max_per_domain

    def normalize_url(self, url: str) -> str:
        """Нормализация URL"""
        parsed = urlparse(url.lower())

        # Убираем tracking параметры
        if parsed.query:
            params = parse_qs(parsed.query)
            clean_params = {
                k: v for k, v in params.items()
                if k not in [
                    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                    'ref', 'fbclid', 'gclid', 'source', 'si', '_hsenc', '_hsmi'
                ]
            }
            query = urlencode(clean_params, doseq=True) if clean_params else ''
        else:
            query = ''

        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/'),
            '',  # params
            query,
            ''  # fragment
        ))

    def is_duplicate(self, url: str) -> bool:
        """Проверка на дубликат с domain diversity"""
        normalized = self.normalize_url(url)

        # Level 1: URL duplicate
        if normalized in self.seen_urls:
            logger.debug(f"⏭️  URL duplicate: {url}")
            return True

        # Level 2: Domain diversity
        domain = urlparse(normalized).netloc
        if self.domain_counts[domain] >= self.max_per_domain:
            logger.debug(f"⏭️  Domain limit reached: {domain}")
            return True

        # Добавляем
        self.seen_urls.add(normalized)
        self.domain_counts[domain] += 1
        return False

    def get_stats(self) -> Dict:
        return {
            "unique_urls": len(self.seen_urls),
            "unique_domains": len(self.domain_counts),
            "domain_distribution": dict(self.domain_counts)
        }


# =====================================================================
# HIGH-LEVEL API
# =====================================================================

async def smart_search(
        queries: List[str],
        max_sources: int = 10,
        config: Optional[SearchConfig] = None
) -> List[Dict]:
    """
    Умный поиск с дедупликацией и domain diversity

    Args:
        queries: Список запросов
        max_sources: Максимум уникальных источников
        config: Конфигурация клиента

    Returns:
        List уникальных результатов с метаданными
    """
    client = JarvisSearchClient(config or SearchConfig())
    deduplicator = URLDeduplicator(max_per_domain=2)

    # Параллельный поиск
    search_results = await client.parallel_search(queries)

    # Собираем все результаты
    all_results = []
    for search_result in search_results:
        all_results.extend(search_result.get("results", []))

    logger.info(f"📦 Total raw results: {len(all_results)}")

    # Дедупликация
    unique_results = []
    duplicates_count = 0

    for result in all_results:
        url = result.get("url", "")
        if not url:
            continue

        if not deduplicator.is_duplicate(url):
            unique_results.append(result)

            if len(unique_results) >= max_sources:
                logger.info(f"✅ Reached max_sources limit ({max_sources})")
                break
        else:
            duplicates_count += 1

    logger.info(f"✅ Unique results: {len(unique_results)}")
    logger.info(f"⏭️  Duplicates filtered: {duplicates_count}")

    stats = deduplicator.get_stats()
    logger.info(f"🌐 Unique domains: {stats['unique_domains']}")
    logger.debug(f"Domain distribution: {stats['domain_distribution']}")

    # Статистика клиента
    client.print_stats()

    return unique_results


# =====================================================================
# CLI INTERFACE
# =====================================================================

async def main():
    """Точка входа для CLI"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Jarvis Search Engine - Production-ready SearXNG client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          python jarvis_search_v2.py "deep learning"
          python jarvis_search_v2.py --verbose "RAG 2024"
          python jarvis_search_v2.py --clear-cache
          python jarvis_search_v2.py --repair-cache
        """
    )

    parser.add_argument('queries', nargs='*', help='Search queries')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')
    parser.add_argument('--max-sources', type=int, default=10, help='Max unique sources')
    parser.add_argument('--cache-ttl', type=int, default=3600, help='Cache TTL in seconds')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache and exit')
    parser.add_argument('--repair-cache', action='store_true', help='Repair corrupted cache files')  # NEW!
    parser.add_argument('--health-check', action='store_true', help='Check SearXNG health')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = SearchConfig(
        base_url="http://localhost:8080",
        min_delay=0.5,
        max_delay=2.0,
        max_concurrent=5,
        max_retries=3,
        cache_ttl=args.cache_ttl if not args.no_cache else 0
    )

    cache = SearchCache(config.cache_dir, config.cache_ttl)

    # NEW: Repair cache
    if args.repair_cache:
        print("🔧 Repairing cache...")
        cache.repair()
        return

    if args.clear_cache:
        cache.clear()
        print("✅ Cache cleared successfully")
        return

    # Health check
    if args.health_check:
        client = JarvisSearchClient(config)
        print("🏥 Checking SearXNG health...")
        health = await client.health_check()
        print(json.dumps(health, indent=2, ensure_ascii=False))
        return

    # Queries
    if args.queries:
        queries = args.queries
    else:
        # Тестовые запросы
        queries = [
            "погода челябинск 08 декабря 2025 год",
        ]
        print(f"📝 Using {len(queries)} test queries\n")

    # Выполняем поиск
    results = await smart_search(queries, max_sources=args.max_sources, config=config)

    # Выводим результаты
    print("\n" + "=" * 70)
    print("🎯 FINAL RESULTS")
    print("=" * 70)

    if not results:
        print("\n❌ No results found. Troubleshooting:")
        print("  1. Check SearXNG: docker ps | grep searxng")
        print("  2. View logs: docker logs jarvis-searxng")
        print("  3. Test manually: curl 'http://localhost:8080/search?q=test&format=json'")
        print("  4. Health check: python jarvis_search_v2.py --health-check")
        print("  5. Enable verbose: python jarvis_search_v2.py --verbose")
        return

    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        engine = result.get('engine', 'unknown')
        snippet = result.get('content', '')[:150]

        print(f"\n{i}. {title}")
        print(f"   URL: {url}")
        print(f"   Engine: {engine}")
        if snippet:
            print(f"   {snippet}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        logger.exception("Fatal error:")
        print(f"\n❌ Fatal error: {e}")
