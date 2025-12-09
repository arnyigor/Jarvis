#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Комплексное тестирование Hybrid RAG v5.2 на реальных PDF и JSON данных.

Используемые файлы:
- printsipy-proektirovaniya-integralnoy-modeli-otsenki-nadezhnosti.pdf (русский, техническая документация)
- IPCC_AR6_SYR_FullVolume.pdf (английский, ~3000 страниц, климатология)
- 2512.07795v1.pdf (arxiv, научная статья)
- arxiv-metadata-oai-snapshot.json (5GB метаданных)
"""

import asyncio
import json
import logging
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import psutil
from tqdm import tqdm

from src.hybrid_rag_system import HybridRAGSystem, HybridConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Конфигурация тестов
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RealWorldTestConfig:
    """Конфигурация для тестирования на реальных данных"""

    # Пути к тестовым файлам
    test_data_dir: Path = Path("./docs")

    # Лимиты для стресс-тестов
    max_indexing_time_small: float = 120.0  # секунд для маленького PDF
    max_indexing_time_large: float = 1800.0  # 30 минут для IPCC (огромный)
    max_search_latency: float = 500.0  # миллисекунд

    # Пороги качества
    min_precision_at_5: float = 0.4
    min_recall_at_5: float = 0.6
    min_ndcg_at_5: float = 0.5

    # Лимиты памяти
    max_memory_growth_mb: float = 500.0  # Рост памяти за сессию
    max_memory_growth_percent: float = 25.0  # % от начального

    # OCR настройки
    ocr_engine: str = "rapidocr"  # "tesseract" | "easyocr" | "rapidocr"
    test_ocr_pdf: bool = False  # Включить тесты сканированных PDF


# ═══════════════════════════════════════════════════════════════════
# Ground Truth для реальных документов
# ═══════════════════════════════════════════════════════════════════

class RealWorldGroundTruth:
    """
    Эталонные запросы и ожидаемые результаты для реальных PDF.
    Составляются вручную после беглого ознакомления с документами.
    """

    @staticmethod
    def get_russian_tech_queries() -> Dict[str, Dict]:
        """
        Запросы для printsipy-proektirovaniya-integralnoy-modeli-otsenki-nadezhnosti.pdf
        (Принципы проектирования интегральной модели оценки надёжности ИВС)
        """
        return {
            "оценка надёжности информационных систем": {
                "keywords": ["надёжность", "информационная система", "оценка"],
                "expected_terms": ["отказоустойчивость", "доступность", "безопасность"],
                "min_results": 3,
                "context": "Должны найтись чанки про методы оценки надёжности ИВС"
            },

            "модели отказоустойчивости": {
                "keywords": ["модель", "отказ", "устойчивость"],
                "expected_terms": ["резервирование", "избыточность", "восстановление"],
                "min_results": 2,
                "context": "Архитектурные подходы к отказоустойчивости"
            },

            "интегральная модель проектирования": {
                "keywords": ["интегральный", "проектирование", "модель"],
                "expected_terms": ["методология", "этап", "критерий"],
                "min_results": 3,
                "context": "Основная тема документа"
            },
        }

    @staticmethod
    def get_ipcc_climate_queries() -> Dict[str, Dict]:
        """
        Запросы для IPCC_AR6_SYR_FullVolume.pdf
        (Доклад IPCC о климатических изменениях, ~3000 страниц)
        """
        return {
            "global warming temperature increase 1.5 degrees": {
                "keywords": ["warming", "temperature", "1.5", "degrees"],
                "expected_terms": ["paris agreement", "mitigation", "adaptation"],
                "min_results": 5,
                "context": "Ключевой порог потепления из Парижского соглашения"
            },

            "climate change impacts biodiversity": {
                "keywords": ["climate", "impact", "biodiversity", "ecosystem"],
                "expected_terms": ["species", "extinction", "habitat"],
                "min_results": 4,
                "context": "Влияние на экосистемы"
            },

            "carbon emissions reduction pathways": {
                "keywords": ["carbon", "emission", "reduction", "pathway"],
                "expected_terms": ["net zero", "renewable", "fossil fuel"],
                "min_results": 5,
                "context": "Сценарии декарбонизации"
            },

            "renewable energy solar wind": {
                "keywords": ["renewable", "solar", "wind", "energy"],
                "expected_terms": ["capacity", "deployment", "cost"],
                "min_results": 3,
                "context": "Возобновляемая энергетика"
            },
        }

    @staticmethod
    def get_arxiv_paper_queries() -> Dict[str, Dict]:
        """
        Запросы для 2512.07795v1.pdf (arxiv статья)
        ВАЖНО: Содержимое неизвестно, но можно протестировать общие паттерны
        """
        return {
            "abstract introduction": {
                "keywords": ["abstract", "introduction"],
                "expected_terms": ["method", "result", "propose"],
                "min_results": 2,
                "context": "Стандартные секции научной статьи"
            },

            "methodology experimental setup": {
                "keywords": ["methodology", "experiment", "setup"],
                "expected_terms": ["dataset", "evaluation", "metric"],
                "min_results": 2,
                "context": "Методологическая часть"
            },

            "conclusion future work": {
                "keywords": ["conclusion", "future", "work"],
                "expected_terms": ["demonstrate", "improve", "limitation"],
                "min_results": 1,
                "context": "Заключение и перспективы"
            },
        }


# ═══════════════════════════════════════════════════════════════════
# Утилиты для работы с arxiv-metadata JSON
# ═══════════════════════════════════════════════════════════════════

class ArxivMetadataIndexer:
    """
    Инкрементальная индексация 5GB JSON с метаданными arxiv.
    Обрабатывает файл построчно, чтобы не загружать всё в память.
    """

    def __init__(self, json_path: Path, output_dir: Path, max_records: int = 10000):
        """
        Args:
            json_path: Путь к arxiv-metadata-oai-snapshot.json
            output_dir: Куда сохранить обработанные чанки как .txt файлы
            max_records: Сколько записей обработать (для тестов ограничиваем)
        """
        self.json_path = json_path
        self.output_dir = output_dir
        self.max_records = max_records
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_to_text_files(self) -> Tuple[int, int]:
        """
        Читает JSON построчно, извлекает title + abstract,
        сохраняет как отдельные .txt файлы.

        Returns:
            (processed_count, skipped_count)
        """
        logger.info(f"📂 Processing {self.json_path.name} (up to {self.max_records} records)...")

        processed = 0
        skipped = 0

        with open(self.json_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=self.max_records, desc="Parsing JSON")):
                if processed >= self.max_records:
                    break

                try:
                    record = json.loads(line.strip())

                    # Извлекаем поля
                    arxiv_id = record.get('id', f'unknown_{line_num}')
                    title = record.get('title', '').strip()
                    abstract = record.get('abstract', '').strip()
                    categories = record.get('categories', '')

                    # Фильтруем пустые
                    if not title or not abstract or len(abstract) < 100:
                        skipped += 1
                        continue

                    # Формируем текст для индексации
                    content = f"""Title: {title}

Categories: {categories}

Abstract:
{abstract}

ArXiv ID: {arxiv_id}
"""

                    # Сохраняем как .txt файл (безопасное имя)
                    safe_id = arxiv_id.replace('/', '_').replace('\\', '_')
                    output_file = self.output_dir / f"arxiv_{safe_id}.txt"

                    with open(output_file, 'w', encoding='utf-8') as out:
                        out.write(content)

                    processed += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Line {line_num}: Invalid JSON - {e}")
                    skipped += 1
                except Exception as e:
                    logger.error(f"❌ Line {line_num}: {e}")
                    skipped += 1

        logger.info(f"✅ Processed {processed} records, skipped {skipped}")
        return processed, skipped


# ═══════════════════════════════════════════════════════════════════
# Основной тестовый класс
# ═══════════════════════════════════════════════════════════════════

class TestRealWorldPDFProcessing(unittest.IsolatedAsyncioTestCase):
    """
    Комплексное тестирование RAG-системы на реальных сложных документах.

    Test Suite:
    1. Индексация (производительность, обработка ошибок)
    2. Качество поиска (Precision/Recall/NDCG)
    3. Стресс-тесты (память, latency, concurrent queries)
    4. Edge cases (большие документы, мультиязычность)
    """

    @classmethod
    def setUpClass(cls):
        """Инициализация тестового окружения"""
        cls.test_config = RealWorldTestConfig()

        # Проверяем наличие тестовых файлов
        cls.test_files = {
            'russian_tech': cls.test_config.test_data_dir / "printsipy-proektirovaniya-integralnoy-modeli-otsenki-nadezhnosti.pdf",
            'ipcc_climate': cls.test_config.test_data_dir / "IPCC_AR6_SYR_FullVolume.pdf",
            'arxiv_paper': cls.test_config.test_data_dir / "2512.07795v1.pdf",
            'arxiv_json': cls.test_config.test_data_dir / "arxiv_pdfs" / "arxiv-metadata-oai-snapshot.json",
        }

        # Создаём директорию если нет
        cls.test_config.test_data_dir.mkdir(parents=True, exist_ok=True)

        # Проверяем какие файлы доступны
        cls.available_files = {
            name: path for name, path in cls.test_files.items()
            if path.exists()
        }

        if not cls.available_files:
            raise FileNotFoundError(
                f"No test files found in {cls.test_config.test_data_dir}. "
                f"Please place PDF files there."
            )

        logger.info(f"📂 Available test files: {list(cls.available_files.keys())}")

        # Ground truth
        cls.ground_truth = {}
        if 'russian_tech' in cls.available_files:
            cls.ground_truth.update(RealWorldGroundTruth.get_russian_tech_queries())
        if 'ipcc_climate' in cls.available_files:
            cls.ground_truth.update(RealWorldGroundTruth.get_ipcc_climate_queries())
        if 'arxiv_paper' in cls.available_files:
            cls.ground_truth.update(RealWorldGroundTruth.get_arxiv_paper_queries())

        # Инициализация RAG системы
        cls.rag_config = HybridConfig(
            static_docs_dir=cls.test_config.test_data_dir,
            enable_progress_bars=True,
            chunk_size=512,
            chunk_overlap=200,
            top_k_bm25=50,
            top_k_semantic=50,
            top_k_final=5,
        )

        cls.rag = HybridRAGSystem(cls.rag_config)

        # Метрики производительности
        cls.perf_metrics = {
            'indexing_times': {},
            'search_latencies': [],
            'memory_usage': [],
        }

    def test_00_gpu_diagnostics(self):
        """Предварительная проверка GPU"""
        import torch

        logger.info("="*80)
        logger.info("GPU Diagnostics")
        logger.info("="*80)

        self.assertTrue(torch.cuda.is_available(), "CUDA not available")

        logger.info(f"   Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"   CUDA Version: {torch.version.cuda}")

        # Тест скорости эмбеддинга
        test_texts = ["test " * 50] * 100

        start = time.perf_counter()
        embeddings = self.rag.embedding_model.encode(test_texts, batch_size=32)
        elapsed = time.perf_counter() - start

        throughput = len(test_texts) / elapsed

        logger.info(f"   Embedding throughput: {throughput:.1f} docs/sec")

        self.assertGreater(throughput, 50, "GPU too slow for embeddings")

    # ───────────────────────────────────────────────────────────────
    # Тест 1: Индексация
    # ───────────────────────────────────────────────────────────────

    def test_01_indexing_small_pdf(self):
        """Тест индексации небольшого PDF (русский техдок)"""
        if 'russian_tech' not in self.available_files:
            self.skipTest("Russian tech PDF not available")

        logger.info("="*80)
        logger.info("TEST 1: Indexing Small PDF (Russian Technical Document)")
        logger.info("="*80)

        start_time = time.perf_counter()

        try:
            indexed_count = self.rag.index_static_documents(force=True)
            elapsed = time.perf_counter() - start_time

            self.perf_metrics['indexing_times']['small_pdf'] = elapsed

            # Проверки
            self.assertGreater(indexed_count, 0, "No documents indexed")
            self.assertLess(
                elapsed,
                self.test_config.max_indexing_time_small,
                f"Indexing too slow: {elapsed:.1f}s"
            )

            # Статистика
            stats = self.rag.collection.count()
            logger.info(f"✅ Indexed {indexed_count} chunks in {elapsed:.2f}s")
            logger.info(f"   Chunks/second: {indexed_count/elapsed:.1f}")
            logger.info(f"   Total docs in ChromaDB: {stats}")

        except Exception as e:
            self.fail(f"Indexing failed: {e}")

    def test_02_indexing_large_pdf(self):
        """Тест индексации большого PDF (IPCC ~3000 страниц)"""
        if 'ipcc_climate' not in self.available_files:
            self.skipTest("IPCC PDF not available")

        logger.info("="*80)
        logger.info("TEST 2: Indexing Large PDF (IPCC AR6, ~3000 pages)")
        logger.info("="*80)
        logger.warning("⚠️ This may take 20-30 minutes on first run...")

        start_time = time.perf_counter()

        try:
            indexed_count = self.rag.index_static_documents(force=False)  # Не force - кэш
            elapsed = time.perf_counter() - start_time

            self.perf_metrics['indexing_times']['large_pdf'] = elapsed

            self.assertLess(
                elapsed,
                self.test_config.max_indexing_time_large,
                f"Large PDF indexing exceeded time limit"
            )

            logger.info(f"✅ Processed in {elapsed:.2f}s ({elapsed/60:.1f} minutes)")

        except Exception as e:
            self.fail(f"Large PDF indexing failed: {e}")

    def test_03_indexing_arxiv_json(self):
        """Тест индексации 5GB JSON (инкрементальный парсинг)"""
        if 'arxiv_json' not in self.available_files:
            self.skipTest("ArXiv JSON not available")

        logger.info("="*80)
        logger.info("TEST 3: Indexing Large JSON (arxiv-metadata, 5GB)")
        logger.info("="*80)

        # Сначала парсим JSON в .txt файлы
        arxiv_dir = self.test_config.test_data_dir / "arxiv_parsed"
        indexer = ArxivMetadataIndexer(
            self.test_files['arxiv_json'],
            arxiv_dir,
            max_records=1000  # Ограничиваем для теста
        )

        processed, skipped = indexer.process_to_text_files()

        self.assertGreater(processed, 0, "No records processed from JSON")

        # Теперь индексируем .txt файлы
        old_docs_dir = self.rag.config.static_docs_dir
        self.rag.config.static_docs_dir = arxiv_dir

        start_time = time.perf_counter()
        indexed_count = self.rag.index_static_documents(force=True)
        elapsed = time.perf_counter() - start_time

        self.rag.config.static_docs_dir = old_docs_dir  # Восстанавливаем

        self.assertGreater(indexed_count, 0, "ArXiv records not indexed")
        logger.info(f"✅ Indexed {indexed_count} arxiv abstracts in {elapsed:.2f}s")

    # ───────────────────────────────────────────────────────────────
    # Тест 2: Качество поиска
    # ───────────────────────────────────────────────────────────────

    async def test_04_search_quality_russian(self):
        """Тест качества поиска на русском техническом документе"""
        if 'russian_tech' not in self.available_files:
            self.skipTest("Russian tech PDF not available")

        logger.info("="*80)
        logger.info("TEST 4: Search Quality (Russian Technical Doc)")
        logger.info("="*80)

        queries = RealWorldGroundTruth.get_russian_tech_queries()

        for query_text, gt in queries.items():
            logger.info(f"\n🔍 Query: '{query_text}'")

            result = await self.rag.hybrid_search(query_text)
            results = result['results']

            # Проверка минимального количества результатов
            self.assertGreaterEqual(
                len(results),
                gt['min_results'],
                f"Too few results for '{query_text}'"
            )

            # Проверка наличия ожидаемых терминов
            all_text = " ".join([r['text'].lower() for r in results])
            found_terms = [
                term for term in gt['expected_terms']
                if term.lower() in all_text
            ]

            coverage = len(found_terms) / len(gt['expected_terms']) * 100

            logger.info(f"   Results: {len(results)}")
            logger.info(f"   Expected terms found: {found_terms} ({coverage:.0f}%)")
            logger.info(f"   Top result preview: {results[0]['text'][:150]}...")

            # Мягкая проверка (хотя бы 50% терминов должны быть)
            self.assertGreater(
                coverage, 50.0,
                f"Low term coverage ({coverage:.0f}%) for '{query_text}'"
            )

    async def test_05_search_quality_english(self):
        """Тест качества поиска на английском документе (IPCC)"""
        if 'ipcc_climate' not in self.available_files:
            self.skipTest("IPCC PDF not available")

        logger.info("="*80)
        logger.info("TEST 5: Search Quality (English IPCC Document)")
        logger.info("="*80)

        queries = RealWorldGroundTruth.get_ipcc_climate_queries()

        precision_scores = []

        for query_text, gt in queries.items():
            logger.info(f"\n🔍 Query: '{query_text}'")

            start = time.perf_counter()
            result = await self.rag.hybrid_search(query_text)
            latency = (time.perf_counter() - start) * 1000

            self.perf_metrics['search_latencies'].append(latency)

            results = result['results']

            # Precision: сколько результатов содержат ключевые слова
            relevant_count = 0
            for res in results:
                text_lower = res['text'].lower()
                if any(kw.lower() in text_lower for kw in gt['keywords']):
                    relevant_count += 1

            precision = relevant_count / len(results) if results else 0
            precision_scores.append(precision)

            logger.info(f"   Latency: {latency:.1f}ms")
            logger.info(f"   Precision@5: {precision:.2f}")
            logger.info(f"   BM25: {result['bm25_count']}, Semantic: {result['semantic_count']}")

            self.assertGreater(
                precision, 0.0,
                f"Zero precision for '{query_text}'"
            )

        avg_precision = np.mean(precision_scores)
        logger.info(f"\n📊 Average Precision@5: {avg_precision:.3f}")

        self.assertGreater(
            avg_precision,
            self.test_config.min_precision_at_5,
            f"Low average precision: {avg_precision:.3f}"
        )

    # ───────────────────────────────────────────────────────────────
    # Тест 3: Производительность и стресс
    # ───────────────────────────────────────────────────────────────

    async def test_06_search_latency(self):
        """Тест latency поиска (должен быть < 500ms)"""
        logger.info("="*80)
        logger.info("TEST 6: Search Latency Performance")
        logger.info("="*80)

        test_queries = [
            "machine learning neural networks",
            "climate change mitigation",
            "оценка надёжности систем",
            "renewable energy sources",
        ]

        latencies = []

        for query in test_queries:
            start = time.perf_counter()
            result = await self.rag.hybrid_search(query)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            logger.info(f"   '{query[:40]}': {latency:.1f}ms")

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        logger.info(f"\n📊 Average latency: {avg_latency:.1f}ms")
        logger.info(f"📊 P95 latency: {p95_latency:.1f}ms")

        self.assertLess(
            avg_latency,
            self.test_config.max_search_latency,
            f"Average latency too high: {avg_latency:.1f}ms"
        )

        self.assertLess(
            p95_latency,
            self.test_config.max_search_latency * 1.5,
            f"P95 latency too high: {p95_latency:.1f}ms"
        )

    async def test_07_concurrent_queries(self):
        """Тест параллельных запросов (stress test)"""
        logger.info("="*80)
        logger.info("TEST 7: Concurrent Query Stress Test")
        logger.info("="*80)

        queries = [f"test query {i}" for i in range(20)]

        start = time.perf_counter()
        tasks = [self.rag.hybrid_search(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - start

        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        qps = len(queries) / elapsed

        logger.info(f"   Total queries: {len(queries)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Time: {elapsed:.2f}s")
        logger.info(f"   QPS: {qps:.1f}")

        self.assertEqual(failed, 0, f"{failed} queries failed")
        self.assertGreater(qps, 5.0, f"QPS too low: {qps:.1f}")

    async def test_08_memory_stability(self):
        """Тест стабильности памяти (поиск memory leaks)"""
        logger.info("="*80)
        logger.info("TEST 8: Memory Stability (Leak Detection)")
        logger.info("="*80)

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        logger.info(f"   Initial memory: {initial_memory:.1f} MB")

        # Прогоняем 100 запросов
        for i in range(100):
            query = f"memory test query {i % 10}"
            await self.rag.hybrid_search(query)

            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                logger.info(f"   After {i} queries: {current_memory:.1f} MB")

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        growth_percent = (memory_growth / initial_memory) * 100

        logger.info(f"\n📊 Memory analysis:")
        logger.info(f"   Initial: {initial_memory:.1f} MB")
        logger.info(f"   Final: {final_memory:.1f} MB")
        logger.info(f"   Growth: {memory_growth:.1f} MB ({growth_percent:.1f}%)")

        self.assertLess(
            memory_growth,
            self.test_config.max_memory_growth_mb,
            f"Memory leak detected: {memory_growth:.1f} MB growth"
        )

        self.assertLess(
            growth_percent,
            self.test_config.max_memory_growth_percent,
            f"Memory growth too high: {growth_percent:.1f}%"
        )

    # ───────────────────────────────────────────────────────────────
    # Тест 4: Edge Cases
    # ───────────────────────────────────────────────────────────────

    async def test_09_edge_case_very_long_query(self):
        """Тест обработки очень длинного запроса"""
        logger.info("="*80)
        logger.info("TEST 9: Edge Case - Very Long Query")
        logger.info("="*80)

        # Запрос из 500 слов
        long_query = " ".join(["machine learning neural network"] * 100)

        try:
            result = await self.rag.hybrid_search(long_query)
            self.assertIsNotNone(result)
            logger.info(f"✅ Handled {len(long_query)} char query")
        except Exception as e:
            self.fail(f"Failed on long query: {e}")

    async def test_10_edge_case_unicode_multilang(self):
        """Тест Unicode и мультиязычных запросов"""
        logger.info("="*80)
        logger.info("TEST 10: Edge Case - Unicode & Multilang")
        logger.info("="*80)

        unicode_queries = [
            "機械学習とディープラーニング",  # Японский
            "الذكاء الاصطناعي",  # Арабский
            "μηχανική μάθηση",  # Греческий
            "Смешанный русский + English query",
            "Emoji test 🤖🔍📊",
        ]

        for query in unicode_queries:
            try:
                result = await self.rag.hybrid_search(query)
                self.assertIsNotNone(result)
                logger.info(f"✅ '{query[:30]}'")
            except Exception as e:
                self.fail(f"Failed on unicode query '{query}': {e}")

    async def test_11_edge_case_empty_query(self):
        """Тест пустого запроса"""
        result = await self.rag.hybrid_search("")
        self.assertEqual(len(result['results']), 0, "Empty query should return no results")

    # ───────────────────────────────────────────────────────────────
    # Финальный отчёт
    # ───────────────────────────────────────────────────────────────

    @classmethod
    def tearDownClass(cls):
        """Вывод финального отчёта"""
        logger.info("\n" + "="*80)
        logger.info("FINAL PERFORMANCE REPORT")
        logger.info("="*80)

        # Индексация
        if cls.perf_metrics['indexing_times']:
            logger.info("\n📊 Indexing Performance:")
            for name, time_sec in cls.perf_metrics['indexing_times'].items():
                logger.info(f"   {name}: {time_sec:.2f}s ({time_sec/60:.1f} min)")

        # Поиск
        if cls.perf_metrics['search_latencies']:
            latencies = cls.perf_metrics['search_latencies']
            logger.info("\n📊 Search Performance:")
            logger.info(f"   Queries tested: {len(latencies)}")
            logger.info(f"   Avg latency: {np.mean(latencies):.1f}ms")
            logger.info(f"   Median: {np.median(latencies):.1f}ms")
            logger.info(f"   P95: {np.percentile(latencies, 95):.1f}ms")
            logger.info(f"   Min: {min(latencies):.1f}ms, Max: {max(latencies):.1f}ms")

        # Статистика коллекции
        stats_final = cls.rag.collection.count()
        logger.info(f"\n📚 Final Collection Stats:")
        logger.info(f"   Total chunks: {stats_final}")

        logger.info("\n" + "="*80)


# ═══════════════════════════════════════════════════════════════════
# Запуск тестов
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Настройка отчётности

    # Запускаем с verbose
    unittest.main(verbosity=2, failfast=False)
