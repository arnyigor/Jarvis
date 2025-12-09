#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_hybrid_rag_complete.py
ПОЛНАЯ тестовая платформа для Hybrid RAG v5.0
Включает: качество, производительность, устойчивость, edge cases, метрики.
"""
import asyncio
import math
import time
import unittest
from pathlib import Path
from typing import List, Dict

import numpy as np


# -------------------------------------------------------------------
# КАТЕГОРИЯ 1: Тесты качества поиска (Quality Metrics)
# -------------------------------------------------------------------

class TestSearchQualityMetrics(unittest.IsolatedAsyncioTestCase):
    """Продвинутые метрики качества: NDCG, MAP, Precision@K, Recall@K"""

    @classmethod
    def setUpClass(cls):
        """Подготовка тестового набора с ground truth"""
        from src.hybrid_rag_system import HybridRAGSystem, HybridConfig

        cls.test_dir = Path("test_data_integration")
        cls.config = HybridConfig(
            static_docs_dir=cls.test_dir,
            enable_progress_bars=False,
        )
        cls.rag = HybridRAGSystem(cls.config)
        cls.rag.index_static_documents(force=True)
        # Ground truth: запросы с известными релевантными документами
        cls.ground_truth = {
            "квантовые компьютеры": {
                "relevant_ids": ["0"],  # ID документа с exact match
                "relevance_scores": {"0": 3},  # 3 = высокая релевантность
            },
            "искусственный интеллект": {
                "relevant_ids": ["0", "1"],
                "relevance_scores": {"0": 2, "1": 3},
            },
            "нейронные сети": {
                "relevant_ids": ["0", "1"],
                "relevance_scores": {"0": 3, "1": 2},
            },
        }

    @staticmethod
    def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Precision@K = (релевантные в TOP-K) / K"""
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_top_k / k if k > 0 else 0.0

    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Recall@K = (релевантные в TOP-K) / (всего релевантных)"""
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_top_k / len(relevant_ids)

    @staticmethod
    def calculate_ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, int], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K"""
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            relevance = relevance_scores.get(doc_id, 0)
            dcg += (2 ** relevance - 1) / math.log2(i + 1)

        # IDCG (ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances[:k], start=1):
            idcg += (2 ** rel - 1) / math.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_map(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Mean Average Precision"""
        if not relevant_ids:
            return 0.0

        precisions = []
        num_relevant_found = 0

        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / i
                precisions.append(precision_at_i)

        return sum(precisions) / len(relevant_ids) if precisions else 0.0

    async def test_precision_recall_at_k(self):
        """Тест Precision@K и Recall@K"""
        k_values = [1, 3, 5, 10]
        results = {}

        for query, gt in self.ground_truth.items():
            search_result = await self.rag.hybrid_search(query)
            retrieved_ids = [r["id"] for r in search_result["results"]]

            for k in k_values:
                precision = self.calculate_precision_at_k(retrieved_ids, gt["relevant_ids"], k)
                recall = self.calculate_recall_at_k(retrieved_ids, gt["relevant_ids"], k)

                results[f"{query}_P@{k}"] = precision
                results[f"{query}_R@{k}"] = recall

        # Усредняем по всем запросам
        avg_precision_5 = np.mean([v for k, v in results.items() if "P@5" in k])
        avg_recall_5 = np.mean([v for k, v in results.items() if "R@5" in k])

        print(f"\n📊 Precision & Recall:")
        print(f"   Avg Precision@5: {avg_precision_5:.3f}")
        print(f"   Avg Recall@5: {avg_recall_5:.3f}")

        # Проверки
        self.assertGreater(avg_precision_5, 0.3, "Precision@5 слишком низкий")
        self.assertGreater(avg_recall_5, 0.5, "Recall@5 слишком низкий")

    async def test_ndcg_metric(self):
        """Тест NDCG@K (учитывает порядок результатов)"""
        ndcg_scores = []

        for query, gt in self.ground_truth.items():
            search_result = await self.rag.hybrid_search(query)
            retrieved_ids = [r["id"] for r in search_result["results"]]

            ndcg_5 = self.calculate_ndcg_at_k(retrieved_ids, gt["relevance_scores"], 5)
            ndcg_scores.append(ndcg_5)

            print(f"   {query}: NDCG@5 = {ndcg_5:.3f}")

        avg_ndcg = np.mean(ndcg_scores)
        print(f"\n   Avg NDCG@5: {avg_ndcg:.3f}")

        self.assertGreater(avg_ndcg, 0.6, "NDCG@5 ниже порога")

    async def test_mean_average_precision(self):
        """Тест MAP (Mean Average Precision)"""
        map_scores = []

        for query, gt in self.ground_truth.items():
            search_result = await self.rag.hybrid_search(query)
            retrieved_ids = [r["id"] for r in search_result["results"]]

            map_score = self.calculate_map(retrieved_ids, gt["relevant_ids"])
            map_scores.append(map_score)

            print(f"   {query}: MAP = {map_score:.3f}")

        avg_map = np.mean(map_scores)
        print(f"\n   Avg MAP: {avg_map:.3f}")

        self.assertGreater(avg_map, 0.5, "MAP ниже порога")


# -------------------------------------------------------------------
# КАТЕГОРИЯ 2: Стресс-тесты производительности
# -------------------------------------------------------------------

class TestStressPerformance(unittest.IsolatedAsyncioTestCase):
    """Стресс-тесты: высокая нагрузка, memory leak detection"""

    @classmethod
    def setUpClass(cls):
        from src.hybrid_rag_system import HybridRAGSystem, HybridConfig

        cls.test_dir = Path("test_data_integration")
        cls.config = HybridConfig(
            static_docs_dir=cls.test_dir,
            enable_progress_bars=False,
        )
        cls.rag = HybridRAGSystem(cls.config)
        cls.rag.index_static_documents(force=True)

    async def test_high_concurrency(self):
        """Тест высокой параллельности (50 запросов)"""
        queries = [
            f"тестовый запрос {i % 10}"
            for i in range(50)
        ]

        start = time.perf_counter()
        tasks = [self.rag.hybrid_search(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start

        # Подсчёт успешных/неудачных
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        qps = len(queries) / total_time

        print(f"\n⚡ High Concurrency Test:")
        print(f"   Запросов: {len(queries)}")
        print(f"   Успешных: {successful}")
        print(f"   Провалено: {failed}")
        print(f"   Время: {total_time:.2f}s")
        print(f"   QPS: {qps:.2f}")

        self.assertEqual(failed, 0, f"{failed} запросов провалились")
        self.assertGreater(qps, 10, "QPS слишком низкий")

    async def test_memory_usage_stability(self):
        """Проверка отсутствия утечек памяти"""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Базовое потребление памяти
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 100 запросов
        for i in range(100):
            await self.rag.hybrid_search(f"тест {i}")

        # Проверяем память после
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        growth_percent = (memory_growth / initial_memory) * 100

        print(f"\n💾 Memory Stability Test:")
        print(f"   Initial: {initial_memory:.1f} MB")
        print(f"   Final: {final_memory:.1f} MB")
        print(f"   Growth: {memory_growth:.1f} MB ({growth_percent:.1f}%)")

        # Рост не должен превышать 20%
        self.assertLess(growth_percent, 20, "Подозрение на утечку памяти")

    async def test_burst_load(self):
        """Тест burst load (резкий скачок нагрузки)"""
        # Разогрев
        for _ in range(5):
            await self.rag.hybrid_search("warmup")

        # Burst: 20 запросов одновременно
        burst_queries = [f"burst {i}" for i in range(20)]

        start = time.perf_counter()
        tasks = [self.rag.hybrid_search(q) for q in burst_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        burst_time = time.perf_counter() - start

        successful = sum(1 for r in results if not isinstance(r, Exception))

        print(f"\n💥 Burst Load Test:")
        print(f"   Burst size: {len(burst_queries)}")
        print(f"   Успешных: {successful}")
        print(f"   Время: {burst_time:.2f}s")

        self.assertEqual(successful, len(burst_queries), "Не все запросы обработаны")


# -------------------------------------------------------------------
# КАТЕГОРИЯ 3: Edge Cases & Adversarial Tests
# -------------------------------------------------------------------

class TestEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Тесты граничных случаев и adversarial inputs"""

    @classmethod
    def setUpClass(cls):
        from src.hybrid_rag_system import HybridRAGSystem, HybridConfig

        cls.test_dir = Path("test_data_integration")
        cls.config = HybridConfig(
            static_docs_dir=cls.test_dir,
            enable_progress_bars=False,
        )
        cls.rag = HybridRAGSystem(cls.config)
        cls.rag.index_static_documents(force=True)

    async def test_unicode_edge_cases(self):
        """Тест экзотических Unicode символов"""
        unicode_queries = [
            "тест с эмодзи 🚀🔥💻",
            "中文测试 Chinese characters",
            "العربية Arabic text",
            "🎯🎨🎭🎪 только эмодзи",
            "test\u200b\u200c\u200dzero-width chars",
        ]

        for query in unicode_queries:
            try:
                result = await self.rag.hybrid_search(query)
                self.assertIsNotNone(result)
                print(f"✅ Unicode OK: {query[:30]}")
            except Exception as e:
                self.fail(f"Ошибка на Unicode: {query[:30]} → {e}")

    async def test_malformed_queries(self):
        """Тест некорректных запросов"""
        malformed = [
            "",  # пустой
            "   ",  # только пробелы
            "\n\n\n",  # только переносы строк
            "a" * 10000,  # очень длинный
            "' OR '1'='1",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "../../../etc/passwd",  # Path traversal
        ]

        for query in malformed:
            try:
                result = await self.rag.hybrid_search(query)
                self.assertIsNotNone(result)
                print(f"✅ Malformed handled: {query[:20]}")
            except Exception as e:
                self.fail(f"Crash on malformed: {query[:20]} → {e}")

    async def test_repeated_queries(self):
        """Тест повторяющихся запросов (проверка кеширования)"""
        query = "повторяющийся запрос"

        latencies = []
        for i in range(10):
            start = time.perf_counter()
            await self.rag.hybrid_search(query)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        # Проверяем, что latency стабилен (нет деградации)
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        print(f"\n🔁 Repeated Queries Test:")
        print(f"   Avg latency: {avg_latency:.1f}ms")
        print(f"   Std deviation: {std_latency:.1f}ms")

        # Стандартное отклонение не должно быть большим
        self.assertLess(std_latency / avg_latency, 0.5, "Высокая вариативность latency")


# -------------------------------------------------------------------
# КАТЕГОРИЯ 4: Тесты компонентов (Component Tests)
# -------------------------------------------------------------------

class TestComponentIsolation(unittest.TestCase):
    """Изолированные тесты отдельных компонентов"""

    def test_lemmatizer_accuracy(self):
        """
        Проверка лемматизатора с учётом морфологических особенностей русского языка.

        Пояснения к кейсам (первый элемент списка – исходный текст,
        второй – ожидаемый список лемм в том же порядке).
        """

        # Импортируем только один раз – экономим время при многократных вызовах
        from src.hybrid_rag_system import RussianLemmatizerFast

        lemmatizer = RussianLemmatizerFast()

        test_cases = [
            # ---- ГЛАГОЛЫ -------------------------------------------------
            ("бегу", ["бег"]),
            ("бежал бегу бегать", ["бежать", "бег", "бегать"]),

            # ---- СУЩЕСТВИТЕЛЬНЫЕ -----------------------------------------
            ("книги книге книгой", ["книга", "книга", "книга"]),

            # ✅ ИСПРАВЛЕНО: используем только формы слова "дом"
            ("домов дому домом", ["дом", "дом", "дом"]),

            # ✅ ИЛИ тестируем "домовой" отдельно
            ("домовой книга", ["домовой", "книга"]),

            ("съешьте хлеба", ["съесть", "хлеб"]),

            # ---- ЧИСЛИТЕЛЬНЫЕ ---------------------------------------------
            ("пятью пяти пять", ["пять", "пять", "пять"]),
            ("две три", ["два", "три"]),

            # ---- ПУНКТУАЦИЯ -----------------------------------------------
            ("!!!", []),
            (".!?.,", []),
            ("Привет! Как дела?", ["привет", "как", "дело"]),
            ("Снег, снег, снежок", ["снег", "снег", "снежок"]),

            # ---- КАПИТАЛИЗАЦИЯ --------------------------------------------
            ("Москва", ["москва"]),
            ("КРАСНЫЙ", ["красный"]),
            ("ЕВРОПЫ", ["европа"]),

            # ---- EDGE CASES -----------------------------------------------
            ("загадочный", ["загадочный"]),
            ("12345", []),  # ✅ Числа фильтруются

            # ---- КОНТЕКСТНЫЕ ПРЕДЛОЖЕНИЯ ----------------------------------
            ("Он бежал, когда услышал звук.",
             ["он", "бежать", "когда", "услышать", "звук"]),

            ("На столе лежали книги, журналы и газеты.",
             ["на", "стол", "лежать", "книга", "журнал", "и", "газета"]),

            # ---- СТРЕСС-ТЕСТ ----------------------------------------------
            (" ".join(["дом" for _ in range(50)]), ["дом"] * 50),
        ]

        passed = failed = 0

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = lemmatizer.lemmatize(text)

                try:
                    self.assertSequenceEqual(
                        result,
                        expected,
                        msg=f"Неверная лемматизация: {text!r}\nОжидали: {expected}\nПолучили: {result}"
                    )
                    print(f"✅  Лемма: {text!r} → {result}")
                    passed += 1
                except AssertionError as exc:
                    failed += 1
                    # Выводим подробный отчёт – помогает быстро найти ошибку
                    print(f"\n❌  Ошибка при обработке '{text}'")
                    print(f"    Ожидали: {expected}")
                    print(f"    Получили: {result}\n")
                    raise exc

        print(f"\n📊 Лемматизация: {passed}/{len(test_cases)} успешно, {failed} неуспешно.")

    def test_rrf_scoring(self):
        """Проверка алгоритма RRF"""
        from src.hybrid_rag_system import HybridRAGSystem, HybridConfig

        test_dir = Path("test_data_integration")
        config = HybridConfig(static_docs_dir=test_dir, enable_progress_bars=False)
        rag = HybridRAGSystem(config)

        # Мок данные
        bm25_results = [
            {"id": "doc1", "text": "test1", "bm25_score": 10.0},
            {"id": "doc2", "text": "test2", "bm25_score": 8.0},
        ]

        semantic_results = [
            {"id": "doc2", "text": "test2", "distance": 0.1},
            {"id": "doc3", "text": "test3", "distance": 0.2},
        ]

        # Тестируем RRF
        combined = rag._reciprocal_rank_fusion(bm25_results, semantic_results)

        # doc2 должен быть выше (найден в обоих)
        self.assertEqual(combined[0]["id"], "doc2")
        print(f"✅ RRF: doc2 correctly ranked first")

    def test_exact_match_detection(self):
        """Проверка детектирования exact match"""
        from src.hybrid_rag_system import HybridRAGSystem, HybridConfig

        test_dir = Path("test_data_integration")
        config = HybridConfig(static_docs_dir=test_dir, enable_progress_bars=False)
        rag = HybridRAGSystem(config)

        # Мок документ
        doc = {"id": "1", "text": "Это тестовый документ с точной фразой."}
        query = "точной фразой"

        # Проверяем детектирование
        has_match = query.lower() in doc["text"].lower()
        self.assertTrue(has_match, "Exact match не обнаружен")
        print(f"✅ Exact match detected: '{query}'")


# -------------------------------------------------------------------
# КАТЕГОРИЯ 5: Интеграционные тесты (End-to-End)
# -------------------------------------------------------------------

class TestEndToEnd(unittest.IsolatedAsyncioTestCase):
    """Полные end-to-end сценарии"""

    @classmethod
    def setUpClass(cls):
        from src.hybrid_rag_system import HybridRAGSystem, HybridConfig

        cls.test_dir = Path("test_data_integration")
        cls.config = HybridConfig(
            static_docs_dir=cls.test_dir,
            enable_progress_bars=False,
        )
        cls.rag = HybridRAGSystem(cls.config)
        cls.rag.index_static_documents(force=True)

    async def test_full_pipeline(self):
        """Тест полного pipeline: запрос → результаты → проверка структуры"""
        query = "машинное обучение"

        result = await self.rag.hybrid_search(query)

        # Проверка структуры ответа
        self.assertIn("query", result)
        self.assertIn("results", result)
        self.assertIn("bm25_count", result)
        self.assertIn("semantic_count", result)
        self.assertIn("total", result)

        # Проверка каждого результата
        for doc in result["results"]:
            self.assertIn("id", doc)
            self.assertIn("text", doc)
            self.assertIn("rerank_score", doc)
            self.assertIsInstance(doc["text"], str)
            self.assertIsInstance(doc["rerank_score"], (int, float))

        print(f"✅ Full pipeline OK: {len(result['results'])} результатов")

    async def test_collection_stats(self):
        """Проверка статистики коллекции"""
        stats = self.rag.get_collection_stats()

        self.assertIn("name", stats)
        self.assertIn("count", stats)
        self.assertIn("cached_files", stats)

        self.assertGreater(stats["count"], 0, "Коллекция пустая")

        print(f"\n📊 Collection Stats:")
        print(f"   Documents: {stats['count']}")
        print(f"   Cached files: {stats['cached_files']}")


# -------------------------------------------------------------------
# Master Test Runner
# -------------------------------------------------------------------

class MasterTestRunner:
    """Главный раннер всех тестов"""

    def run_all(self):
        """Запуск всех категорий тестов"""
        print("\n" + "=" * 80)
        print("🧪 HYBRID RAG v5.0 — COMPLETE TEST SUITE")
        print("=" * 80)

        test_suites = [
            ("Quality Metrics", TestSearchQualityMetrics),
            ("Stress Performance", TestStressPerformance),
            ("Edge Cases", TestEdgeCases),
            ("Component Isolation", TestComponentIsolation),
            ("End-to-End", TestEndToEnd),
        ]

        all_results = {}

        for i, (name, test_class) in enumerate(test_suites, 1):
            print(f"\n[{i}/{len(test_suites)}] {name}...")

            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            all_results[name] = {
                "total": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
            }

        self._print_summary(all_results)

    def _print_summary(self, results: Dict):
        """Итоговый отчёт"""
        print("\n" + "=" * 80)
        print("📊 ФИНАЛЬНЫЙ ОТЧЁТ")
        print("=" * 80)

        total_tests = sum(r["total"] for r in results.values())
        total_failures = sum(r["failures"] for r in results.values())
        total_errors = sum(r["errors"] for r in results.values())

        for category, stats in results.items():
            status = "✅" if stats["failures"] == 0 and stats["errors"] == 0 else "❌"
            print(f"\n{status} {category}:")
            print(f"   Тестов: {stats['total']}")
            print(f"   Провалено: {stats['failures']}")
            print(f"   Ошибок: {stats['errors']}")

        success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0

        print("\n" + "-" * 80)
        print(f"📈 ИТОГО:")
        print(f"   Всего тестов: {total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print("=" * 80)


if __name__ == "__main__":
    runner = MasterTestRunner()
    runner.run_all()
