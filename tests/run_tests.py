import asyncio
import time
import numpy as np
import uuid
import sys
import tempfile
from pathlib import Path

# --- ИМПОРТЫ ВАШЕЙ СИСТЕМЫ ---
# Предполагается, что ваш класс лежит в src/hybrid_rag_system.py
try:
    from src.hybrid_rag_system import HybridRAGSystem, HybridConfig
except ImportError:
    # Если файл лежит рядом, пробуем прямой импорт (для теста)
    sys.path.append(".")
    from src.hybrid_rag_system import HybridRAGSystem, HybridConfig

# --- ВАШ КОД ТЕСТОВ (Stress Worker) ---
async def stress_worker(rag, queries, duration, results_list):
    end_time = time.time() + duration
    count = 0
    while time.time() < end_time:
        q = queries[count % len(queries)]
        start = time.perf_counter()
        try:
            # Вызов поиска (предполагаем метод hybrid_search или search)
            # Если у вас метод называется иначе (например asearch), поправьте тут:
            if hasattr(rag, 'hybrid_search'):
                await rag.hybrid_search(q)
            else:
                await rag.search(q) # Fallback

            latency = (time.perf_counter() - start) * 1000
            results_list.append({"status": "ok", "latency": latency})
        except Exception as e:
            results_list.append({"status": "error", "error": str(e)})
        count += 1
        await asyncio.sleep(0.01)

# --- ВАШ КОД ТЕСТОВ (Ramp Up) ---
async def run_ramp_up_test(rag, max_users=10, step_duration=5):
    print(f"\n📈 ЗАПУСК STRESS TEST (Макс. юзеров: {max_users})")
    queries = ["Пьер Безухов", "Андрей Болконский", "Наташа Ростова", "война 1812", "Наполеон"]

    for users in range(1, max_users + 1, 2):
        print(f"   🌊 Нагрузка: {users} одновременных пользователей...")
        results = []
        tasks = [stress_worker(rag, queries, step_duration, results) for _ in range(users)]

        start_step = time.time()
        await asyncio.gather(*tasks)

        latencies = [r["latency"] for r in results if r["status"] == "ok"]
        errors = [r for r in results if r["status"] == "error"]

        if not latencies:
            print("      ❌ Нет успешных запросов!")
            continue

        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        rps = len(latencies) / step_duration

        print(f"      RPS: {rps:.2f} req/s | p95: {p95:.0f}ms | p99: {p99:.0f}ms | Errors: {len(errors)}")

        if p95 > 3000: # Лимит 3 сек
            print("      ⚠️ Латентность слишком высока! Остановка теста.")
            break

# --- ВАШ КОД ТЕСТОВ (Needle) ---
class NeedleInHaystackTest:
    def __init__(self, system: HybridRAGSystem):
        self.system = system

    async def run(self, haystack_file_path: Path):
        secret_code = str(uuid.uuid4())[:8]
        needle_text = f"Секретный код запуска системы защиты Пьера Безухова: {secret_code}."

        print(f"\n🧪 ЗАПУСК 'Needle in a Haystack'...")
        print(f"   Иголка: '{needle_text}'")

        if not haystack_file_path.exists():
            print(f"   ❌ Ошибка: Файл {haystack_file_path} не найден!")
            return

        original_text = haystack_file_path.read_text(encoding="utf-8")
        insert_pos = len(original_text) // 2
        modified_text = original_text[:insert_pos] + "\n\n" + needle_text + "\n\n" + original_text[insert_pos:]

        # Создаем временный файл в папке docs, чтобы система его подхватила
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                         dir=self.system.config.static_docs_dir, encoding='utf-8') as tmp:
            tmp.write(modified_text)
            tmp_path = Path(tmp.name)

        try:
            print("   🔄 Переиндексация (force=True)... Это может занять время.")
            # Важно: запускаем синхронную индексацию в executor, если она блокирующая,
            # но здесь просто вызовем напрямую, так как это тест
            self.system.index_static_documents(force=True)

            query = "Какой секретный код запуска у Пьера Безухова?"
            print(f"   🔍 Поиск: '{query}'")

            # Адаптация под возвращаемый формат вашей системы
            results = await self.system.hybrid_search(query) # Или просто search(query)

            # Предполагаем, что results - это dict c ключом 'results' или список
            if isinstance(results, dict) and 'results' in results:
                items = results['results']
            else:
                items = results

            found = False
            # Проверяем топ-5
            for i, item in enumerate(items[:5]):
                text = item.get('text', '') if isinstance(item, dict) else str(item)
                if secret_code in text:
                    print(f"   ✅ УСПЕХ! Найдено на позиции #{i+1}")
                    print(f"      Контекст: {text[:100]}...")
                    found = True
                    break

            if not found:
                print("   ❌ ПРОВАЛ. Иголка не найдена в топ-5.")
                if items:
                    print(f"      Топ-1 был: {items[0].get('text', '')[:50]}...")

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                print("   🧹 Временный файл удален.")

# --- ТОЧКА ВХОДА (MAIN) ---
async def main():
    # 1. Настройка путей
    base_dir = Path(".")
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Создадим фиктивный файл "Война и мир", если его нет, для теста
    haystack_file = docs_dir / "war_and_peace.txt"
    if not haystack_file.exists():
        print("⚠️ Файл war_and_peace.txt не найден, создаем тестовый...")
        haystack_file.write_text("Много текста " * 5000, encoding='utf-8')

    # 2. Инициализация системы
    config = HybridConfig(
        static_docs_dir=docs_dir,
        chromadb_dir=base_dir / "chromadb_test",
    )

    print("⚙️ Инициализация RAG системы...")
    rag = HybridRAGSystem(config)

    # 3. Запуск теста "Иголка"
    needle_test = NeedleInHaystackTest(rag)
    await needle_test.run(haystack_file)

    # 4. Запуск стресс-теста
    # Сначала проиндексируем обычный контент, чтобы было что искать
    print("\n⚙️ Подготовка к стресс-тесту (индексация чистого файла)...")
    rag.index_static_documents(force=True)

    await run_ramp_up_test(rag, max_users=5, step_duration=5)

if __name__ == "__main__":
    asyncio.run(main())
