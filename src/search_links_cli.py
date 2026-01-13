import argparse
import json
import random
import sys
import time
import urllib.parse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# Функция для логирования в stderr, чтобы не засорять stdout (где будет JSON)
def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_links_from_yandex(queries, headless=False, global_unique=True, limit=5):
    """
    Основная функция скрапинга.
    """
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")

    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    # Отключаем лишние логи самого драйвера
    chrome_options.add_argument("--log-level=3")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    results = {}
    global_seen_urls = set()

    try:
        log(f"🚀 Начинаю поиск по {len(queries)} запросам...")

        for i, query in enumerate(queries):
            log(f"\n[{i + 1}/{len(queries)}] 🔎 Ищу: {query}")

            encoded_query = urllib.parse.quote(query)
            url = f"https://yandex.ru/search/?text={encoded_query}&lr=213"

            driver.get(url)

            try:
                wait = WebDriverWait(driver, 15)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.serp-list")))

                links_elements = driver.find_elements(By.CSS_SELECTOR, "li.serp-item a.organic__url")

                unique_urls_for_query = []
                seen_in_this_query = set()

                for el in links_elements:
                    href = el.get_attribute("href")

                    if href and "yandex.ru" not in href and "http" in href:
                        clean_href = href.rstrip('/')

                        if clean_href in seen_in_this_query:
                            continue
                        if global_unique and clean_href in global_seen_urls:
                            continue

                        unique_urls_for_query.append(href)
                        seen_in_this_query.add(clean_href)
                        global_seen_urls.add(clean_href)

                top_links = unique_urls_for_query[:limit]
                results[query] = top_links

                log(f"✅ Найдено: {len(top_links)}")

            except Exception as e:
                log(f"⚠️ Ошибка: {e}")

            if i < len(queries) - 1:
                sleep_time = random.uniform(2, 5)
                log(f"⏳ Жду {sleep_time:.1f} сек...")
                time.sleep(sleep_time)

    finally:
        log("🏁 Браузер закрыт.")
        driver.quit()

    return results


if __name__ == "__main__":
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Yandex Scraper для RAG пайплайна")

    # Позиционные аргументы (сами запросы)
    parser.add_argument('queries', metavar='Query', type=str, nargs='+',
                        help='Список поисковых запросов')

    # Опциональные флаги
    parser.add_argument('--headless', action='store_true', help='Запуск без окна браузера')
    parser.add_argument('--no-unique', action='store_false', dest='unique',
                        help='Отключить глобальную уникальность ссылок')
    parser.add_argument('--limit', type=int, default=5, help='Количество ссылок на один запрос')
    parser.add_argument('--output', type=str, default=None, help='Путь к файлу для сохранения (опционально)')

    args = parser.parse_args()

    # Запуск логики
    data = get_links_from_yandex(
        queries=args.queries,
        headless=args.headless,
        global_unique=args.unique,
        limit=args.limit
    )

    # ВЫВОД РЕЗУЛЬТАТА

    # 1. Формируем JSON строку
    json_output = json.dumps(data, indent=4, ensure_ascii=False)

    # 2. Если указан файл --output, пишем туда
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        log(f"\n💾 Результат сохранен в файл: {args.output}")

    # 3. ВСЕГДА печатаем JSON в stdout (для пайплайнов)
    # Если вы не хотите видеть дублирование при сохранении в файл, можно добавить условие.
    # Но для unix-way обычно печатают в stdout.
    print(json_output)
