import time
import random
import json
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def get_links_from_yandex(queries, headless=False, global_unique=False):
    """
    Ищет список запросов в Яндексе и возвращает словарь {вопрос: [ссылки]}.

    Args:
        queries (list): Список запросов.
        headless (bool): Скрывать ли браузер.
        global_unique (bool): Если True, ссылка, найденная в первом запросе,
                              не попадет в результаты второго запроса.
                              Полезно для сбора базы RAG без повторов.
    """

    # Настройка браузера
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")

    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    results = {}

    # Множество для отслеживания всех ссылок за сессию (если нужен global_unique)
    global_seen_urls = set()

    try:
        print(f"🚀 Начинаю поиск по {len(queries)} запросам...")

        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] 🔎 Ищу: {query}")

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

                    # Базовая фильтрация
                    if href and "yandex.ru" not in href and "http" in href:
                        # Нормализация: убираем лишний слеш в конце для корректного сравнения
                        clean_href = href.rstrip('/')

                        # Проверка 1: нет ли дублей внутри текущей выдачи (бывает реклама + органика)
                        if clean_href in seen_in_this_query:
                            continue

                        # Проверка 2: глобальная уникальность (если включена)
                        if global_unique and clean_href in global_seen_urls:
                            continue

                        # Если все ок, добавляем
                        unique_urls_for_query.append(href) # Сохраняем оригинальный href
                        seen_in_this_query.add(clean_href)
                        global_seen_urls.add(clean_href)

                # Берем топ-5 уже ПОСЛЕ удаления дубликатов
                top_links = unique_urls_for_query[:5]
                results[query] = top_links

                print(f"✅ Найдено уникальных ссылок: {len(top_links)}")
                for link in top_links:
                    print(f"   🔗 {link}")

            except Exception as e:
                print(f"⚠️ Ошибка при обработке запроса: {e}")

            sleep_time = random.uniform(3, 7)
            print(f"⏳ Жду {sleep_time:.1f} сек...")
            time.sleep(sleep_time)

    finally:
        print("\n🏁 Завершение работы, закрываю браузер.")
        driver.quit()

    return results

if __name__ == "__main__":
    questions = [
        "SpaceX Starship Flight 6 Mechazilla capture",
        "SpaceX Starship Flight 6 Mechazilla not captured reason",
        "SpaceX Starship Flight 6 heat shield changes vs previous flight",
        "SpaceX Starship Flight 6 launch to splashdown duration",
        "SpaceX Starship Flight 7 Mechazilla capture",
        "SpaceX Starship Flight 7 heat shield modifications",
        "SpaceX Starship Flight 7 launch to splashdown duration"
    ]

    # global_unique=True полезен, если вы хотите собрать датасет и не качать одно и то же.
    # Если нужно топ-5 для КАЖДОГО запроса независимо (даже если они повторяются), ставьте False.
    data = get_links_from_yandex(questions, headless=False, global_unique=True)

    with open("search_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("\n💾 Результаты сохранены в search_results.json")
