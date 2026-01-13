#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Локальный AI-агент на базе FunctionGemma (via llama.cpp).
Улучшенная версия с "заземлением" (Grounding) знаний о Windows-приложениях.
"""

import json
import sys
import logging
import argparse
import subprocess
import webbrowser
import shlex
from typing import Dict, Any, Optional, Callable

# Проверка наличия библиотеки
try:
    from llama_cpp import Llama
except ImportError:
    sys.exit("Ошибка: Библиотека llama-cpp-python не установлена.\nВыполните: pip install llama-cpp-python")

# ------------------------------------------------------------------
# 1. Настройки и Константы
# ------------------------------------------------------------------

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Agent")

# Путь к модели (Укажите ваш актуальный путь)
DEFAULT_MODEL_PATH = "g:\\AIModels\\lmstudio\\models\\unsloth\\functiongemma-270m-it-GGUF\\functiongemma-270m-it-UD-Q8_K_XL.gguf"

# ------------------------------------------------------------------
# 2. Определение инструментов и знаний (Schema & Knowledge)
# ------------------------------------------------------------------

# "Шпаргалка" для модели, чтобы она знала названия исполняемых файлов
APP_KNOWLEDGE_BASE = """
WINDOWS APP KNOWLEDGE BASE (Use these commands in run_command):
- Notepad (Блокнот)      -> command: 'notepad.exe'
- Calculator (Калькулятор)-> command: 'calc.exe'
- Paint                  -> command: 'mspaint.exe'
- Explorer (Проводник)   -> command: 'explorer.exe'
- Command Prompt (CMD)   -> command: 'start cmd'
- VS Code (IDE)          -> command: 'code'
- System Settings        -> command: 'start ms-settings:'
"""

# JSON схема инструментов
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Открывает веб-сайт в браузере по умолчанию.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL адрес (например, https://google.com)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Запускает приложения Windows или выполняет консольные команды.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Имя файла .exe или команда (например: 'notepad.exe', 'ping google.com')."
                    }
                },
                "required": ["command"]
            }
        }
    }
]

# ------------------------------------------------------------------
# 3. Реализация функций (Tools Implementation)
# ------------------------------------------------------------------

def open_url(url: str) -> str:
    """Открывает ссылку в браузере."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        webbrowser.open(url, new=2)
        return f"[✓] Браузер запущен: {url}"
    except Exception as exc:
        return f"[✗] Ошибка открытия URL: {exc}"

def run_command(command: str) -> str:
    """
    Выполняет команду с подтверждением пользователя.
    """
    print(f"\n[⚠️ SECURITY CHECK] Агент хочет выполнить команду:")
    print(f">> {command}")

    # Простая эвристика: если это просто запуск известного exe, помечаем как Safe
    is_safe = command.lower() in ['notepad.exe', 'calc.exe', 'mspaint.exe', 'explorer.exe', 'code']
    hint = "(Безопасное приложение)" if is_safe else "(Системная команда)"

    confirm = input(f"Разрешить {hint}? (y/n): ").strip().lower()
    if confirm != 'y':
        return "[🚫] Отмена: Пользователь отклонил выполнение."

    try:
        # Используем shlex.split для правильной обработки аргументов (кроме shell=True сложных кейсов)
        # Но для Windows и shell=True лучше передавать строку целиком.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Если команда (например, notepad) запустила GUI и вернула управление не сразу,
        # subprocess может ждать закрытия программы.
        # В реальных агентах часто используют subprocess.Popen для GUI приложений, чтобы не блокировать скрипт.

        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            if not output: return "[✓] Команда запущена успешно (вывода нет)."
            return f"[✓] Вывод:\n{output}"
        else:
            return f"[✗] Ошибка (код {result.returncode}): {error}"

    except subprocess.TimeoutExpired:
        return "[⚠] Тайм-аут: Приложение запущено или команда выполняется слишком долго."
    except Exception as exc:
        return f"[✗] Системная ошибка: {exc}"

FUNCTION_MAP: Dict[str, Callable] = {
    "open_url": open_url,
    "run_command": run_command
}

# ------------------------------------------------------------------
# 4. Класс Агента (LLM Wrapper)
# ------------------------------------------------------------------

class LocalAgent:
    def __init__(self, model_path: str, n_ctx: int = 2048):
        logger.info(f"Инициализация модели: {model_path}")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=0,  # 0 для CPU, поставьте -1 для всех слоев на GPU
                verbose=False    # Отключаем спам в консоль от llama.cpp
            )
        except Exception as e:
            logger.error(f"FATAL: Не удалось загрузить модель.\n{e}")
            sys.exit(1)

    def generate_action(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Формирует промпт и получает JSON-действие от модели.
        """

        # Собираем системный промпт
        tools_json = json.dumps(TOOLS_SCHEMA, indent=2, ensure_ascii=False)

        system_content = (
            "You are a helpful Windows PC assistant.\n"
            "You have access to the following tools:\n"
            f"{tools_json}\n\n"
            f"{APP_KNOWLEDGE_BASE}\n"
            "INSTRUCTIONS:\n"
            "1. To open a website, use 'open_url'.\n"
            "2. To open a desktop app (Notepad, Calculator), use 'run_command' with the filename from the Knowledge Base.\n"
            "3. You must respond with a valid JSON object ONLY. No commentary."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query}
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.1,  # Минимальная температура для строгости
                max_tokens=256,
                response_format={"type": "json_object"} # Принудительный JSON (Grammar)
            )

            raw_content = response["choices"][0]["message"]["content"]

            # Попытка распарсить JSON
            try:
                action_data = json.loads(raw_content)
                return action_data
            except json.JSONDecodeError:
                # Иногда модель может добавить лишние пробелы или md-теги
                clean_content = raw_content.replace("```json", "").replace("```", "").strip()
                try:
                    return json.loads(clean_content)
                except:
                    logger.warning(f"Не удалось распарсить JSON: {raw_content}")
                    return None

        except Exception as e:
            logger.error(f"Ошибка инференса: {e}")
            return None

# ------------------------------------------------------------------
# 5. Главный цикл (Main Loop)
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PC Agent v2")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Путь к GGUF файлу")
    args = parser.parse_args()

    print("--- Запуск AI Агента ---")
    agent = LocalAgent(model_path=args.model)

    print("\nГотов к работе. Введите команду (например: 'Открой блокнот', 'Запусти youtube').")
    print("Для выхода введите 'exit'.\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit", "выход"]:
                print("Завершение работы.")
                break

            # Генерация действия
            action = agent.generate_action(user_input)

            if not action:
                print("Assistant: (Я не понял запрос или не смог выбрать инструмент)")
                continue

            # Нормализация ключей (некоторые модели путают name/function)
            func_name = action.get("name") or action.get("function")
            arguments = action.get("arguments") or action.get("parameters") or {}

            # Проверка существования функции
            if func_name in FUNCTION_MAP:
                logger.info(f"Вызов: {func_name} | Аргументы: {arguments}")

                # Выполнение функции
                try:
                    result = FUNCTION_MAP[func_name](**arguments)
                    print(result)
                except TypeError as e:
                    print(f"[!] Ошибка аргументов функции: {e}")
            else:
                # Если модель решила просто поболтать (вернула JSON с полем message или подобным)
                if "message" in action:
                    print(f"Assistant: {action['message']}")
                else:
                    print(f"[!] Модель попыталась вызвать несуществующую функцию: '{func_name}'")

        except KeyboardInterrupt:
            print("\nПринудительная остановка.")
            break
        except Exception as e:
            logger.exception("Непредвиденная ошибка в главном цикле:")

if __name__ == "__main__":
    main()
