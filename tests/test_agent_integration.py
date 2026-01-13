# tests/test_agent_integration.py

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agent import AgentBrain
from src.memory import WorkingMemory, EpisodicMemory


@pytest.fixture
def mock_llm():
    """Подменяем LLM‑объект. Возвращает фиксированную строку."""
    llm = MagicMock()
    async def dummy_generate(prompt: str) -> str:
        # Для простоты просто возвращаем «Hello» + часть промпта
        return f"Generated reply for: {prompt[:30]}..."
    llm.generate.side_effect = dummy_generate
    return llm


@pytest.fixture
def agent_instance(mock_llm):
    """Создаём агент без реального Chroma и инструментов."""
    # Минимальный стек памяти (только в память)
    mem_stack = {
        "working": WorkingMemory(max_turns=3),   # небольшие лимиты, чтобы не держать слишком много
        "episodic": EpisodicMemory(),
        "rag": MagicMock()  # Мок – ни один документ не возвращается
    }

    # Регистрация одного инструмента (PythonExecTool)
    tool_registry = {
        "python_exec": PythonExecTool()
    }

    # Инициализируем модель‑менеджер, передавая mock‑LLM
    model_manager = MagicMock()
    model_manager.generate = mock_llm.generate

    agent = AgentBrain(
        memory_stack=mem_stack,
        tools=tool_registry,
        model=model_manager  # ваш интерфейс ModelManager / LLM обёртка
    )
    return agent


def test_basic_agent_response(agent_instance):
    """Проверяем, что агент отвечает на простой запрос."""
    user_query = "Привет, как дела?"
    # Запускаем асинхронный метод run()
    result = pytest.run_until_complete(
        agent_instance.run(user_query)
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert "Generated reply for" in result   # проверяем, что LLM действительно сгенерировал


def test_agent_tool_usage(agent_instance):
    """Проверим, что агент использует инструмент при необходимости."""
    # Запрос, который должен вызвать Python‑exec (вычисление sqrt(9))
    user_query = "Сколько корня из девяти?"
    result = pytest.run_until_complete(
        agent_instance.run(user_query)
    )

    # Внутри _think LLM выдаст план «Use python_exec» – проверяем, что ответ содержит результат вычисления
    assert "3.0" in result   # ожидаемый вывод sqrt(9)

