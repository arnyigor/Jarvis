import json
import asyncio
from typing import List, Dict, Callable
from openai import AsyncOpenAI

class AgentBrain:
    def __init__(self, config, rag_system):
        self.rag = rag_system
        # Клиент для LM Studio
        self.client = AsyncOpenAI(base_url=config.llm_url, api_key="lm-studio")

    async def run(self, query: str, callback: Callable = None) -> Dict:
        """
        Запускает цикл мышления.
        """
        # --- ШАГ 1: Поиск (Retrieval) ---
        if callback: callback("Retrieval", "Searching knowledge base...")

        # Выполняем поиск
        docs = self.rag.search(query)

        # !!! ГЛАВНОЕ ИЗМЕНЕНИЕ !!!
        # Формируем предпросмотр чанков для UI
        if docs:
            preview_lines = []
            for i, doc in enumerate(docs[:3]): # Берем топ-3 для показа
                # Обрезаем текст до 150 символов и убираем переносы строк для красоты
                snippet = doc['text'][:150].replace("\n", " ")
                score = doc.get('score', 0.0)
                preview_lines.append(f"> **Doc {i+1}** (score: {score:.2f}): _{snippet}..._")

            docs_preview = "\n\n".join(preview_lines)

            # Отправляем чанки в UI
            if callback: callback("Found Context", f"\n\n{docs_preview}")
        else:
            if callback: callback("Retrieval", "No relevant documents found.")

        # Подготовка контекста для LLM
        context_str = "\n".join([f"- {d['text']}" for d in docs])

        # --- ШАГ 2: Формирование промпта ---
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question.\n"
            "If the context is insufficient, rely on your general knowledge but mention it.\n"
            "Format your answer in Markdown."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        # --- ШАГ 3: Генерация (Thinking) ---
        if callback: callback("Thinking", "Analyzing context and generating answer...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="gpt-oss-20b",
                messages=messages,
                temperature=0.7
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error communicating with LLM: {e}"

        return {
            "answer": answer,
            "sources": docs
        }
