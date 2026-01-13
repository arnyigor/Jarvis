# memory.py
from collections import deque
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Загрузить один раз при старте
# Для Qwen‑3‑Next (пример)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
# Загрузить один раз глобально
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")

def get_embedding(text: str) -> list:
    return embedding_model.encode([text], normalize_embeddings=True)[0].tolist()

class WorkingMemory:
    def __init__(self, max_turns=10, max_tokens=4096):
        self.buffer = deque(maxlen=max_turns*2)
        self.max_tokens = max_tokens

    def push(self, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        self.buffer.append(msg)

    def get_recent(self, token_limit: int) -> List[Dict]:
        # Начинаем с конца и идём назад
        tokens_used = 0
        result = []

        for msg in reversed(self.buffer):
            msg_tokens = len(tokenizer.encode(msg["content"], add_special_tokens=False))
            if tokens_used + msg_tokens > token_limit:
                break
            result.append(msg)
            tokens_used += msg_tokens

        return list(reversed(result))  # Вернуть в исходном порядке

class EpisodicMemory:
    """Сводки старых фрагментов беседы."""
    def __init__(self):
        self.summaries: List[Dict[str, Any]] = []

    def add(self, summary_text: str) -> None:
        self.summaries.append({"summary": summary_text,
                               "timestamp": time.time()})

class LongTermMemory:
    """Обёртка ChromaDB (векторное хранилище)."""
    def __init__(self, chroma_path: Path):
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(
            name="jarvis",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, doc_id: str, text: str) -> None:
        embedding = get_embedding(text)  # sentence‑transformer
        self.collection.add(
            documents=[text],
            ids=[doc_id],
            embeddings=[embedding]
        )

    def query(self, query_text: str, k=5) -> List[Dict[str, Any]]:
        # Убедитесь, что у вас есть функция встраивания!
        emb = get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=k,
            include=["documents", "distances"]
        )
        return [
            {
                "id": r["ids"][0],
                "score": r["distances"][0] if r["distances"] else 0.0,
                "text": r["documents"][0]
            }
            for r in results
        ]