#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid RAG v5.0 – Защита от прерывания индексации, атомарность кеша и BM25.
"""
import asyncio
import hashlib
import io
import json
import logging
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import chromadb
import torch
import trafilatura
from chromadb.config import Settings
from docling.document_converter import DocumentConverter
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

warnings = None  # silence future warnings – same behaviour as original

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hybrid_rag.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# Пытаемся импортировать pymorphy3, но делаем это внутри функции инициализации,
# чтобы не грузить его сразу при импорте модуля (хотя это не критично, pymorphy легкий)
try:
    import pymorphy3
    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False

# Глобальная переменная для воркера
_morph_analyzer = None

def init_worker():
    """Эта функция вызывается один раз при старте каждого процесса"""
    global _morph_analyzer
    if PYMORPHY_AVAILABLE:
        _morph_analyzer = pymorphy3.MorphAnalyzer()

def lemmatize_text_worker(text: str) -> str:
    """Функция, которая будет выполняться в отдельном процессе"""
    global _morph_analyzer
    if _morph_analyzer is None:
        # Fallback, если вдруг инит не сработал (хотя должен)
        if PYMORPHY_AVAILABLE:
            _morph_analyzer = pymorphy3.MorphAnalyzer()
        else:
            return text # Возвращаем как есть

    # Логика лемматизации (копируем вашу)
    words = re.findall(r'\b\w+\b', text.lower())
    lemmas = []

    for word in words:
        if word.isdigit(): continue

        # Если это спецкод или англ - оставляем
        if re.match(r'[a-z0-9]+', word):
            lemmas.append(word)
            continue

        try:
            # Берем нормальную форму
            parsed = _morph_analyzer.parse(word)[0]
            lemma = parsed.normal_form
            if re.match(r'^[а-яёa-z0-9]+$', lemma, re.IGNORECASE):
                lemmas.append(lemma)
        except:
            lemmas.append(word)

    return " ".join(lemmas)


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def validate_gpu_availability() -> Tuple[bool, str]:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return True, f"{device_name} ({vram_total:.1f} GB VRAM)"
    else:
        return False, "CPU only (CUDA not available)"


def compute_file_hash(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash {file_path}: {e}")
        return ""


def check_disk_space(required_bytes: int, path: Path | str) -> None:
    """
    Проверяем, хватает ли диска для записи `required_bytes`.
    Если нет – бросаем RuntimeError.
    """
    stat = shutil.disk_usage(str(path))
    available = stat.free
    # 20 % запаса на случай «памяти под временные файлы»
    if available < required_bytes * 1.2:
        raise RuntimeError(
            f"Insufficient disk space: {available / 1e9:.1f} GB free "
            f"(needs ~{required_bytes / 1e9:.1f} GB)"
        )


# -------------------------------------------------------------------
# Lemmatizer
# -------------------------------------------------------------------
try:
    import pymorphy3  # ✅ Используем pymorphy3 вместо pymorphy2

    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False


class RussianLemmatizerFast:
    """Быстрый лемматизатор на pymorphy3"""

    def __init__(self):
        if not PYMORPHY_AVAILABLE:
            raise ImportError("pymorphy3 not installed. Install: pip install pymorphy3 pymorphy3-dicts-ru")

        logger.info("🔄 Initializing pymorphy3 lemmatizer...")
        self.morph = pymorphy3.MorphAnalyzer()
        logger.info("✅ Fast lemmatizer ready")

    def lemmatize(self, text: str) -> List[str]:
        """
        Лемматизация с фильтрацией пунктуации и чисел.

        Args:
            text: Входной текст

        Returns:
            Список лемм (только слова, без чисел и пунктуации)

        Examples:
            >>> lemmatizer = RussianLemmatizerFast()
            >>> lemmatizer.lemmatize("12345")
            []
            >>> lemmatizer.lemmatize("слово 123 текст")
            ['слово', 'текст']
        """
        # ✅ Извлекаем только буквенные токены
        words = re.findall(r'\b\w+\b', text.lower())
        lemmas = []

        for word in words:
            # ✅ Пропускаем чистые числа
            if word.isdigit():
                continue

            # ✅ Пропускаем слова без букв
            if not re.search(r'[а-яёa-z]', word, re.IGNORECASE):
                continue

            try:
                # Берем первый (самый вероятный) разбор
                parsed = self.morph.parse(word)[0]
                lemma = parsed.normal_form

                # ✅ Фильтруем: только буквы (БЕЗ цифр)
                if re.match(r'^[а-яёa-z]+$', lemma, re.IGNORECASE):
                    lemmas.append(lemma)
            except Exception:
                # Если не удалось разобрать, берем исходное слово
                if re.match(r'^[а-яёa-z]+$', word, re.IGNORECASE):
                    lemmas.append(word)

        return lemmas

    def lemmatize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Батчевая лемматизация.
        pymorphy3 не поддерживает нативный батчинг,
        но можно распараллелить через multiprocessing (опционально).
        """
        return [self.lemmatize(text) for text in texts]


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
@dataclass
class HybridConfig:
    static_docs_dir: Path

    # Models
    embedding_model: str = "intfloat/multilingual-e5-small"
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Text processing
    chunk_size: int = 512
    chunk_overlap: int = 200
    min_chunk_length: int = 100

    # Storage
    chromadb_dir: Path = field(default_factory=lambda: Path("./chromadb"))
    collection_name: str = "jarvis_knowledge"

    # Indexes
    bm25_index_file: Path = field(
        default_factory=lambda: Path("./chromadb/bm25_index.json")
    )
    index_cache_file: Path = field(
        default_factory=lambda: Path("./chromadb/index_cache.json")
    )
    use_lemmatization: bool = True

    # Indexing
    reindex_interval_days: int = 7
    batch_size: int = 100

    # Embedding
    embedding_batch_size: int = 8
    normalize_embeddings: bool = True

    # Query‑time
    top_k_bm25: int = 50
    top_k_semantic: int = 50
    top_k_final: int = 5
    rerank_threshold: float = 0.0
    exact_match_boost: float = 10.0

    # Performance
    enable_progress_bars: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("./models_cache"))

    def __post_init__(self):
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_index_file.parent.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Index Cache with transactions
# -------------------------------------------------------------------
class IndexCache:
    """
    Кеш индексации с поддержкой атомарных операций.
    """

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self):
        if not self.cache_file.exists():
            logger.info("Index cache not found, starting fresh")
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

            if not isinstance(self.cache, dict):
                logger.warning("⚠️  Invalid cache format, resetting")
                self.cache = {}
            logger.info(f"✅ Index cache loaded: {len(self.cache)} files")
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}, resetting")
            self.cache = {}

    def _save_cache(self):
        """Атомарно сохраняем кэш. Проверяем место перед записью."""
        # 1️⃣ Превью – сериализуем в строку, чтобы узнать размер
        data_str = json.dumps(
            self.cache,
            indent=2,
            ensure_ascii=False
        )
        required_bytes = len(data_str.encode("utf-8"))

        # 2️⃣ Проверяем свободный диск (путь к каталогу CROMA)
        check_disk_space(required_bytes, self.cache_file.parent)

        # 3️⃣ Теперь атомарная запись в tmp‑файл
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.cache_file.parent,
                    delete=False,
                    suffix=".tmp",
            ) as tmp_file:
                tmp_file.write(data_str)
                tmp_path = Path(tmp_file.name)

            # атомарное переименование
            tmp_path.replace(self.cache_file)
            logger.debug(
                f"✅ Index cache saved atomically: {len(self.cache)} files"
            )
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()

    def is_indexed(self, file_path: Path, current_hash: str) -> bool:
        key = str(file_path)
        return key in self.cache and self.cache[key].get("hash", "") == current_hash

    def mark_indexed(self, file_path: Path, file_hash: str, chunk_count: int):
        """Отметка файла – commit позже."""
        self.cache[str(file_path)] = {
            "hash": file_hash,
            "chunk_count": chunk_count,
            "indexed_at": datetime.now().isoformat(),
        }

    def commit(self):
        """Сохраняем все изменения atomically."""
        self._save_cache()

    def rollback(self):
        """Откат изменений – просто перезагружаем старую копию."""
        logger.warning("⚠️  Rolling back index cache changes")
        self._load_cache()

    def get_indexed_files(self) -> Set[str]:
        return set(self.cache.keys())

    def clear(self):
        self.cache = {}
        self._save_cache()


# -------------------------------------------------------------------
# BM25 Index with transactions
# -------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────
#  BM25‑ANN индекс (TF‑IDF + FAISS)
# ────────────────────────────────────────────────────────────────

import faiss
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BM25ANNIndex:
    """
    Хранилище с «BM25‑подобной» эвристикой, но реализованное как ANN.
    Использует TF‑IDF‑векторы и FAISS (inner‑product) для O(log n) поиска.
    """

    def __init__(self,
                 index_file: Path,
                 use_lemmatization: bool = True):
        self.index_file = index_file
        self.use_lemmatization = use_lemmatization

        # Векторизатор и FAISS‑индекс загружаются в память
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.corpus_ids: List[str] = []

        # Лемматизатор (если включён)
        self.lemmatizer = RussianLemmatizerFast() if use_lemmatization else None

        self._load_index()

    def _reset_index(self):
        logger.warning("🗑️  Сбрасываем повреждённый TF‑IDF/FAISS индекс")
        self.vectorizer = None
        self.faiss_index = None
        self.corpus_ids.clear()

    def _load_index(self):
        """Загружает сохранённые объекты (vectorizer + FAISS) из disk."""
        if not self.index_file.exists():
            logger.info("BM25‑ANN индекс не найден, будет построен при первой индексации")
            return

        try:
            data = joblib.load(str(self.index_file))
            self.vectorizer = data["vectorizer"]
            self.faiss_index = data["faiss_index"]
            self.corpus_ids = data.get("corpus_ids", [])
            logger.info(f"✅ Загрузили TF‑IDF/FAISS индекс: {len(self.corpus_ids)} документов")
        except Exception as e:
            logger.error(f"❌ Не удалось загрузить BF‑ANN индекс: {e}")
            self._reset_index()

    def _save_index(self):
        """Сохраняем vectorizer + FAISS. Проверяем место."""
        # Сначала сериализуем всё в память, чтобы узнать размер
        buf = io.BytesIO()
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "faiss_index": self.faiss_index,
                "corpus_ids": self.corpus_ids,
            },
            buf,
        )
        required_bytes = buf.tell()  # длина сериализованного объекта

        check_disk_space(required_bytes, self.index_file.parent)

        try:
            with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.index_file.parent,
                    delete=False,
                    suffix=".tmp",
            ) as tmp:
                tmp.write(buf.getvalue())
                tmp_path = Path(tmp.name)

            tmp_path.replace(self.index_file)
            logger.info("✅ TF‑IDF/FAISS индекс сохранён")
        except Exception as e:
            logger.error(f"❌ Не удалось сохранить индексацию: {e}")

    # ------------------------------------------------------------------
    def build_index(self, documents: List[Dict]):
        """
        Построение векторизатора и FAISS‑индекса из списка документов.
        С ПАРАЛЛЕЛЬНОЙ ЛЕММАТИЗАЦИЕЙ.
        """
        # 1️⃣ Собираем тексты и ids
        raw_texts = [doc["text"] for doc in documents] # Исходные тексты
        self.corpus_ids = [doc["id"] for doc in documents]

        # 2️⃣ Лемматизация (Pre-processing)
        raw_texts = [doc["text"] for doc in documents]

        logger.info("🔄 Подготовка текстов для BM25...")

        if self.use_lemmatization:
            logger.info(f"⚡ Запуск параллельной лемматизации ({len(raw_texts)} docs)...")

            # Ограничиваем количество процессов!
            # На Windows лучше не жадничать. 4-6 процессов достаточно.
            # Если поставить cpu_count(), память кончится.
            num_workers = min(6, cpu_count())

            try:
                # ВАЖНО: передаем initializer, чтобы создать MorphAnalyzer один раз на процесс
                with Pool(processes=num_workers, initializer=init_worker) as pool:
                    processed_texts = pool.map(lemmatize_text_worker, raw_texts)

            except Exception as e:
                logger.error(f"❌ Multiprocessing failed: {e}. Fallback to serial.")
                # Fallback: инициализируем локально и делаем в цикле
                init_worker()
                processed_texts = [lemmatize_text_worker(t) for t in raw_texts]
        else:
            processed_texts = raw_texts

        # 3️⃣ TF‑IDF векторизация (Уже на готовых строках)
        logger.info("🔄 TF‑IDF‑векторизация...")

        # ВАЖНО:
        # 1. tokenizer=None, preprocessor=None -> мы уже всё сделали сами
        # 2. token_pattern=r"(?u)\b\w+\b" -> стандартный, разбивает по пробелам (то, что нам надо)
        # ИЛИ token_pattern=None + tokenizer=lambda x: x.split()

        # Самый простой способ для пре-лемматизированного текста "word1 word2":
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w\w+\b", # Стандартный паттерн (слова от 2 букв)
            lowercase=True # На всякий случай, хотя лемматизатор уже low
        )

        tfidf_matrix: csr_matrix = self.vectorizer.fit_transform(processed_texts)

        # 4️⃣ Создаём FAISS‑индекс
        dim = tfidf_matrix.shape[1]
        logger.info(f"🔧 Конструируем FAISS индекс размерности {dim}")
        index_flat: faiss.Index = faiss.IndexFlatIP(dim)
        self.faiss_index = index_flat

        self.faiss_index.add(tfidf_matrix.toarray())
        logger.info(f"✅ FAISS готов – {len(self.corpus_ids)} векторов")

        # 5️⃣ Сохраняем результат
        self._save_index()

    # Вспомогательный метод внутри класса (или вне класса) для Pool.map
    def _lemmatize_text_helper(self, text):
        # Обертка, чтобы вызывать метод лемматизатора
        return self.lemmatizer.lemmatize(text)


    # ------------------------------------------------------------------
    def search(self,  query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Поиск по запросу с помощью FAISS и TF‑IDF.
        Возвращает список (doc_id, score), отсортированный по убыванию.
        """
        if self.faiss_index is None or not self.corpus_ids:
            logger.warning("FAISS индекс пуст – возвращаем []")
            return []

        # 1️⃣ Векторизуем запрос
        query_vec: np.ndarray = self.vectorizer.transform([query]).toarray()

        # 2️⃣ ANN поиск
        distances, indices = self.faiss_index.search(query_vec, top_k)
        results: List[Tuple[str, float]] = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS иногда возвращает -1
                continue
            doc_id = self.corpus_ids[idx]
            results.append((doc_id, float(dist)))

        return results

    # ------------------------------------------------------------------
    def clear(self):
        """Полностью удаляем индекс (в память и с диска)."""
        self._reset_index()
        if self.index_file.exists():
            try:
                self.index_file.unlink()
                logger.info("❌ Удалён TF‑IDF/FAISS файл")
            except Exception as e:
                logger.error(f"❌ Не удалось удалить файл индекса: {e}")


# -------------------------------------------------------------------
# Integrity Checker
# -------------------------------------------------------------------
class IntegrityChecker:
    """
    Проверка согласованности между ChromaDB, BM25 и IndexCache.
    Восстанавливает целостность после прерывания индексации.
    """

    @staticmethod
    def check_consistency(
            chroma_count: int,
            bm25_count: int,
            cache_files: int,
    ) -> Tuple[bool, str]:
        if chroma_count == 0 and bm25_count == 0 and cache_files == 0:
            return True, "Empty state (all zeros) - OK"

        max_discrepancy = 0.1  # 10%

        if bm25_count > 0 and chroma_count > 0:
            ratio = abs(bm25_count - chroma_count) / max(bm25_count, chroma_count)
            if ratio > max_discrepancy:
                return False, f"BM25/ChromaDB mismatch: {bm25_count} vs {chroma_count}"

        if cache_files > 0 and chroma_count == 0:
            return False, f"Cache has {cache_files} files but ChromaDB is empty"

        return True, "Consistency check passed"

    @staticmethod
    def suggest_recovery(
            chroma_count: int,
            bm25_count: int,
            cache_files: int,
    ) -> str:
        if chroma_count == 0 and (bm25_count > 0 or cache_files > 0):
            return "⚠️  Detected incomplete indexing. Run with force=True to rebuild."

        if bm25_count == 0 and chroma_count > 0:
            return "⚠️  BM25 index missing. It will be rebuilt automatically."

        if abs(bm25_count - chroma_count) > 100:
            return "⚠️  Significant mismatch. Consider force=True to rebuild."

        return "✅ No recovery needed"


# -------------------------------------------------------------------
# Hybrid RAG System
# -------------------------------------------------------------------
class HybridRAGSystem:
    """Hybrid RAG с защитой от прерываний."""

    def __init__(self, config: HybridConfig):
        self.config = config
        self._last_index_time: Optional[datetime] = None

        # Init components
        self.index_cache = IndexCache(config.index_cache_file)
        self.bm25_index = BM25ANNIndex(
            config.bm25_index_file, use_lemmatization=config.use_lemmatization
        )

        cuda_available, device_info = validate_gpu_availability()
        self.device = "cuda" if cuda_available else "cpu"
        logger.info(device_info)

        # Load models
        try:
            self.embedding_model = SentenceTransformer(
                config.embedding_model,
                cache_folder=str(config.cache_dir),
                device=self.device,
            )
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Embedding model failed: {e}")
            raise

        try:
            self.reranker = CrossEncoder(
                config.rerank_model, device=self.device
            )
            logger.info("✅ Reranker loaded")
        except Exception as e:
            logger.error(f"❌ Reranker failed: {e}")
            raise

        # ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(config.chromadb_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            chroma_count = self.collection.count()
            logger.info(f"✅ ChromaDB ready (docs={chroma_count})")
        except Exception as e:
            logger.error(f"❌ ChromaDB init failed: {e}")
            raise

        # Consistency check at startup
        self._check_integrity()

    def _check_integrity(self):
        """Проверка согласованности с поддержкой BM25ANNIndex"""
        chroma_count = self.collection.count()

        # ✅ ИСПРАВЛЕНИЕ: Универсальная проверка BM25 индекса
        if hasattr(self.bm25_index, 'corpus_ids'):
            bm25_count = len(self.bm25_index.corpus_ids)
        elif hasattr(self.bm25_index, 'corpus_texts'):
            bm25_count = len(self.bm25_index.corpus_texts)
        else:
            bm25_count = 0

        cache_files = len(self.index_cache.get_indexed_files())

        is_consistent, msg = IntegrityChecker.check_consistency(
            chroma_count, bm25_count, cache_files
        )
        if is_consistent:
            logger.info(f"✅ Integrity check: {msg}")
        else:
            logger.warning(f"⚠️  Integrity check FAILED: {msg}")
            recovery_msg = IntegrityChecker.suggest_recovery(
                chroma_count, bm25_count, cache_files
            )
            logger.warning(recovery_msg)

    # -------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------
    def index_static_documents(self, force: bool = False) -> int:
        """
        Индексация с защитой от прерываний.
        Все изменения коммитятся только в конце. При ошибке – откат.
        """
        if not force and self._last_index_time:
            elapsed = datetime.now() - self._last_index_time
            if elapsed < timedelta(days=self.config.reindex_interval_days):
                logger.info(
                    f"⏭️  Skipping reindex (last {elapsed.days}d ago)"
                )
                return 0

        if force:
            logger.info("🗑️  Force reindex: clearing...")
            try:
                self.chroma_client.delete_collection(self.config.collection_name)
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self.index_cache.clear()
                self.bm25_index.clear()
            except Exception as e:
                logger.error(f"Failed to clear: {e}")

        logger.info(f"📚 Indexing from {self.config.static_docs_dir}")

        documents: List[Dict] = []
        doc_id_start = self.collection.count()
        doc_id = doc_id_start

        stats = {
            "total_files": 0,
            "new_files": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0,
        }

        file_list = list(self.config.static_docs_dir.rglob("*"))
        file_list = [
            f
            for f in file_list
            if f.is_file() and f.suffix in {".md", ".txt", ".html", ".pdf"}
        ]

        stats["total_files"] = len(file_list)

        if not file_list:
            logger.warning("⚠️  No files found")
            return 0

        try:
            progress = tqdm(
                file_list,
                desc="Processing",
                disable=not self.config.enable_progress_bars,
            )

            for file_path in progress:
                try:
                    progress.set_postfix({"file": file_path.name[:30]})
                    file_hash = compute_file_hash(file_path)
                    if not file_hash:
                        stats["failed"] += 1
                        continue

                    if not force and self.index_cache.is_indexed(
                            file_path, file_hash
                    ):
                        stats["skipped"] += 1
                        continue

                    # extract content
                    if file_path.suffix == ".pdf":
                        content = self._extract_pdf_docling(file_path)
                    else:
                        content = self._extract_html_or_text(file_path)

                    if not content or len(content) < self.config.min_chunk_length:
                        stats["skipped"] += 1
                        continue

                    chunks = self._chunk_text(content, file_path.name)

                    if not chunks:
                        stats["skipped"] += 1
                        continue

                    for chunk in chunks:
                        documents.append(
                            {
                                "id": str(doc_id),
                                "text": chunk["text"],
                                "source": str(file_path.relative_to(self.config.static_docs_dir)),
                                "chunk_index": chunk["index"],
                                "file_hash": file_hash,
                                "metadata": {
                                    "source_type": "static",
                                    "indexed_at": datetime.now().isoformat(),
                                    "file_type": file_path.suffix,
                                    "file_hash": file_hash,
                                },
                            }
                        )
                        doc_id += 1

                    # mark in cache (but not commit yet)
                    self.index_cache.mark_indexed(file_path, file_hash, len(chunks))
                    stats["new_files"] += 1
                    stats["chunks"] += len(chunks)

                except Exception as e:
                    stats["failed"] += 1
                    logger.error(f"Error processing {file_path.name}: {e}")

            if not documents:
                logger.info("⚠️  No new documents")
                return 0

            # embeddings
            logger.info(f"🔄 Embedding {len(documents)} chunks...")

            all_texts = [doc["text"] for doc in documents]
            embeddings = self.embedding_model.encode(
                all_texts,
                batch_size=self.config.embedding_batch_size,
                show_progress_bar=self.config.enable_progress_bars,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
            )

            # store to chromadb
            logger.info("💾 Storing in ChromaDB...")
            self._add_documents_in_batches(documents, embeddings)

            # BM25 index
            logger.info("🔄 Building BM25 index...")
            self.bm25_index.build_index(documents)

            # commit all changes atomically
            logger.info("✅ Committing changes...")
            self.index_cache.commit()

            self._last_index_time = datetime.now()

            logger.info(f"✅ Indexing complete!")
            logger.info(
                f"   Files: {stats['new_files']} new, "
                f"{stats['skipped']} skipped, {stats['failed']} failed"
            )
            logger.info(f"   Chunks: {stats['chunks']} added")

            return len(documents)

        except KeyboardInterrupt:
            logger.warning("⚠️  Indexing interrupted by user!")
            logger.warning("🔄 Rolling back changes...")
            self.index_cache.rollback()
            raise

        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")
            logger.warning("🔄 Rolling back changes...")
            self.index_cache.rollback()
            raise

    # -------------------------------------------------------------------
    def _add_documents_in_batches(
            self,
            documents: List[Dict],
            embeddings,
            batch_size: Optional[int] = None,
    ):
        if batch_size is None:
            batch_size = self.config.batch_size

        total = len(documents)
        for i in range(0, total, batch_size):
            batch_docs = documents[i: i + batch_size]
            batch_embeddings = embeddings[i: i + batch_size]

            try:
                self.collection.add(
                    ids=[doc["id"] for doc in batch_docs],
                    embeddings=batch_embeddings.tolist(),
                    documents=[doc["text"] for doc in batch_docs],
                    metadatas=[
                        {
                            **doc["metadata"],
                            "source": doc["source"],
                            "chunk_index": doc["chunk_index"],
                        }
                        for doc in batch_docs
                    ],
                )
            except Exception as e:
                logger.error(f"Failed to add batch: {e}")
                raise

    # -------------------------------------------------------------------
    def _extract_pdf_docling(self, file_path: Path) -> Optional[str]:
        try:
            converter = DocumentConverter()
            doc = converter.convert(str(file_path))
            text_parts = [block.text for block in doc.blocks if hasattr(block, "text") and block.text]
            return "\n\n".join(text_parts) if text_parts else None
        except Exception:
            return None

    # -------------------------------------------------------------------
    def _extract_html_or_text(self, file_path: Path) -> Optional[str]:
        try:
            if file_path.suffix == ".html":
                html = file_path.read_text(encoding="utf-8", errors="ignore")
                return trafilatura.extract(html)
            else:
                return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    # -------------------------------------------------------------------
    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        chunks = []
        words = text.split()
        chunk_size_words = int(self.config.chunk_size / 1.3)
        overlap_words = int(self.config.chunk_overlap / 1.3)

        for i in range(0, len(words), chunk_size_words - overlap_words):
            chunk_words = words[i: i + chunk_size_words]
            chunk_text = " ".join(chunk_words)

            if len(chunk_text) >= self.config.min_chunk_length:
                chunks.append({"text": chunk_text, "index": len(chunks)})

        return chunks

    # -------------------------------------------------------------------
    # Search helpers
    # -------------------------------------------------------------------
    def _search_bm25(self, query: str) -> List[Dict]:
        logger.info("🔎 BM25‑ANN search…")
        try:
            hits = self.bm25_index.search(query, top_k=self.config.top_k_bm25)
        except Exception as e:
            logger.error(f"BM25‑ANN failed: {e}")
            return []

        if not hits:
            return []

        # получаем документы из Chroma по ids
        doc_ids = [hit[0] for hit in hits]
        bm25_scores = {hit[0]: hit[1] for hit in hits}

        results_dict = self.collection.get(ids=doc_ids)

        parsed = []
        for i, doc_id in enumerate(results_dict["ids"]):
            parsed.append(
                {
                    "id": doc_id,
                    "text": results_dict["documents"][i],
                    "metadata": results_dict["metadatas"][i],
                    "bm25_score": bm25_scores.get(doc_id, 0.0),
                    "search_type": "bm25",
                }
            )
        return parsed

    def _search_semantic(self, query_embedding) -> List[Dict]:
        logger.info("🔎 Semantic search…")
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.config.top_k_semantic,
            )
            parsed = []
            for i, doc_id in enumerate(results["ids"][0]):
                parsed.append(
                    {
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": (
                            results["distances"][0][i]
                            if "distances" in results
                            else None
                        ),
                        "search_type": "semantic",
                    }
                )
            return parsed
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
            self,
            bm25_results: List[Dict],
            semantic_results: List[Dict],
            k: int = 60
    ) -> List[Dict]:
        """RRF с правильной дедупликацией"""
        logger.info("🔄 Applying Reciprocal Rank Fusion...")

        rrf_scores: Dict[str, float] = {}
        documents: Dict[str, Dict] = {}

        # BM25 ранги
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

            # ✅ Сохраняем первое вхождение + помечаем источники
            if doc_id not in documents:
                documents[doc_id] = doc
                documents[doc_id]["sources"] = ["bm25"]
            else:
                documents[doc_id]["sources"].append("bm25")

        # Semantic ранги
        for rank, doc in enumerate(semantic_results, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

            if doc_id not in documents:
                documents[doc_id] = doc
                documents[doc_id]["sources"] = ["semantic"]
            else:
                documents[doc_id]["sources"].append("semantic")

        # Сортировка
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        combined = []
        top_n_candidates = 60  # Достаточно большое число для RTX 5070

        for doc_id in sorted_ids[:top_n_candidates]:
            # -------------------------------------------------------------------------
            doc = documents[doc_id]
            doc["rrf_score"] = rrf_scores[doc_id]
            doc["source_type"] = "static"

            # ✅ Приоритет типа: если в обоих, то "hybrid"
            if len(doc["sources"]) > 1:
                doc["search_type"] = "hybrid"
            else:
                doc["search_type"] = doc["sources"][0]

            combined.append(doc)

        logger.info(f"   RRF combined: {len(combined)} unique docs")

        return combined

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Перенаряжает результаты с помощью Cross‑Encoder и адаптивного бонуса
        за точное совпадение. При наличии exact‑match гарантирует его появление
        в топ‑k (если есть). В случае отказа ранжера возвращаем исходный порядок.
        """
        if not results:
            return []

        # ---------- 1️⃣ Подаём пары (query, doc) в Cross‑Encoder ----------
        pairs = [[query, res["text"][:512]] for res in results]
        try:
            raw_scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.error(f"Cross‑Encoder ранжирование упало: {e}")
            # fallback – оставляем порядок без изменения
            return sorted(
                results,
                key=lambda r: r.get("bm25_score", 0),
                reverse=True,
            )[: self.config.top_k_final]

        query_lower = query.lower()
        best_exact_score = None

        # ---------- 2️⃣ Применяем адаптивный бонус ----------
        for res, base in zip(results, raw_scores):
            base_score: float = float(base)
            has_exact = query_lower in res["text"].lower()

            if has_exact:
                qlen = len(query_lower)
                bonus = (
                    2.0 if qlen < 10
                    else 5.0 if qlen < 30
                    else 8.0
                )
                final_score = base_score + bonus
            else:
                final_score = base_score

            res.update(
                {
                    "rerank_score_base": base_score,
                    "has_exact_match": has_exact,
                    "rerank_score": final_score,
                }
            )

            if has_exact and (best_exact_score is None or final_score > best_exact_score):
                best_exact_score = final_score

        # ---------- 3️⃣ Сортировка по окончательному баллу ----------
        sorted_res = sorted(results, key=lambda r: r["rerank_score"], reverse=True)

        # ---------- 4️⃣ Threshold guard – гарантируем появление exact match ----------
        if best_exact_score is not None:
            idx_best = next(
                i for i, r in enumerate(sorted_res) if r["has_exact_match"]
            )
            if idx_best >= self.config.top_k_final:
                # поднимаем score чуть выше текущего k‑го
                kth_score = sorted_res[self.config.top_k_final - 1]["rerank_score"]
                sorted_res[idx_best]["rerank_score"] = kth_score + 0.05

        # ---------- 5️⃣ Финальная отборка ----------
        final_top = sorted(
            sorted_res,
            key=lambda r: r["rerank_score"],
            reverse=True,
        )[: self.config.top_k_final]

        return final_top

    async def search_web_fresh(self, query: str) -> List[Dict]:
        logger.info(f"🌐 Fresh web search: '{query}'")
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.config.searxng_url}/search",
                        params={"q": query, "format": "json"},
                        timeout=aiohttp.ClientTimeout(total=self.config.searxng_timeout),
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
        except Exception as e:
            logger.error(f"SearXNG error: {e}")
            return []

        raw_results = data.get("results", [])[:10]
        extracted_docs = []

        for result in raw_results:
            try:
                html = trafilatura.fetch_url(result["url"])
                if not html:
                    continue
                text = trafilatura.extract(html)
                if text and len(text) > 100:
                    extracted_docs.append(
                        {
                            "text": text[:5000],
                            "url": result["url"],
                            "title": result.get("title", ""),
                            "source_type": "fresh",
                        }
                    )
            except Exception:
                continue

        logger.info(f"✓ Extracted {len(extracted_docs)} fresh documents")
        return extracted_docs

    async def hybrid_search(self, query: str, use_fresh: bool = False) -> Dict:
        logger.info(f"🔍 Hybrid search: '{query[:50]}...'")

        async def bm25_task():
            return self._search_bm25(query)

        async def semantic_task():
            emb = self.embedding_model.encode(
                query,
                normalize_embeddings=self.config.normalize_embeddings,
            )
            return self._search_semantic(emb)

        tasks = [bm25_task(), semantic_task()]
        if use_fresh:
            tasks.append(self.search_web_fresh(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        bm25_res = results[0] if not isinstance(results[0], Exception) else []
        sem_res = results[1] if not isinstance(results[1], Exception) else []
        fresh_res = (
            results[2]
            if len(results) > 2 and not isinstance(results[2], Exception)
            else []
        )

        logger.info(f"   BM25: {len(bm25_res)}, Semantic: {len(sem_res)}, Fresh: {len(fresh_res)}")

        combined = self._reciprocal_rank_fusion(bm25_res, sem_res)

        for res in fresh_res:
            combined.append({**res, "source_type": "fresh"})

        logger.info(f"📊 Reranking {len(combined)} documents…")
        ranked = self._rerank_results(query, combined)

        return {
            "query": query,
            "results": ranked,
            "bm25_count": len(bm25_res),
            "semantic_count": len(sem_res),
            "fresh_count": len(fresh_res),
            "total": len(ranked),
            "timestamp": datetime.now().isoformat(),
        }

    # -------------------------------------------------------------------
    def get_collection_stats(self) -> Dict:
        return {
            "name": self.config.collection_name,
            "count": self.collection.count(),
            "last_indexed": (
                self._last_index_time.isoformat()
                if self._last_index_time
                else None
            ),
            "cached_files": len(self.index_cache.get_indexed_files()),
            "bm25_docs": len(self.bm25_index.corpus_ids),
        }

    def reset_collection(self):
        logger.warning("🗑️  Resetting collection")
        self.chroma_client.delete_collection(self.config.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.index_cache.clear()
        self.bm25_index.clear()
        self._last_index_time = None


# -------------------------------------------------------------------
# Demo entry point
# -------------------------------------------------------------------
async def main():
    config = HybridConfig(
        static_docs_dir=project_root() / "docs",
        embedding_model="intfloat/multilingual-e5-small",
        rerank_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        chromadb_dir=project_root() / "chromadb",
        enable_progress_bars=True,
    )

    logger.info("=" * 80)
    logger.info("HYBRID RAG SYSTEM v5.0 – BM25 + Лемматизация")
    logger.info("=" * 80)

    system = HybridRAGSystem(config)

    # Step 1: Indexing
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: INDEXING")
    logger.info("=" * 80)

    indexed = system.index_static_documents(force=False)
    if indexed:
        stats = system.get_collection_stats()
        logger.info(
            f"\n📊 Stats:\n   ChromaDB: {stats['count']} docs\n"
            f"   BM25: {stats['bm25_docs']} docs\n   Cached: {stats['cached_files']} files"
        )

    # Step 2: Demo
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: HYBRID SEARCH DEMO")
    logger.info("=" * 80)

    queries = [
        "Что будет через пять лет",
        "княжна Марья мечтала, как мечтают всегда девушки",
    ]
    for q in queries:
        logger.info(f"\n{'=' * 60}\n🔍 Query: '{q}'\n{'=' * 60}")
        res = await system.hybrid_search(q, use_fresh=False)
        logger.info(
            f"\n📊 Results: {res['total']} documents\n"
            f"   BM25:{res['bm25_count']}\n   Semantic:{res['semantic_count']}\n   Fresh:{res['fresh_count']}"
        )
        # В функции main()
        # В main():
        for i, doc in enumerate(res['results'][:5], 1):
            base_score = doc.get('rerank_score_base', 0)
            final_score = doc.get('rerank_score', 0)
            rrf_score = doc.get('rrf_score', 0)
            has_exact = doc.get('has_exact_match', False)
            search_type = doc.get('search_type', 'unknown')

            # Маркеры
            exact_marker = " 🎯 EXACT" if has_exact else ""
            hybrid_marker = " 🔀 HYBRID" if search_type == "hybrid" else ""

            logger.info(f"\n{i}. [Rerank {final_score:.3f}] [RRF {rrf_score:.4f}]{exact_marker}{hybrid_marker}")

            if has_exact and base_score != final_score:
                bonus = final_score - base_score
                logger.info(f"   (Base: {base_score:.3f} → Boosted: +{bonus:.1f})")

            logger.info(f"   Source: {doc.get('metadata', {}).get('source', 'unknown')}")
            logger.info(f"   Type: {search_type}")
            logger.info(f"   Text: {doc['text'][:150]}...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
