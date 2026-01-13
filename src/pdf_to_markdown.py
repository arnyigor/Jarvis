#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf_to_markdown_v3.py
Оптимизированная версия для Docling v2 с улучшенным управлением памятью и VLM.
"""

import argparse
import logging
import os
import gc
from pathlib import Path
from typing import Optional

# --- Загрузка окружения ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Импорты Docling и Torch ---
try:
    import torch
except ImportError:
    torch = None

try:
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
        InputFormat,
    )
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        AcceleratorOptions,
        AcceleratorDevice
    )
    from docling.datamodel.base_models import InputFormat
    from docling_core.types.doc import ImageRefMode, ContentLayer
except ImportError as exc:
    raise RuntimeError(
        "Docling не установлен или версия несовместима. "
        "Установите: pip install docling[torch]"
    ) from exc

LOGGER = logging.getLogger("docling_worker")


def get_torch_device(force_cpu: bool = False) -> str:
    """Определяет доступное устройство."""
    if force_cpu or torch is None or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def cleanup_gpu():
    """Принудительная очистка VRAM."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def build_converter(
        device_str: str,
        artifacts_path: Optional[Path] = None,
        enable_vlm: bool = False,
        batch_size_mult: float = 1.0
) -> DocumentConverter:
    """
    Создает конвертер с явной типизацией параметров.
    batch_size_mult: множитель для уменьшения/увеличения дефолтных батчей.
    """

    # 1. Настройка ускорителя
    if device_str == "cuda":
        acc_device = AcceleratorDevice.CUDA
        # Базовые значения для GPU (можно тюнить множителем)
        num_threads = 4
        pipeline_batch_size = int(16 * batch_size_mult)
        ocr_batch_size = int(32 * batch_size_mult)
    else:
        acc_device = AcceleratorDevice.CPU
        num_threads = os.cpu_count() or 8
        pipeline_batch_size = 4
        ocr_batch_size = 4

    acc_options = AcceleratorOptions(
        num_threads=num_threads,
        device=acc_device
    )

    # 2. Настройка пайплайна
    # В Docling v2 параметры часто группируются
    pipeline_opts = PdfPipelineOptions(
        accelerator_options=acc_options,
        do_ocr=True,
        do_table_structure=True,
        do_formula_enrichment=True,
        # VLM (описание картинок) - самая ресурсоемкая операция
        do_picture_description=enable_vlm,
    )

    # Применяем размеры батчей (если атрибуты существуют в текущей версии Docling)
    # Используем явное присваивание, ожидая, что API стабилен для v2+
    if hasattr(pipeline_opts, "layout_batch_size"):
        pipeline_opts.layout_batch_size = pipeline_batch_size

    if hasattr(pipeline_opts, "ocr_options") and hasattr(pipeline_opts.ocr_options, "batch_size"):
        # В некоторых версиях OCR options вложены
        pass
    elif hasattr(pipeline_opts, "ocr_batch_size"):
        pipeline_opts.ocr_batch_size = ocr_batch_size

    # Настройка таблиц
    if hasattr(pipeline_opts, "table_structure_options"):
        pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE

    # Указание пути к моделям (если нужно офлайн использование)
    if artifacts_path:
        pipeline_opts.artifacts_path = str(artifacts_path)

    # 3. Сборка
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )


def convert_pdf(
        input_path: Path,
        output_path: Optional[Path] = None,
        use_gpu: bool = False,
        describe_images: bool = False,
        overwrite: bool = False,
        batch_mult: float = 1.0
) -> Path:

    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".md")

    if output_path.exists() and not overwrite:
        LOGGER.warning(f"Файл {output_path.name} существует. Пропуск.")
        return output_path

    # Определение устройства
    device = get_torch_device(force_cpu=not use_gpu)
    LOGGER.info(f"🔧 Device: {device.upper()} | VLM: {describe_images} | Batch x{batch_mult}")

    try:
        converter = build_converter(
            device_str=device,
            enable_vlm=describe_images,
            batch_size_mult=batch_mult
        )

        LOGGER.info(f"🚀 Processing: {input_path.name}")

        # Конвертация
        res = converter.convert(str(input_path))
        doc = res.document

        # Экспорт
        # image_mode=MARKDOWN создает ссылки, но для VLM важно, чтобы описание попало в текст.
        # В Docling описание картинки обычно добавляется в контент-блок Picture.
        md_text = doc.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED,
            image_placeholder="*[Image Description]*",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_text, encoding="utf-8")

        LOGGER.info(f"✅ Saved to: {output_path}")
        return output_path

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            LOGGER.error("❌ GPU OOM Error. Попробуйте уменьшить --batch-mult (например 0.5)")
        raise e
    except Exception as e:
        LOGGER.exception(f"❌ Critical error processing {input_path.name}")
        raise
    finally:
        # Важно очищать ресурсы, особенно если скрипт будет встроен в цикл
        cleanup_gpu()


def main():
    parser = argparse.ArgumentParser(description="Docling PDF to Markdown Converter")

    # Основные аргументы
    parser.add_argument("pdf", type=Path, nargs="?", help="Input PDF path")
    parser.add_argument("-o", "--output", type=Path, help="Output MD path")

    # Флаги
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acc")
    parser.add_argument("--vlm", action="store_true", help="Enable Vision Language Model for images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # Тюнинг
    parser.add_argument("--batch-mult", type=float, default=1.0,
                        help="Batch size multiplier (0.5 for low VRAM, 2.0 for A100)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logs")

    # Defaults from ENV
    parser.set_defaults(
        gpu=os.getenv("PDF_TO_MD_USE_GPU", "false").lower() in ("true", "1", "yes"),
        vlm=os.getenv("PDF_TO_MD_VLM", "false").lower() in ("true", "1", "yes"),
        overwrite=os.getenv("PDF_TO_MD_OVERWRITE", "false").lower() in ("true", "1", "yes"),
    )

    args = parser.parse_args()

    # Fallback для Input
    if not args.pdf:
        env_path = os.getenv("PDF_PATH")
        if env_path:
            args.pdf = Path(env_path).resolve()
        else:
            parser.error("Не указан входной файл (аргумент или PDF_PATH в .env)")

    # Логгирование
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] Docling: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Запуск
    convert_pdf(
        args.pdf,
        args.output,
        use_gpu=args.gpu,
        describe_images=args.vlm,
        overwrite=args.overwrite,
        batch_mult=args.batch_mult
    )

if __name__ == "__main__":
    main()
