#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf_to_markdown.py

CLI‑утилита для конвертации PDF‑документов в Markdown.
Поддержка чтения входных/выходных путей из ``.env`` файла
(для удобства тестирования и CI‑пайпов).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# ----------------------------------------------------------------------
# Попытка загрузить переменные окружения из .env (необязательно)
# ----------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()   # ищет .env в текущей рабочей директории и выше
except Exception:  # pragma: no cover
    # Если ``python‑dotenv`` не установлен – просто игнорируем.
    pass


# ----------------------------------------------------------------------
# Зависимости Docling (>=1.20.0)
# ----------------------------------------------------------------------
try:
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,          # формат‑опция для PDF
        InputFormat,             # перечисление форматов ввода
    )
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        f"Не удалось импортировать Docling API. Убедитесь, что "
        f"docling>=1.20.0 установлен.\n{exc}"
    ) from exc


# ----------------------------------------------------------------------
# Опционально – PyTorch для CUDA‑ускорения
# ----------------------------------------------------------------------
try:
    import torch  # type: ignore
except Exception:
    torch = None   # Путь без GPU будет использоваться

LOGGER = logging.getLogger("pdf_to_markdown")


def detect_cuda_available() -> bool:
    """
    Проверяет, доступна ли CUDA для PyTorch.

    Возвращает:
        bool: ``True`` если модуль ``torch`` установлен и
              ``torch.cuda.is_available() == True``.
    """
    if torch is None:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        # На некоторых платформах может быть отсутствует атрибут .cuda
        return False


def build_converter(use_gpu: bool) -> DocumentConverter:
    """
    Создаёт конвертер Docling с оптимальными настройками ускорения.

    В Docling 1.20+ конструктор принимает *словарь* `format_options`,
    где ключ – это `InputFormat`, а значение – объект `FormatOption`.
    Для PDF мы создаём `PdfFormatOption` и передаём туда наш
    ``ThreadedPdfPipelineOptions`` (batch‑size, ускоритель и т.п.).
    """
    if use_gpu and detect_cuda_available():
        device = AcceleratorDevice.CUDA
        LOGGER.info("CUDA обнаружена, использование GPU.")
    else:
        device = AcceleratorDevice.CPU
        if use_gpu:
            LOGGER.warning(
                "Флаг --gpu включён, но CUDA недоступна. Переходим на CPU."
            )
        else:
            LOGGER.debug("Запуск в режиме CPU.")

    accelerator_options = AcceleratorOptions(device=device)

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=accelerator_options,
        layout_batch_size=16 if device == AcceleratorDevice.CUDA else 8,
        ocr_batch_size=16 if device == AcceleratorDevice.CUDA else 4,
        table_batch_size=4,
    )

    # ------------------------------------------------------------------
    # Формируем словарь format_options для PDF
    # ------------------------------------------------------------------
    pdf_format_option = PdfFormatOption(
        pipeline_options=pipeline_options,
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: pdf_format_option}
    )
    return converter


def convert_pdf_to_markdown(
        input_path: Path,
        output_path: Optional[Path] = None,
        use_gpu: bool = False,
        strict_text: bool = False,
        overwrite: bool = False,
) -> Path:
    """Конвертирует PDF‑файл в Markdown."""
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")
    if input_path.is_dir():
        raise IsADirectoryError(
            f"Ожидался файл, но получена директория: {input_path}"
        )

    output_path = Path(output_path or input_path.with_suffix(".md"))

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Файл {output_path} уже существует. "
            "Укажите --overwrite для перезаписи."
        )

    converter = build_converter(use_gpu=use_gpu)

    LOGGER.info("Начинаем конвертацию: %s → %s", input_path, output_path)
    try:
        conv_result = converter.convert(str(input_path))
        doc = conv_result.document
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Ошибка Docling при конвертации %s", input_path)
        raise RuntimeError(f"Docling‑превращение завершилось ошибкой: {exc}") from exc

    markdown_text = doc.export_to_markdown(strict_text=strict_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(markdown_text, encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        LOGGER.exception("Не удалось записать Markdown в %s", output_path)
        raise RuntimeError(f"Ошибка записи файла {output_path}: {exc}") from exc

    LOGGER.info("Конвертация завершена, файл сохранён в %s", output_path)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Разбирает аргументы командной строки. Путь входного PDF – обязательный.
    """
    parser = argparse.ArgumentParser(
        description="Преобразование PDF в Markdown с помощью Docling."
    )
    parser.add_argument("pdf", type=Path, help="Путь к входному PDF‑файлу.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Путь к выходному .md‑файлу (по умолчанию: тот же путь с "
            "расширением .md)."
        ),
    )
    parser.add_argument("--gpu", action="store_true", help="Использовать CUDA, если доступна.")
    parser.add_argument(
        "--strict-text",
        action="store_true",
        help="Экспортировать только текст (полезен для RAG).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать существующий файл вывода.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Вывод DEBUG‑лога.")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    """Настраивает базовое логирование."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Точка входа CLI. Возвращает код возврата: 0 – успех, 1 – ошибка.
    Поддержка чтения пути PDF/Markdown из переменных окружения
    ``PDF_INPUT`` и ``PDF_OUTPUT`` (необязательно).
    """

    # --------------------------------------------------------------
    # Конструируем argv с учётом .env‑переменных
    # --------------------------------------------------------------
    argv_list: list[str]
    if argv is None:
        argv_list = sys.argv[1:]
    else:
        argv_list = list(argv)  # копия, чтобы не мутировать переданное значение

    pdf_env = os.getenv("PDF_INPUT")
    output_env = os.getenv("PDF_OUTPUT")

    # Если позиционный аргумент отсутствует – берём из env
    if not any(not arg.startswith("-") for arg in argv_list) and pdf_env:
        argv_list.insert(0, pdf_env)

    # Добавляем флаг вывода, если он не задан и присутствует в env
    if (
            "-o" not in argv_list
            and "--output" not in argv_list
            and output_env
    ):
        argv_list.extend(["-o", output_env])

    args = parse_args(argv_list)
    configure_logging(args.verbose)

    try:
        convert_pdf_to_markdown(
            input_path=args.pdf,
            output_path=args.output,
            use_gpu=args.gpu,
            strict_text=args.strict_text,
            overwrite=args.overwrite,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Произошла ошибка: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
