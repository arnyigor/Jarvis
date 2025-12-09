#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm   # опционально, чтобы видеть прогресс

BASE_URL = "https://export.arxiv.org/api/query"
PARAMS = {
    "search_query": "cat:cs.AI",
    "start": 0,
    "max_results": 100,
    "sortBy": "submittedDate",
}
PDF_DIR = Path(__file__).parent / "arxiv_pdfs"   # папка рядом со скриптом
MAX_PDFS = 10

# --- Создаём директорию, если её нет -----------------------
PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- Настраиваем сессию --------------------------------------------------
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount("https://", adapter)          # повторить запрос при падении
session.headers.update({
    "User-Agent": f"arxiv-downloader/1.0 (+https://github.com/<your-username>)"
})

# --- Получаем XML‑фид ----------------------------------------------------
resp = session.get(BASE_URL, params=PARAMS, timeout=15)
resp.raise_for_status()
feed = ET.fromstring(resp.content)

# --- Итерируем по <entry> и сохраняем первые 10 PDF ---------------------
ns = {"atom": "http://www.w3.org/2005/Atom"}
for i, entry in enumerate(feed.findall("atom:entry", ns)):
    if i >= MAX_PDFS:
        break
    arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    out_path = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"  # replace для Windows
    try:
        r = session.get(pdf_url, stream=True, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc=f"Downloading {arxiv_id}", unit="B", unit_scale=True):
                if chunk:
                    fh.write(chunk)
    except requests.RequestException as exc:
        print(f"[WARN] Не удалось скачать {pdf_url}: {exc}")
