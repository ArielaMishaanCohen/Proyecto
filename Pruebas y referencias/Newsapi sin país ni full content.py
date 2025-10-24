#!/usr/bin/env python3
import os
import json
import time
import random
from datetime import datetime
from newsapi import NewsApiClient
from newspaper import Article

# =========================
# CONFIG
# =========================
API_KEY = "51a2fc81ffe543709ef96c0e284e7f39"
OUT_DIR = os.path.join("1 Data", "raw")
MAX_REQUESTS = 200  # total máximo de requests
SLEEP_BETWEEN = 0.7
HOURLY_REQUEST_LIMIT = 30  # límite de requests por cada 15 minutos
PAGE_SIZE = 20  # máximo que permite NewsAPI por página

# Inicializa NewsAPI
newsapi = NewsApiClient(api_key=API_KEY)

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_raw(payload: dict, meta_slug: str):
    ensure_outdir()
    filename = f"{ts()}__{meta_slug}.json"
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def fetch_full_content(url):
    """Extrae el contenido completo de una noticia usando newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"ERROR extracting content: {str(e)}"

def run_harvest(max_requests=MAX_REQUESTS):
    seen_links = set()
    total_reqs = total_articles = total_unique = 0
    page = 1
    empty_streak = 0
    MAX_EMPTY_PAGES = 2
    start_time = datetime.now()

    while total_reqs < max_requests:
        # Control de límite horario (30 requests cada 15 minutos)
        if total_reqs > 0 and total_reqs % HOURLY_REQUEST_LIMIT == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed < 900:
                wait_time = 900 - elapsed
                print(f"⚠️ Límite horario alcanzado. Esperando {wait_time:.2f}s...")
                time.sleep(wait_time)
            start_time = datetime.now()

        try:
            response = newsapi.get_everything(
                q="news",
                #language="en",
                sort_by="publishedAt",
                page=page,
                page_size=PAGE_SIZE
            )
        except Exception as e:
            save_raw({"error": str(e), "page": page}, f"newsapi_error__page-{page}")
            total_reqs += 1
            break

        total_reqs += 1
        articles = response.get("articles", [])
        total_articles += len(articles)
        new_here = 0

        for a in articles:
            url = a.get("url")
            if not url or url in seen_links:
                continue
            seen_links.add(url)

            # Extraemos contenido completo
            full_content = fetch_full_content(url)
            a["full_content"] = full_content

            new_here += 1

        total_unique += new_here
        save_raw(response, f"newsapi_everything__page-{page}")

        print(f"✅ Página {page} extraída. Nuevos: {new_here}, Total único: {total_unique}, Requests: {total_reqs}")

        empty_streak = empty_streak + 1 if new_here == 0 else 0
        if empty_streak >= MAX_EMPTY_PAGES or len(articles) == 0:
            print("ℹ️ No hay más páginas o artículos nuevos se han agotado. Finalizando la recolección.")
            break

        page += 1
        time.sleep(SLEEP_BETWEEN)

    summary = {
        "requests_made": total_reqs,
        "total_articles_returned": total_articles,
        "unique_articles_added": total_unique,
        "raw_dir": OUT_DIR
    }
    save_raw(summary, "summary_newsapi_everything")
    print("\n--- RESUMEN DE LA RECOLECCIÓN ---")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_harvest()