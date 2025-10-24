#!/usr/bin/env python3
import os
import json
import time
import random
import urllib.parse
import urllib.request
from datetime import datetime
from urllib.error import HTTPError, URLError

# =========================
# TUS CONFIGURACIONES
# =========================
apikey = "28a29e00c078647cd55c0a929afd3416"
OUT_DIR = os.path.join("data", "raw")
MAX_REQUESTS = 100
category = "general"
REQUEST_TIMEOUT = 15  # segundos

BASE = "https://gnews.io/api/v4/top-headlines"
COMMON_PARAMS = {
    "category": category,   # nos quedamos en "general"
    "lang": "any",
    "country": "any",
    "max": "10",
    "apikey": apikey,
    # *opcional*: si quieres forzar orden por fecha, descomenta:
    # "sortby": "publishedAt",
}

# =========================
# HELPERS
# =========================
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

def fetch(url: str, timeout: int = REQUEST_TIMEOUT, retries: int = 3, base_sleep: float = 0.8):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read().decode("utf-8")
                return json.loads(data)
        except (HTTPError, URLError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.25))
    return None

def build_url(params: dict):
    return f"{BASE}?{urllib.parse.urlencode(params)}"

# =========================
# MAIN
# =========================
def run():
    seen_urls = set()
    total_reqs = 0
    total_articles = 0
    total_new = 0
    total_empty = 0

    # Recorremos páginas 1..N hasta MAX_REQUESTS
    # Nota: si el API deja de tener resultados, romperemos el loop.
    page = 1
    while total_reqs < MAX_REQUESTS:
        params = {**COMMON_PARAMS, "page": str(page)}
        url = build_url(params)

        # Delay corto entre requests (amable con el API free)
        time.sleep(0.6)

        try:
            payload = fetch(url)
        except Exception as e:
            save_raw({"error": str(e), "url": url}, "top-headlines__error")
            total_reqs += 1
            page += 1
            continue

        # Guardar SIEMPRE la respuesta cruda
        save_raw(payload, f"top-headlines__category-{category}__page-{page}")

        total_reqs += 1

        articles = payload.get("articles") or []
        total_articles += len(articles)

        # Si no hay artículos, contabiliza vacío y prueba siguiente página;
        # si encadenamos varios vacíos, rompemos.
        if not articles:
            total_empty += 1
            # si llevamos 3 vacíos seguidos, es muy probable que no haya más páginas útiles
            if total_empty >= 3:
                break
            page += 1
            continue

        # Dedup por URL
        new_here = 0
        for a in articles:
            u = a.get("url")
            if not u:
                continue
            if u in seen_urls:
                continue
            seen_urls.add(u)
            new_here += 1

        total_new += new_here

        # Heurística: si una página no trajo nada nuevo, aumenta un contador de páginas "sin nuevos"
        # y si ya van varias seguidas, rompemos. (Evita gastar requests en páginas repetidas.)
        if new_here == 0:
            total_empty += 1
            if total_empty >= 3:
                break
        else:
            total_empty = 0  # resetea si hubo algo nuevo

        page += 1

    summary = {
        "requests_made": total_reqs,
        "total_articles_returned": total_articles,
        "unique_articles_added": len(seen_urls),
        "raw_dir": OUT_DIR,
        "last_page_attempted": page - 1,
    }
    save_raw(summary, "summary_top_headlines_general")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run()