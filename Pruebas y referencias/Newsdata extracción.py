#!/usr/bin/env python3
import os
import json
import time
import random
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError

# =========================
# CONFIG (EDITA AQUÍ)
# =========================
API_KEY = "pub_ae938e782309499483762d7b875da563"
OUT_DIR = os.path.join("1 Data", "raw")
MAX_REQUESTS = 200 # Ajustado al límite diario del plan gratuito.
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN = 0.7
HOURLY_REQUEST_LIMIT = 30 # Límite de peticiones por cada 15 minutos.

BASE_URL = "https://newsdata.io/api/1/news"
#BASE_URL = "https://newsdata.io/api/1/news"

BASE_PARAMS = {
    "apikey": API_KEY,
    "q": "news",       # palabra clave que te interese
    "language": "en",
    "size": "10"
    #"from_date": "2025-10-24"  # solo noticias recientes
}

"""BASE_PARAMS = {
    "apikey": API_KEY,
    "full_content": "1",
    "size": "10", 
    "q": "news" 
}"""

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

def build_url(params: dict):
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"

def fetch(url: str, timeout=REQUEST_TIMEOUT, retries=3, base_sleep=0.8):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (NewsCollector/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read().decode("utf-8", errors="ignore")
                return json.loads(data)
        except HTTPError as e:
            if e.code == 429:
                print(f"⚠️ Demasiadas peticiones (HTTP 429). Esperando 15 minutos antes de reintentar...")
                time.sleep(900)  # Espera 15 minutos (15 * 60 segundos)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.3))
        except (URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.3))
    return None

def run_harvest(max_requests=MAX_REQUESTS):
    """
    Recolecta noticias recientes desde NewsData.io usando /latest, respetando los límites del plan gratuito.
    """

    seen_ids, seen_links = set(), set()
    total_reqs = total_articles = total_unique = 0

    next_page = None
    page_idx = 1
    empty_streak = 0
    MAX_EMPTY_PAGES = 2
    
    start_time = datetime.now()

    while total_reqs < max_requests:
        # Control del límite de 30 peticiones cada 15 minutos
        if total_reqs > 0 and total_reqs % HOURLY_REQUEST_LIMIT == 0:
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < 900:  # 15 minutos
                wait_time = 900 - elapsed_time.total_seconds()
                print(f"⚠️ Límite horario de 30 peticiones alcanzado. Esperando {wait_time:.2f} segundos...")
                time.sleep(wait_time)
            start_time = datetime.now() # Reinicia el temporizador
        
        params = {**BASE_PARAMS}
        if next_page:
            params["page"] = next_page

        url = build_url(params)
        time.sleep(SLEEP_BETWEEN)

        try:
            payload = fetch(url)
        except Exception as e:
            save_raw({"error": str(e), "url": url}, f"newsdata_error__page-{page_idx}")
            total_reqs += 1
            break

        if not payload:
            print("❌ No se recibieron datos, la API pudo haber fallado. Terminando la recolección.")
            break

        save_raw(payload, f"newsdata_latest__page-{page_idx}")
        total_reqs += 1

        results = payload.get("results") or []
        total_articles += len(results)

        new_here = 0
        for a in results:
            aid = a.get("article_id")
            link = a.get("link")
            key = aid or link
            if not key:
                continue
            if (aid and aid in seen_ids) or (link and link in seen_links):
                continue
            if aid: seen_ids.add(aid)
            if link: seen_links.add(link)
            new_here += 1

        total_unique += new_here

        print(f"✅ Página {page_idx} extraída. Artículos nuevos en esta página: {new_here}. Total único: {total_unique}. Total de peticiones: {total_reqs}")

        empty_streak = empty_streak + 1 if new_here == 0 else 0
        next_page = payload.get("nextPage")

        if not next_page or empty_streak >= MAX_EMPTY_PAGES:
            print("ℹ️ No hay más páginas o el flujo de artículos nuevos se ha detenido. Finalizando la recolección.")
            break

        page_idx += 1

    summary = {
        "requests_made": total_reqs,
        "total_articles_returned": total_articles,
        "unique_articles_added": total_unique,
        "raw_dir": OUT_DIR
    }
    save_raw(summary, "summary_newsdata_latest")
    print("\n--- RESUMEN DE LA RECOLECCIÓN ---")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_harvest()

