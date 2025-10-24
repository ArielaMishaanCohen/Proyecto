#!/usr/bin/env python3
import os
import json
import time
import pickle
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from newspaper import Article

# =========================
# CONFIG
# =========================
API_KEY = "51a2fc81ffe543709ef96c0e284e7f39"
OUT_DIR = os.path.join("1_Data", "raw")
SEEN_URLS_FILE = "seen_urls.pkl"
MAX_REQUESTS = 200  # total máximo de requests
SLEEP_BETWEEN = 0.7
HOURLY_REQUEST_LIMIT = 30  # límite de requests por cada 15 minutos
PAGE_SIZE = 20  # máximo que permite NewsAPI por página
DAYS_BACK = 30  # Cuántos días hacia atrás buscar noticias

# Inicializa NewsAPI
newsapi = NewsApiClient(api_key=API_KEY)

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def load_seen_urls():
    """Carga URLs ya vistas de ejecuciones anteriores."""
    if os.path.exists(SEEN_URLS_FILE):
        try:
            with open(SEEN_URLS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            print("⚠️ No se pudo cargar el archivo de URLs vistas. Creando uno nuevo.")
            return set()
    return set()

def save_seen_urls(seen_urls):
    """Guarda URLs vistas para futuras ejecuciones."""
    try:
        with open(SEEN_URLS_FILE, 'wb') as f:
            pickle.dump(seen_urls, f)
        print(f"✅ URLs guardadas en {SEEN_URLS_FILE}")
    except Exception as e:
        print(f"❌ Error guardando URLs: {e}")

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

def get_country_from_source(article):
    """Intenta inferir el país desde la fuente o URL."""
    url = article.get("url", "")
    
    # Mapeo de dominios a países
    domain_to_country = {
        ".gr": "Greece",
        "skai.gr": "Greece",
        ".uk": "United Kingdom",
        "bbc.co.uk": "United Kingdom",
        "bbc.com": "United Kingdom",
        "theguardian.com": "United Kingdom",
        "cnn.com": "United States",
        "nytimes.com": "United States",
        "washingtonpost.com": "United States",
        "foxnews.com": "United States",
        "nbcnews.com": "United States",
        "abcnews.go.com": "United States",
        "usatoday.com": "United States",
        "wsj.com": "United States",
        ".es": "Spain",
        "elpais.com": "Spain",
        "elmundo.es": "Spain",
        ".fr": "France",
        "lemonde.fr": "France",
        "lefigaro.fr": "France",
        ".de": "Germany",
        "spiegel.de": "Germany",
        "bild.de": "Germany",
        ".it": "Italy",
        "corriere.it": "Italy",
        "repubblica.it": "Italy",
        ".cn": "China",
        ".jp": "Japan",
        ".au": "Australia",
        "abc.net.au": "Australia",
        ".ca": "Canada",
        "cbc.ca": "Canada",
        ".in": "India",
        "timesofindia.com": "India",
        ".br": "Brazil",
        "globo.com": "Brazil",
        ".mx": "Mexico",
        ".ar": "Argentina",
        "clarin.com": "Argentina",
        ".ru": "Russia",
        ".nl": "Netherlands",
        ".se": "Sweden",
        ".no": "Norway",
        ".dk": "Denmark",
        ".fi": "Finland",
        ".pt": "Portugal",
        ".tr": "Turkey",
        ".sa": "Saudi Arabia",
        ".ae": "United Arab Emirates",
        ".il": "Israel",
        ".za": "South Africa",
        ".nz": "New Zealand",
        ".kr": "South Korea",
        ".th": "Thailand",
        ".sg": "Singapore",
        ".my": "Malaysia",
        ".id": "Indonesia",
        ".ph": "Philippines",
        ".vn": "Vietnam",
        ".pl": "Poland",
        ".cz": "Czech Republic",
        ".ro": "Romania",
        ".hu": "Hungary",
        ".at": "Austria",
        ".ch": "Switzerland",
        ".be": "Belgium",
        ".ie": "Ireland",
    }
    
    for domain, country in domain_to_country.items():
        if domain in url.lower():
            return country
    
    return "Unknown"

def run_harvest(max_requests=MAX_REQUESTS, days_back=DAYS_BACK):
    seen_links = load_seen_urls()
    initial_seen_count = len(seen_links)
    
    total_reqs = total_articles = total_unique = 0
    page = 1
    empty_streak = 0
    MAX_EMPTY_PAGES = 2
    start_time = datetime.now()
    
    # Calcular rango de fechas
    fecha_hasta = datetime.now().strftime('%Y-%m-%d')
    fecha_desde = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO RECOLECCIÓN DE NOTICIAS")
    print(f"{'='*60}")
    print(f"📅 Período: {fecha_desde} hasta {fecha_hasta}")
    print(f"🔗 URLs ya vistas previamente: {initial_seen_count}")
    print(f"🎯 Máximo de requests: {max_requests}")
    print(f"{'='*60}\n")

    while total_reqs < max_requests:
        # Control de límite horario (30 requests cada 15 minutos)
        if total_reqs > 0 and total_reqs % HOURLY_REQUEST_LIMIT == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed < 900:
                wait_time = 900 - elapsed
                print(f"⚠️ Límite horario alcanzado. Esperando {wait_time:.0f}s...")
                time.sleep(wait_time)
            start_time = datetime.now()

        try:
            response = newsapi.get_everything(
                q="news",
                from_param=fecha_desde,
                to=fecha_hasta,
                sort_by="publishedAt",
                page=page,
                page_size=PAGE_SIZE, 
                language = "en"
            )
        except Exception as e:
            print(f"❌ Error en request: {str(e)}")
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
            
            # Inferimos el país
            a["inferred_country"] = get_country_from_source(a)

            new_here += 1

        total_unique += new_here
        save_raw(response, f"newsapi_everything__page-{page}")

        print(f"✅ Página {page} | Nuevos: {new_here} | Total único: {total_unique} | Requests: {total_reqs}/{max_requests}")

        empty_streak = empty_streak + 1 if new_here == 0 else 0
        if empty_streak >= MAX_EMPTY_PAGES or len(articles) == 0:
            print("ℹ️ No hay más artículos nuevos. Finalizando...")
            break

        page += 1
        time.sleep(SLEEP_BETWEEN)

    # Guardar URLs vistas
    save_seen_urls(seen_links)
    
    summary = {
        "requests_made": total_reqs,
        "total_articles_returned": total_articles,
        "unique_articles_added": total_unique,
        "previous_urls_count": initial_seen_count,
        "total_urls_now": len(seen_links),
        "date_range": f"{fecha_desde} to {fecha_hasta}",
        "raw_dir": OUT_DIR
    }
    save_raw(summary, "summary_newsapi_everything")
    
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE LA RECOLECCIÓN")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_harvest()