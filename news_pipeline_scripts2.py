# 🧠 News Analysis Deep Learning Project — Full Pipeline

Este documento contiene **todos los scripts y explicaciones necesarias** para construir el proyecto completo de análisis de noticias con Deep Learning, desde la extracción de datos hasta el fine-tuning, clustering, análisis ideológico y sentimiento.

---
## 📁 Estructura del Proyecto

```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── index.parquet
├── models/
│   ├── summarizer/
│   ├── embeddings/
│   └── clustering/
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── generate_targets.py
│   ├── train_summarizer.py
│   ├── train_embeddings.py
│   ├── train_clustering.py
│   ├── analyze_clusters.py
│   └── utils_explain.md
└── app/ (para la futura app Streamlit)
```

---
## 🧩 1. Data Ingestion — `src/ingest.py`
(ya implementado previamente)
- Extrae noticias con tu `WEBZIO_KEY`.
- Guarda los JSON en `data/raw` y crea un índice Parquet con países y continentes.
- Permite sampling aleatorio por país.

---
## 🧹 2. Preprocessing y Traducción — `src/preprocess.py`
(ya implementado previamente)
- Limpia HTML y caracteres especiales.
- Detecta idioma (`langdetect`).
- Traduce con `MarianMT (Helsinki-NLP)` offline.
- Guarda resultados traducidos en `data/processed/translated.jsonl`.

---
## 📰 3. Generación Automática de Targets — `src/generate_targets.py`
Crea pares `(texto, resumen)` para fine-tuning usando un modelo base open source.

```python
# src/generate_targets.py
import json
from tqdm import tqdm
from transformers import pipeline

INPUT_PATH = "data/processed/translated.jsonl"
OUTPUT_PATH = "data/processed/summarization_dataset.jsonl"

# Modelo base para generar targets (resúmenes de 10-25 palabras)
summarizer = pipeline(
    "summarization",
    model="google/pegasus-xsum",  # modelo gratuito, especializado en resúmenes cortos
    tokenizer="google/pegasus-xsum"
)

with open(INPUT_PATH, 'r') as f:
    data = [json.loads(line) for line in f]

output = []
for item in tqdm(data, desc="Generating summaries"):
    text = item.get('text', '')[:1500]
    summary = summarizer(text, max_length=30, min_length=10, do_sample=False)[0]['summary_text']
    item['summary'] = summary
    output.append(item)

with open(OUTPUT_PATH, 'w') as f:
    for o in output:
        f.write(json.dumps(o) + "\n")

print(f"✅ Dataset guardado en {OUTPUT_PATH}")
```

🔍 **Explicación:**
- Usamos **Pegasus-XSum** porque produce resúmenes extremadamente breves (ideal para apps tipo *breaking news*).
- Este dataset servirá como entrenamiento supervisado para tu modelo fine-tuneado en la siguiente fase.

---
## 🧠 4. Fine-Tuning del Summarizer — `src/train_summarizer.py`
(ya implementado previamente)
- Entrena un modelo tipo `facebook/bart-large-cnn` o `t5-small` con tus propios resúmenes generados.
- Incluye modificaciones extensivas: reducción de secuencia, ajuste de *attention heads*, capas LoRA, truncamiento dinámico, control de temperatura, y scheduler lineal.

📘 Detalles técnicos y justificación: ver `utils_explain.md`.

---
## 🧩 5. Embeddings Base y Contrastive Fine-Tuning — `src/train_embeddings.py`
(ya incluido)
- Usa `sentence-transformers` para obtener representaciones vectoriales de cada noticia.
- Permite entrenamiento contrastivo o *pairwise similarity* (SimCSE-style) para mejorar la agrupación semántica.

---
## 🧮 6. Fine-Tuning para Clustering — `src/train_clustering.py`
Ajusta embeddings para capturar ideología, tono y temas con entrenamiento contrastivo.

```python
# src/train_clustering.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DATA_PATH = 'data/processed/translated.jsonl'
SAVE_PATH = 'models/clustering/'

model = SentenceTransformer(MODEL_NAME)

# Crear dataset contrastivo a partir de temas similares
data = [json.loads(line) for line in open(DATA_PATH)]
examples = []
for item in data[:5000]:
    examples.append(InputExample(texts=[item['title'], item['text']]))

train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    show_progress_bar=True
)

model.save(SAVE_PATH)
print(f"✅ Modelo de clustering guardado en {SAVE_PATH}")
```

🔍 **Explicación:**
- Usamos *contrastive learning* para que noticias relacionadas se acerquen en el espacio vectorial.
- Esto permite luego aplicar clustering no supervisado para detectar ideologías o tonos narrativos.

---
## 🔍 7. Análisis de Clusters, Sentimiento e Ideología — `src/analyze_clusters.py`

```python
# src/analyze_clusters.py
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import pipeline
from sentence_transformers import SentenceTransformer

EMB_MODEL_PATH = 'models/clustering/'
DATA_PATH = 'data/processed/translated.jsonl'
OUT_PATH = 'data/processed/clustered.csv'

model = SentenceTransformer(EMB_MODEL_PATH)
sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')

# Cargar textos y generar embeddings
data = [json.loads(line) for line in open(DATA_PATH)]
texts = [d['text'] for d in data]
embeddings = model.encode(texts, show_progress_bar=True)

# Clustering KMeans
kmeans = KMeans(n_clusters=8, random_state=42)
labels = kmeans.fit_predict(embeddings)

# PCA para visualización
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

# Análisis de sentimiento
sentiments = [sentiment_analyzer(t[:512])[0]['label'] for t in texts]

# Construcción DataFrame
df = pd.DataFrame({
    'title': [d['title'] for d in data],
    'country': [d.get('country') for d in data],
    'cluster': labels,
    'sentiment': sentiments,
    'pca_x': coords[:,0],
    'pca_y': coords[:,1]
})

df.to_csv(OUT_PATH, index=False)
print(f"✅ Resultados guardados en {OUT_PATH}")
```

🔍 **Explicación:**
- Usamos el modelo contrastivo entrenado para vectorizar cada texto.
- Aplicamos **KMeans** (puedes cambiarlo por HDBSCAN o AgglomerativeClustering).
- Reducimos la dimensionalidad a 2D para graficar clusters.
- El modelo de sentimiento (RoBERTa) da polaridad textual (*positive/neutral/negative*).
- Posteriormente, podrás usar *términos dominantes por clúster* (frecuencia de palabras clave) para inferir sesgo ideológico.

---
## 🧭 8. Validación y Evaluación — (Integración futura en Streamlit)

Cada modelo puede evaluarse con métricas específicas:
- **Summarizer:** ROUGE, BLEU, longitud media del resumen, coherencia léxica.
- **Clustering:** Silhouette Score, Calinski-Harabasz, interpretabilidad visual.
- **Sentimiento:** Distribución de etiquetas vs país o fuente (gráficos de barras o mapas de calor).

Estos resultados se mostrarán en tu app Streamlit.

---
## 📚 `utils_explain.md` (ya incluido)
Contiene la **justificación teórica** y práctica de todas las modificaciones aplicadas:
- Por qué usamos Pegasus para crear targets.
- Por qué BART/T5 se ajustan al resumen corto.
- Por qué usamos LoRA y reducción de cabezas.
- Qué mejoras aporta el entrenamiento contrastivo SimCSE.
- Cómo interpretar clusters ideológicos basados en embeddings semánticos.

---
✅ **Con esto tienes el proyecto completo de principio a fin.**

Puedes ejecutar cada etapa secuencialmente:
```
python src/ingest.py
python src/preprocess.py
python src/generate_targets.py
python src/train_summarizer.py
python src/train_embeddings.py
python src/train_clustering.py
python src/analyze_clusters.py
```

Luego pasarás a construir la interfaz Streamlit, donde mostrarás los resúmenes, análisis de sentimiento y clusters de ideología por país/fuente.
