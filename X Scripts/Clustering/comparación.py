## 1. Librerías y configuraciones

import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator

# ==============================================================================
# 2. DEFINICIÓN DE CONFIGURACIONES
# ==============================================================================

# 🚨 Rutas y entorno
BASE_DIR = Path.cwd().parent
DATA_PATH = BASE_DIR / "1_Data" / "processed" / "clustering_data_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "2_Modelos" / "clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -- 2.1. Modelos Base a Comparar (Encoder) --
MODELS_TO_COMPARE = {
    "distilbert": "distilbert-base-uncased",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2", 
}

# -- 2.2. Parámetros FIJOS para el Fine-Tuning (Tu configuración original) --
FINE_TUNING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 3,
    'learning_rate': 2e-5,
    'warmup_steps': 100,
    'evaluation_steps': 50,
    'save_steps': 100,
    'max_seq_length': 256,
    'val_split': 0.2,
    'random_seed': RANDOM_SEED
}

# -- 2.3. Parámetros VARIABLES para el HDBSCAN (3 Configuraciones de Clustering) --
CONFIGS = {
    # Configuración 1: Conservadora (Clusters grandes y limpios, métrica L2)
    "config_1": {
        "MIN_CLUSTER_SIZE": 15,
        "MIN_SAMPLES": 8,
        "METRIC": 'euclidean',
    },
    # Configuración 2: Detallada (Clusters más pequeños, métrica coseno)
    "config_2": {
        "MIN_CLUSTER_SIZE": 5,
        "MIN_SAMPLES": 3,
        "METRIC": 'cosine',
    },
    # Configuración 3: Intermedia (Balanceada, métrica L2)
    "config_3": {
        "MIN_CLUSTER_SIZE": 20,
        "MIN_SAMPLES": 10,
        "METRIC": 'euclidean',
    }
}

# Diccionario para almacenar resultados de todos los 6 experimentos (2 Modelos x 3 Configs)
all_results = {}

# Log de configuración
print("="*80)
print(f"DEVICE: {DEVICE} | SEED: {RANDOM_SEED}")
print(f"MODELOS A COMPARAR: {list(MODELS_TO_COMPARE.keys())}")
print(f"CONFIGURACIONES HDBSCAN: {list(CONFIGS.keys())}")
print("="*80)

## 3. Carga y Preparación de Datos

def load_data(path: Path) -> pd.DataFrame:
    """Carga el archivo CSV y realiza una limpieza inicial."""
    df = pd.read_csv(path)
    required_cols = ['text', 'title', 'source']
    df = df.dropna(subset=required_cols)
    df['text'] = df['text'].astype(str)
    print(f"✓ Datos cargados desde: {path}")
    print(f"Registros totales: {len(df)}")
    return df

df = load_data(DATA_PATH)
text_list = df['text'].tolist()

# 3.1. Preparación de pares para Fine-Tuning (Train/Val)
all_examples = []
source_groups = df.groupby('source')['text'].apply(list).to_dict()

for source, texts in source_groups.items():
    if len(texts) < 3: continue
    for i in range(len(texts)):
        for j in range(i+1, min(i+4, len(texts))): # Max 3 pares por texto
            all_examples.append(InputExample(texts=[texts[i], texts[j]], label=0.9))

train_size = int(len(all_examples) * (1 - FINE_TUNING_CONFIG['val_split']))
train_examples = all_examples[:train_size]
val_examples = all_examples[train_size:]

print(f"✓ Pares de entrenamiento (Similarity): {len(train_examples)}")
print(f"✓ Pares de validación (Similarity): {len(val_examples)}")

# 3.2. Creación del Triplet Evaluator (Para evaluar el Fine-Tuning)
source_to_texts = df.groupby('source')['text'].apply(list).to_dict()
sources = list(source_to_texts.keys())
anchors, positives, negatives = [], [], []

for source in sources:
    texts = source_to_texts[source]
    if len(texts) < 2: continue
    for i in range(min(5, len(texts))): 
        if i + 1 >= len(texts): break
        anchor = texts[i]
        positive = texts[i + 1]
        other_sources = [s for s in sources if s != source]
        if not other_sources: continue
        
        neg_source = random.choice(other_sources)
        if not source_to_texts[neg_source]: continue
        
        negative = random.choice(source_to_texts[neg_source])
        
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)

if len(anchors) == 0:
    print("⚠ No se pudieron crear triplets suficientes. Evaluación de fine-tuning desactivada.")
    evaluator = None 
else:
    print(f"✓ Creados {len(anchors)} triplets para evaluación")
    evaluator = TripletEvaluator(anchors=anchors, positives=positives, negatives=negatives, name='validation', show_progress_bar=False)

# ==============================================================================
# 4. FUNCIÓN CENTRAL DE EXPERIMENTACIÓN
# ==============================================================================

def run_experiment(model_key: str, model_name: str, config: Dict) -> Dict:
    """
    1. Fine-Tunes el modelo base.
    2. Genera los embeddings con el modelo fine-tuned.
    3. Aplica HDBSCAN para cada una de las 3 configuraciones.
    4. Guarda los resultados intermedios.
    """
    
    experiment_model_dir = OUTPUT_DIR / f"exp_model_{model_key}"
    experiment_model_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}\nMODELO: {model_key.upper()} ({model_name})\n{'='*80}")
    
    # --- PASO 1: FINE-TUNING ---
    ft_output_path = experiment_model_dir / "fine_tuned_model"
    
    try:
        # Cargar modelo base
        model = SentenceTransformer(model_name)
        model.to(DEVICE)
        
        # DataLoader y Loss
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=FINE_TUNING_CONFIG['batch_size'])
        train_loss = losses.MultipleNegativesRankingLoss(model)
        num_train_steps = len(train_dataloader) * FINE_TUNING_CONFIG['num_epochs']
        warmup_steps = min(FINE_TUNING_CONFIG['warmup_steps'], num_train_steps // 10)
        
        print(f"-> Iniciando Fine-Tuning (Steps: {num_train_steps}, Epochs: {FINE_TUNING_CONFIG['num_epochs']})")
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=FINE_TUNING_CONFIG['num_epochs'],
            warmup_steps=warmup_steps,
            output_path=str(ft_output_path),
            show_progress_bar=False, # Desactivar para evitar LookupError
            evaluation_steps=FINE_TUNING_CONFIG['evaluation_steps'],
            save_best_model=True,
            optimizer_params={'lr': FINE_TUNING_CONFIG['learning_rate']},
        )
        
        # Cargar el mejor modelo fine-tuned para generar embeddings
        model = SentenceTransformer(str(ft_output_path))
        model.to(DEVICE)
        print("-> Fine-Tuning completado. Cargado el mejor modelo.")
        
    except Exception as e:
        print(f"\n❌ Error en Fine-Tuning para {model_key}: {str(e)}. Usando modelo base sin FT.")
        model = SentenceTransformer(model_name) # Si falla FT, usa el base pre-entrenado
        model.to(DEVICE)
        
    # --- PASO 2: GENERAR EMBEDDINGS ---
    print("-> Generando Embeddings...")
    embeddings_np = model.encode(
        text_list,
        batch_size=FINE_TUNING_CONFIG['batch_size'],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True 
    )
    print(f"-> Embeddings generados: {embeddings_np.shape}")

    # Liberar modelo y caché de GPU antes del Clustering (Crítico para memoria)
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    # --- PASO 3: APLICAR CLUSTERING (3 configuraciones) ---
    model_results = {}
    
    for config_name, cluster_config in CONFIGS.items():
        experiment_key = f"{model_key}_{config_name}"
        print(f"\n--- INICIANDO {experiment_key} (MinSize: {cluster_config['MIN_CLUSTER_SIZE']}) ---")
        
        try:
            hdbscan = HDBSCAN(
                min_cluster_size=cluster_config["MIN_CLUSTER_SIZE"],
                min_samples=cluster_config["MIN_SAMPLES"],
                metric=cluster_config["METRIC"],
            )
            cluster_labels = hdbscan.fit_predict(embeddings_np)
            
            # Cálculo de Métricas (Solo puntos no-ruido)
            non_noise_indices = cluster_labels != -1
            labels_filtered = cluster_labels[non_noise_indices]
            embeddings_filtered = embeddings_np[non_noise_indices]
            n_clusters = len(np.unique(labels_filtered))
            n_noise = np.sum(cluster_labels == -1)

            if n_clusters < 2 or len(labels_filtered) < 2:
                s_score = -999.0
                db_score = 999.0
            else:
                s_score = silhouette_score(embeddings_filtered, labels_filtered)
                db_score = davies_bouldin_score(embeddings_filtered, labels_filtered)
            
            metrics = {
                "n_clusters": n_clusters,
                "n_noise_points": int(n_noise),
                "silhouette_score": float(s_score), # Métrica a maximizar
                "davies_bouldin_score": float(db_score),
            }

            all_results[experiment_key] = {
                "model_key": model_key,
                "model_name": model_name,
                "config_name": config_name,
                "config": cluster_config,
                "metrics": metrics,
                "cluster_labels": cluster_labels,
                "embeddings": embeddings_np, # Se usa para la exportación final
            }
            
            print(f"✅ {experiment_key} completado. Silhouette: {s_score:.4f}, Clusters: {n_clusters}")
            
        except Exception as e:
            print(f"\n❌ Error en {experiment_key}: {str(e)}")
            continue

# ==============================================================================
# 5. EJECUCIÓN DEL BUCLE DE EXPERIMENTACIÓN
# ==============================================================================

for model_key, model_name in MODELS_TO_COMPARE.items():
    run_experiment(model_key, model_name, CONFIGS)

# ==============================================================================
# 6. EVALUACIÓN Y EXPORTACIÓN FINAL
# ==============================================================================

def evaluate_and_export_best(df: pd.DataFrame, all_results: Dict, output_dir: Path):
    """Selecciona el mejor experimento y exporta los resultados."""
    
    if not all_results:
        print("\n❌ No se pudo completar ningún experimento. No se puede exportar.")
        return

    # 1. Consolidar resultados y seleccionar el mejor
    df_results_summary = pd.DataFrame.from_dict({
        k: {
            **v['config'], 
            **v['metrics'], 
            'model_key': v['model_key'],
            'experiment_key': k,
        } 
        for k, v in all_results.items()
    }, orient='index').reset_index(names=['experiment_key'])
    
    # Seleccionar el mejor: Maximizando Silhouette Score
    best_experiment_key = df_results_summary['silhouette_score'].idxmax()
    best_model_data = all_results[best_experiment_key]
    
    # Guardar la tabla de comparación
    output_summary_path = output_dir / "clustering_model_comparison.csv"
    df_results_summary.to_csv(output_summary_path, index=False)
    
    # 2. Exportar el DataFrame final del mejor modelo
    best_cluster_labels = best_model_data['cluster_labels']
    best_embeddings = best_model_data['embeddings']
    
    # Aplicar PCA para la visualización (exacto al script original)
    pca_export = PCA(n_components=2, random_state=RANDOM_SEED)
    embeddings_2d_export = pca_export.fit_transform(best_embeddings)
    
    # Crear y exportar DataFrame final
    df_results = df.copy()
    df_results['cluster_label'] = best_cluster_labels
    df_results['pca_dim_1'] = embeddings_2d_export[:, 0]
    df_results['pca_dim_2'] = embeddings_2d_export[:, 1]
    
    # Exportar a CSV (mismas columnas que el script original)
    output_csv_path = output_dir / "clustering_results_consolidated.csv"
    df_results[['text', 'title', 'source', 'cluster_label', 'pca_dim_1', 'pca_dim_2']].to_csv(
        output_csv_path, 
        index=False, 
        encoding='utf-8'
    )
    
    # 3. Guardar la configuración del mejor experimento (similar al original)
    best_config_to_save = {
        'fine_tuning_config': FINE_TUNING_CONFIG,
        'best_clustering_config': best_model_data['config'],
        'model_name': best_model_data['model_name'],
        'experiment_key': best_experiment_key,
        'metrics': best_model_data['metrics']
    }
    with open(output_dir / "best_clustering_config_and_metrics.json", 'w') as f:
        json.dump(best_config_to_save, f, indent=2)

    # 4. Mostrar Resumen
    print("\n" + "="*80)
    print("✅ MEJOR MODELO SELECCIONADO Y RESULTADOS EXPORTADOS")
    print("="*80)
    print(f"Mejor Experimento: {best_experiment_key.upper()}")
    print(f"  - Modelo: {best_model_data['model_name']}")
    print(f"  - Configuración HDBSCAN: {best_model_data['config_name']}")
    print(f"  - Silhouette Score (Max): {best_model_data['metrics']['silhouette_score']:.4f}")
    
    print("\n📁 Archivos generados:")
    print(f"  -> {output_summary_path}")
    print(f"  -> {output_csv_path}")


# Ejecutar la evaluación y exportación
evaluate_and_export_best(df, all_results, OUTPUT_DIR)










## 7. Análisis Final del Mejor Modelo Seleccionado

import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter
from pathlib import Path

# --- Rutas de Archivos (Ajustadas a la nueva estructura) ---
OUTPUT_DIR = Path.cwd().parent / "2_Modelos" / "clustering"
output_csv_path = OUTPUT_DIR / "clustering_results_consolidated.csv"
metrics_json_path = OUTPUT_DIR / "best_clustering_config_and_metrics.json"
analysis_path = OUTPUT_DIR / "cluster_analysis.json"
viz_path = OUTPUT_DIR / "clusters_visualization.png"


# --- 1. Cargar Datos y Métricas ---
try:
    df_results = pd.read_csv(output_csv_path)
    print(f"✓ Cargados {len(df_results)} registros desde el CSV.")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo de resultados: {output_csv_path}. Asegúrese de que el paso anterior se ejecutó correctamente.")
    exit()

try:
    with open(metrics_json_path, 'r') as f:
        best_model_data = json.load(f)
    metrics = best_model_data['metrics']
    print(f"✓ Métricas cargadas para el mejor experimento: {best_model_data['experiment_key']}.")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo de métricas: {metrics_json_path}. No se puede mostrar el resumen completo.")
    metrics = {}

# Extraer las columnas necesarias
cluster_labels = df_results['cluster_label'].values
embeddings_2d = df_results[['pca_dim_1', 'pca_dim_2']].values
df_with_labels = df_results # Alias para claridad
# --- 2. Definición de Stop Words y Función de Limpieza ---

# Lista de stop words en INGLÉS de NLTK
try:
    # Usamos la lista de NLTK para inglés
    ENGLISH_STOP_WORDS = set(stopwords.words('english')) 
except NameError:
    print("❌ ADVERTENCIA: NLTK no pudo cargar stopwords. Usando una lista vacía.")
    ENGLISH_STOP_WORDS = set()


def clean_text_for_wordcloud(text: str) -> str:
    """Limpia el texto: minúsculas, elimina puntuación y números."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text) # Eliminar puntuación
    text = re.sub(r'\d+', '', text)      # Eliminar números
    return text

def get_top_keywords(text_list: list, n_keywords: int = 20) -> Counter:
    """Cuenta la frecuencia de palabras después de limpieza y stop word removal."""
    all_words = []
    STOP_WORDS_SET = ENGLISH_STOP_WORDS # Usamos las palabras clave en inglés
    for text in text_list:
        clean_t = clean_text_for_wordcloud(text)
        words = clean_t.split()
        for word in words:
            if word and word not in STOP_WORDS_SET and len(word) > 2:
                all_words.append(word)
    return Counter(all_words).most_common(n_keywords)

# --- 3. Imprimir Métricas (ya calculadas) ---
if metrics:
    print("\n" + "="*50)
    print("MÉTRICAS DEL MEJOR MODELO")
    print("="*50)
    print(f"Modelo / Config: {best_model_data['experiment_key']}")
    print(f"Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}")
    print(f"Davies-Bouldin Index: {metrics.get('davies_bouldin_score', 'N/A'):.4f}")
    print(f"Clusters Encontrados: {metrics.get('n_clusters', 'N/A')}")
    print(f"Puntos Ruido (#): {metrics.get('n_noise_points', 'N/A')}")

# --- 4. Análisis por Cluster y Generación de Nubes de Palabras ---

analysis = []
unique_labels = sorted(set(cluster_labels))

# Crear directorio para las nubes de palabras
wordcloud_dir = OUTPUT_DIR / "cluster_wordclouds"
wordcloud_dir.mkdir(exist_ok=True)

print("\n" + "="*50)
print("ANÁLISIS DESCRIPTIVO POR CLUSTER")
print("="*50)

# Inicializar WordCloud
# Las stopwords de la WordCloud también usan el set de inglés
wc = WordCloud(
    background_color="white", 
    width=800, 
    height=400, 
    max_words=50, 
    stopwords=ENGLISH_STOP_WORDS # Usamos las stopwords en inglés
)

# Colores para el gráfico PCA
n_clusters = len([l for l in unique_labels if l != -1])
colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
color_map = {l: colors[i] for i, l in enumerate([l for l in unique_labels if l != -1])}
color_map[-1] = 'black'

# Iterar sobre cada cluster
for cluster_id in unique_labels:
    cluster_docs = df_with_labels[df_with_labels['cluster_label'] == cluster_id]
    cluster_texts = cluster_docs['text'].tolist()
    
    # ----------------------------------------------------
    # A. Análisis de Outliers (-1)
    # ----------------------------------------------------
    if cluster_id == -1:
        if len(cluster_docs) > 0:
            print(f"\nOutliers (Noticias únicas - Total: {len(cluster_docs)}):")
            for i, (_, row) in enumerate(cluster_docs.head(2).iterrows(), 1):
                print(f"  {i}. {row['title']}")
        continue

    # ----------------------------------------------------
    # B. Análisis de Cluster Válido
    # ----------------------------------------------------
    
    # Palabras Clave
    top_keywords = get_top_keywords(cluster_texts, n_keywords=10)
    keyword_freq_dict = dict(top_keywords)
    top_keywords_text = ", ".join([k for k, v in top_keywords])
    
    # Fuentes más frecuentes
    top_sources = cluster_docs['source'].value_counts().head(3)
    
    # Generar Nube de Palabras
    try:
        if keyword_freq_dict:
            wc.generate_from_frequencies(keyword_freq_dict)
            wc_path = wordcloud_dir / f"cluster_{cluster_id}_wordcloud.png"
            wc.to_file(str(wc_path))
        else:
            wc_path = None
    except Exception as e:
        print(f"  ⚠️ Error al generar WordCloud para Cluster {cluster_id}: {e}")
        wc_path = None


    # Guardar la información para el JSON final
    cluster_info = {
        'cluster_id': int(cluster_id),
        'size': len(cluster_docs),
        'percentage': float(len(cluster_docs) / len(df_results) * 100),
        'top_keywords': keyword_freq_dict,
        'sample_titles': cluster_docs['title'].head(3).tolist(),
        'top_sources': top_sources.to_dict(),
        'wordcloud_file': str(wc_path) if wc_path else 'N/A'
    }
    analysis.append(cluster_info)
    
    # Imprimir resumen
    print(f"\nCluster {cluster_id} (Tamaño: {len(cluster_docs)}, {cluster_info['percentage']:.1f}%):")
    print(f"  Palabras Clave: {top_keywords_text}")
    
    if not top_sources.empty:
        print(f"  Fuentes principales: {', '.join(top_sources.index.tolist())}")
    
    print(f"  Ejemplos Títulos: 1. {cluster_info['sample_titles'][0]}")
    if len(cluster_info['sample_titles']) > 1:
        print(f"                  2. {cluster_info['sample_titles'][1]}")
    
# --- 5. Visualización PCA (Tu código de plot con mejoras) ---

print("\n" + "="*50)
print("VISUALIZACIÓN PCA 2D")
print("="*50)

plt.figure(figsize=(14, 10))

# Usar los embeddings 2D ya exportados en el CSV
for label in unique_labels:
    mask = cluster_labels == label
    marker = 'x' if label == -1 else 'o'
    label_text = 'Outliers' if label == -1 else f'Cluster {label} ({np.sum(mask)} docs)'
    alpha = 0.2 if label == -1 else 0.6
    size = 20 if label == -1 else 40
    
    color = color_map.get(label, 'black')
    
    plt.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=[color],
        label=label_text,
        marker=marker,
        alpha=alpha,
        s=size,
        edgecolors='white',
        linewidth=0.5
    )

plt.xlabel('Primera Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.title(f'Clustering de Noticias (PCA 2D)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(viz_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Visualización de clusters (PCA) guardada en {viz_path}")
plt.close() 


# --- 6. Guardar el Análisis Final ---
with open(analysis_path, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print(f"\n✓ Análisis de clusters guardado en {analysis_path}")