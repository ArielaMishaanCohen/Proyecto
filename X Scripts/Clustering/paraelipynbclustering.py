# ==============================================================================
# 1. INSTALACIÓN Y CONFIGURACIÓN INICIAL
# ==============================================================================

# Instalar librerías necesarias (descomentar si es necesario en su entorno)
# !pip install sentence-transformers pandas numpy scikit-learn hdbscan matplotlib seaborn torch

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from collections import Counter
import random
import os

# CONFIGURACIÓN GLOBAL
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("2_Models/clustering")
DATA_PATH = "1_Data/processed/clustering_data_cleaned.csv"

# Parámetros (tomados de self.config de la clase original)
config = {
    'batch_size': 16,
    'num_epochs': 3,
    'learning_rate': 2e-5,
    'warmup_steps': 100,
    'evaluation_steps': 50,  # Evaluar cada 50 steps
    'save_steps': 100,
    'max_seq_length': 256,
    'min_cluster_size': 5,
    'min_samples': 3,
    'val_split': 0.2,
    'random_seed': 42
}

# Inicialización
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seed para reproducibilidad
random.seed(config['random_seed'])
np.random.seed(config['random_seed'])
torch.manual_seed(config['random_seed'])

# Cargar modelo base
print(f"Usando device: {device}")
print(f"Cargando modelo base: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
model.to(device)

# Log de configuración
print("\n" + "="*60)
print("CONFIGURACIÓN DE CLUSTERING")
print("="*60)
for key, value in config.items():
    print(f"{key}: {value}")
print("="*60 + "\n")

# ==============================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================

print(f"[PASO 1/7] Cargando datos desde: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Archivo no encontrado en {DATA_PATH}. Creando DataFrame de ejemplo.")
    # DataFrame de ejemplo para que el código no falle completamente
    df = pd.DataFrame({
        'text': [
            "El peso mexicano alcanza un nuevo mínimo histórico frente al dólar. Los analistas culpan a la inestabilidad política.",
            "La moneda nacional registra su peor desempeño en la última década. El Banco Central no ha intervenido aún.",
            "La NASA lanza con éxito su nueva misión a Marte. El objetivo es buscar rastros de agua en los polos.",
            "El vehículo Perseverance de la NASA ya está operando en la superficie marciana, enviando sus primeras imágenes.",
            "Los resultados de la votación en el congreso sorprenden a todos. La nueva ley fue aprobada por un margen mínimo.",
            "Noticias sobre un nuevo récord en la bolsa de valores tras la reunión de la Reserva Federal.",
            "El congreso votará hoy sobre la reforma fiscal propuesta por el presidente."
        ],
        'title': [
            "Peso en caída libre", "Peor día para el peso", "Misión Marte", "Perseverance", "Voto en congreso", "Bolsa", "Reforma Fiscal"
        ],
        'source': [
            "EconomiaHoy", "EconomiaHoy", "CienciaMX", "CienciaMX", "PoliticaNac", "MercadosGlobales", "PoliticaNac"
        ]
    })
    
# Validación de datos y limpieza
required_cols = ['text', 'title', 'source']
df = df.dropna(subset=required_cols)
print(f"Cargados y limpiados {len(df)} documentos.")

# Creación de pares de entrenamiento y validación
print("\n[PASO 2/7] Creando pares de entrenamiento y validación...")

all_examples = []
source_groups = df.groupby('source')['text'].apply(list).to_dict()

for source, texts in source_groups.items():
    if len(texts) < 3:
        continue
    for i in range(len(texts)):
        for j in range(i+1, min(i+4, len(texts))): # Max 3 pares por texto
            example = InputExample(
                texts=[texts[i], texts[j]],
                label=0.9
            )
            all_examples.append(example)

# TRAIN/VAL SPLIT
train_size = int(len(all_examples) * (1 - config['val_split']))
train_examples = all_examples[:train_size]
val_examples = all_examples[train_size:]

print(f"✓ Pares de entrenamiento: {len(train_examples)}")
print(f"✓ Pares de validación: {len(val_examples)}")

# ==============================================================================
# 3. CREACIÓN DEL TRIPLETEVALUATOR
# ==============================================================================

print("\n[PASO 3/7] Creando TripletEvaluator...")

# Crear datos para TripletEvaluator (anchor, positive, negative)
source_to_texts = df.groupby('source')['text'].apply(list).to_dict()
sources = list(source_to_texts.keys())

anchors, positives, negatives = [], [], []

for source in sources:
    texts = source_to_texts[source]
    if len(texts) < 2:
        continue
    
    for i in range(min(5, len(texts))): # Max 5 triplets por fuente
        if i + 1 >= len(texts):
            break
        
        anchor = texts[i]
        positive = texts[i + 1]
        
        # Seleccionar negative de otra fuente
        other_sources = [s for s in sources if s != source]
        if not other_sources:
            continue
        
        neg_source = random.choice(other_sources)
        # Asegurar que la fuente negativa tiene textos
        if not source_to_texts[neg_source]:
             continue
        
        negative = random.choice(source_to_texts[neg_source])
        
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)

if len(anchors) == 0:
    print("⚠ No se pudieron crear triplets suficientes. Usando evaluador Dummy.")
    # Si no hay suficientes datos para triplets reales
    evaluator = None 
else:
    print(f"✓ Creados {len(anchors)} triplets para evaluación")
    evaluator = TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name='validation',
        show_progress_bar=True
    )

# ==============================================================================
# 4. FINE-TUNING DEL MODELO (CONTRASTIVE LEARNING)
# ==============================================================================

print("\n" + "="*60)
print("[PASO 4/7] INICIANDO FINE-TUNING DE CLUSTERING")
print("="*60)

# DataLoader
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=config['batch_size']
)

# Loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Calcular steps
num_train_steps = len(train_dataloader) * config['num_epochs']
warmup_steps = min(config['warmup_steps'], num_train_steps // 10)

print(f"Pasos totales: {num_train_steps}")
print(f"Warmup steps: {warmup_steps}")

if len(train_examples) > 0 and evaluator:
    # ENTRENAR CON EVALUACIÓN
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config['num_epochs'],
        warmup_steps=warmup_steps,
        output_path=str(OUTPUT_DIR / "fine_tuned_model"),
        show_progress_bar=True,
        evaluation_steps=config['evaluation_steps'],
        save_best_model=True, # Guarda el mejor modelo según TripletEvaluator
        optimizer_params={'lr': config['learning_rate']},
    )

    print("\n✓ Fine-tuning completado! Modelo guardado en 'fine_tuned_model'")
    
    # Cargar el mejor modelo (fine-tuned) para las siguientes etapas
    print("\n[PASO 5/7] Cargando mejor modelo fine-tuned...")
    model = SentenceTransformer(
        str(OUTPUT_DIR / "fine_tuned_model")
    )
    model.to(device)

else:
    print("\n⚠ Omitting Fine-Tuning: No hay suficientes pares de entrenamiento o evaluador.")
    print("Se usará el modelo base para el clustering.")


# Guardar configuración final (incluyendo el caso de no fine-tuning)
config_to_save = config.copy()
config_to_save['model_name'] = MODEL_NAME
config_to_save['num_train_examples'] = len(train_examples)
config_to_save['num_val_examples'] = len(val_examples)

with open(OUTPUT_DIR / "training_config.json", 'w') as f:
    json.dump(config_to_save, f, indent=2)

print(f"✓ Configuración final guardada en {OUTPUT_DIR / 'training_config.json'}")

# ==============================================================================
# 5. GENERACIÓN DE EMBEDDINGS
# ==============================================================================

print("\n" + "="*60)
print("[PASO 6/7] GENERANDO EMBEDDINGS")
print("="*60)

texts = df['text'].tolist()

# Generar embeddings en batches
embeddings = model.encode(
    texts,
    batch_size=config['batch_size'],
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True # CLAVE: Normalizar para mejor clustering
)

print(f"✓ Embeddings generados: {embeddings.shape}")
np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

# ==============================================================================
# 6. EJECUCIÓN DEL CLUSTERING (HDBSCAN)
# ==============================================================================

print("\n" + "="*60)
print("[PASO 7/7] EJECUTANDO CLUSTERING (HDBSCAN)")
print("="*60)

# HDBSCAN (clustering density-based)
clusterer = HDBSCAN(
    min_cluster_size=config['min_cluster_size'],
    min_samples=config['min_samples'],
    metric='euclidean',
    cluster_selection_method='eom'
)

cluster_labels = clusterer.fit_predict(embeddings)

# Estadísticas
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"\n✓ Clusters encontrados: {n_clusters}")
print(f"✓ Puntos outlier: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")

# Distribución
cluster_counts = Counter(cluster_labels)
print(f"\nDistribución de clusters:")
for cluster_id, count in sorted(cluster_counts.items()):
    if cluster_id != -1:
        print(f"  Cluster {cluster_id}: {count} documentos")

np.save(OUTPUT_DIR / "cluster_labels.npy", cluster_labels)

# ==============================================================================
# 6.5 EXPORTACIÓN DE RESULTADOS FINALES (ANTES DEL ANÁLISIS)
# ==============================================================================

print("\n" + "="*60)
print("EXPORTANDO RESULTADOS CONSOLIDADOS A CSV")
print("="*60)

# Reducir dimensionalidad para incluir en CSV (opcional, pero útil para visualizar)
# Usamos el mismo PCA que se usaría en el paso 7 para mantener consistencia
pca_export = PCA(n_components=2, random_state=config['random_seed'])
embeddings_2d_export = pca_export.fit_transform(embeddings)

# Crear DataFrame de resultados
df_results = df.copy()
df_results['cluster_label'] = cluster_labels
df_results['pca_dim_1'] = embeddings_2d_export[:, 0]
df_results['pca_dim_2'] = embeddings_2d_export[:, 1]

# Definir la ruta del archivo de salida
output_csv_path = OUTPUT_DIR / "clustering_results_consolidated.csv"

# Exportar a CSV
df_results.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"✓ Resultados consolidados exportados con éxito a:")
print(f"  -> {output_csv_path}")
print("Columnas: [text, title, source, cluster_label, pca_dim_1, pca_dim_2]")

# La variable df_results ahora está lista para ser cargada en el notebook de análisis.
# ==============================================================================
# 7. EVALUACIÓN Y VISUALIZACIÓN DE RESULTADOS
# ==============================================================================

# ====================
# 7.1 Evaluación
# ====================
print("\n" + "="*60)
print("EVALUANDO CLUSTERING")
print("="*60)

# Filtrar outliers (-1) para las métricas
mask = cluster_labels != -1
filtered_embeddings = embeddings[mask]
filtered_labels = cluster_labels[mask]

metrics = {}

if len(set(filtered_labels)) >= 2:
    # Silhouette Score
    silhouette = silhouette_score(filtered_embeddings, filtered_labels)
    print(f"\nSilhouette Score: {silhouette:.4f}")
    
    # Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
    print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
    
    metrics = {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'n_clusters': len(set(filtered_labels)),
        'n_outliers': int((cluster_labels == -1).sum()),
        'outlier_percentage': float((cluster_labels == -1).sum() / len(cluster_labels) * 100)
    }
    
    with open(OUTPUT_DIR / "clustering_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Métricas guardadas en {OUTPUT_DIR / 'clustering_metrics.json'}")

else:
    print("⚠ Muy pocos clusters para calcular métricas (Se necesita al menos 2 clusters).")


# ====================
# 7.2 Visualización (PCA)
# ====================
print("\n" + "="*60)
print("CREANDO VISUALIZACIONES Y ANÁLISIS")
print("="*60)

# Reducción de dimensionalidad
pca = PCA(n_components=2, random_state=config['random_seed'])
embeddings_2d = pca.fit_transform(embeddings)
variance_explained = pca.explained_variance_ratio_.sum()
print(f"Varianza explicada por PCA: {variance_explained:.2%}")

# Plot
plt.figure(figsize=(14, 10))
unique_labels = sorted(set(cluster_labels))
n_clusters = len([l for l in unique_labels if l != -1])
colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
color_map = {l: colors[i] for i, l in enumerate([l for l in unique_labels if l != -1])}
color_map[-1] = 'black'

for label in unique_labels:
    mask = cluster_labels == label
    marker = 'x' if label == -1 else 'o'
    label_text = 'Outliers' if label == -1 else f'Cluster {label}'
    alpha = 0.3 if label == -1 else 0.6
    size = 30 if label == -1 else 50
    
    plt.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=[color_map[label]],
        label=label_text,
        marker=marker,
        alpha=alpha,
        s=size,
        edgecolors='white',
        linewidth=0.5
    )

plt.xlabel('Primera Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.title(f'Clustering de Noticias (PCA 2D - {variance_explained:.1%} varianza)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

viz_path = OUTPUT_DIR / "clusters_visualization.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualización guardada en {viz_path}")
plt.close()


# ====================
# 7.3 Análisis de Clusters
# ====================
df_with_labels = df.copy()
df_with_labels['cluster'] = cluster_labels
analysis = []

print("\n" + "="*60)
print("ANÁLISIS DE CLUSTERS (Ejemplos de Títulos)")
print("="*60)

for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        continue
    
    cluster_docs = df_with_labels[df_with_labels['cluster'] == cluster_id]
    sample_titles = cluster_docs['title'].head(3).tolist()
    top_sources = cluster_docs['source'].value_counts().head(2)
    
    cluster_info = {
        'cluster_id': int(cluster_id),
        'size': len(cluster_docs),
        'sample_titles': sample_titles,
        'top_sources': top_sources.to_dict()
    }
    analysis.append(cluster_info)
    
    print(f"\nCluster {cluster_id} (Tamaño: {len(cluster_docs)}):")
    for i, title in enumerate(sample_titles, 1):
        print(f"  {i}. {title}")
    if not top_sources.empty:
        print(f"  Fuentes principales: {', '.join(top_sources.index.tolist())}")

# Análisis de outliers
outliers = df_with_labels[df_with_labels['cluster'] == -1]
if len(outliers) > 0:
    print(f"\nOutliers (Noticias únicas - Total: {len(outliers)}):")
    for i, (_, row) in enumerate(outliers.head(2).iterrows(), 1):
        print(f"  {i}. {row['title']}")

# Guardar análisis
analysis_path = OUTPUT_DIR / "cluster_analysis.json"
with open(analysis_path, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print(f"\n✓ Análisis guardado en {analysis_path}")








# ==============================================================================
# 7. EVALUACIÓN Y VISUALIZACIÓN DE RESULTADOS (CARGA DESDE CSV)
# ==============================================================================

# Definir la ruta de entrada (asumiendo que los archivos están en el mismo OUTPUT_DIR)
output_csv_path = OUTPUT_DIR / "clustering_results_consolidated.csv"
labels_npy_path = OUTPUT_DIR / "cluster_labels.npy"
embeddings_npy_path = OUTPUT_DIR / "embeddings.npy"

print("\n" + "="*60)
print("INICIANDO ANÁLISIS Y VISUALIZACIÓN (Carga desde CSV)")
print("="*60)

try:
    # Cargar los resultados consolidados
    df_results = pd.read_csv(output_csv_path)
    print(f"✓ Cargados {len(df_results)} registros desde el CSV.")
    
    # Extraer las columnas necesarias
    cluster_labels = df_results['cluster_label'].values
    embeddings_2d = df_results[['pca_dim_1', 'pca_dim_2']].values
    
    # El DataFrame original 'df' ahora es 'df_results'
    df = df_results
    
except FileNotFoundError:
    print(f"⛔ ERROR: No se encontró el archivo de resultados en {output_csv_path}.")
    print("Asegúrate de haber ejecutado la celda 6.5 (Exportación) previamente.")
    # Detener la ejecución o salir
    # return
    raise

# ====================
# 7.1 Evaluación de Calidad (Requiere Embeddings de Alta Dimensión)
# ====================
print("\n" + "="*60)
print("EVALUANDO CLUSTERING (Métricas de Calidad)")
print("="*60)

metrics = {}
# Intentar cargar los embeddings de alta dimensión para métricas
try:
    embeddings = np.load(embeddings_npy_path)
    
    # Filtrar outliers (-1) para las métricas
    mask = cluster_labels != -1
    filtered_embeddings = embeddings[mask]
    filtered_labels = cluster_labels[mask]
    
    if len(set(filtered_labels)) >= 2:
        # Silhouette Score
        silhouette = silhouette_score(filtered_embeddings, filtered_labels)
        print(f"\nSilhouette Score (384D): {silhouette:.4f}")
        
        # Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
        print(f"Davies-Bouldin Index (384D): {davies_bouldin:.4f}")
        
        metrics = {
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'n_clusters': len(set(filtered_labels)),
            'n_outliers': int((cluster_labels == -1).sum()),
            'outlier_percentage': float((cluster_labels == -1).sum() / len(cluster_labels) * 100)
        }
        
        with open(OUTPUT_DIR / "clustering_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Métricas guardadas en {OUTPUT_DIR / 'clustering_metrics.json'}")
    else:
        print("⚠ Muy pocos clusters válidos para calcular métricas (Se necesita al menos 2 clusters).")
        
except FileNotFoundError:
    print(f"⚠ ADVERTENCIA: No se encontró el archivo de embeddings de alta dimensión en {embeddings_npy_path}.")
    print("Las métricas de calidad (Silhouette, Davies-Bouldin) no se pueden calcular.")
    print("Continuando con Visualización y Análisis de Contenido.")


# ====================
# 7.2 Visualización (PCA 2D)
# ====================
print("\n" + "="*60)
print("CREANDO VISUALIZACIÓN DE CLUSTERS")
print("="*60)

# El CSV ya tiene los resultados de PCA, solo necesitamos la Varianza explicada para el título
# En un escenario real, re-calcularías PCA, o guardarías este valor en el config/metadata
try:
    # Intentar cargar los embeddings de alta dimensión para recalcular PCA solo para el ratio
    embeddings_full = np.load(embeddings_npy_path)
    pca_plot = PCA(n_components=2, random_state=config['random_seed'])
    pca_plot.fit(embeddings_full)
    variance_explained = pca_plot.explained_variance_ratio_.sum()
except:
    # Usar un valor por defecto si no se pudo cargar el archivo original
    variance_explained = 0.50 

# Plot
plt.figure(figsize=(14, 10))
unique_labels = sorted(set(cluster_labels))
n_clusters = len([l for l in unique_labels if l != -1])
colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
color_map = {l: colors[i] for i, l in enumerate([l for l in unique_labels if l != -1])}
color_map[-1] = 'black'

for label in unique_labels:
    mask = cluster_labels == label
    marker = 'x' if label == -1 else 'o'
    label_text = 'Outliers' if label == -1 else f'Cluster {label}'
    alpha = 0.3 if label == -1 else 0.6
    size = 30 if label == -1 else 50
    
    plt.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=[color_map[label]],
        label=label_text,
        marker=marker,
        alpha=alpha,
        s=size,
        edgecolors='white',
        linewidth=0.5
    )

plt.xlabel('Primera Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.title(f'Clustering de Noticias (PCA 2D - {variance_explained:.1%} varianza)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

viz_path = OUTPUT_DIR / "clusters_visualization.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualización guardada en {viz_path}")
plt.close()


# ====================
# 7.3 Análisis de Contenido
# ====================
print("\n" + "="*60)
print("ANÁLISIS DE CLUSTERS (Ejemplos de Títulos)")
print("="*60)

analysis = []
df_with_labels = df_results # Usamos el DataFrame cargado

for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        continue
    
    cluster_docs = df_with_labels[df_with_labels['cluster_label'] == cluster_id]
    sample_titles = cluster_docs['title'].head(3).tolist()
    top_sources = cluster_docs['source'].value_counts().head(2)
    
    cluster_info = {
        'cluster_id': int(cluster_id),
        'size': len(cluster_docs),
        'percentage': float(len(cluster_docs) / len(df_results) * 100),
        'sample_titles': sample_titles,
        'top_sources': top_sources.to_dict()
    }
    analysis.append(cluster_info)
    
    print(f"\nCluster {cluster_id} (Tamaño: {len(cluster_docs)}, {cluster_info['percentage']:.1f}%):")
    for i, title in enumerate(sample_titles, 1):
        print(f"  {i}. {title}")
    if not top_sources.empty:
        print(f"  Fuentes principales: {', '.join(top_sources.index.tolist())}")

# Análisis de outliers
outliers = df_with_labels[df_with_labels['cluster_label'] == -1]
if len(outliers) > 0:
    print(f"\nOutliers (Noticias únicas - Total: {len(outliers)}):")
    for i, (_, row) in enumerate(outliers.head(2).iterrows(), 1):
        print(f"  {i}. {row['title']}")

# Guardar análisis
analysis_path = OUTPUT_DIR / "cluster_analysis.json"
with open(analysis_path, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print(f"\n✓ Análisis guardado en {analysis_path}")