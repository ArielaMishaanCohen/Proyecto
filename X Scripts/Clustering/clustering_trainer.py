"""
Archivo: 5_Scripts/clustering_trainer.py

Fine-tuning para clustering de noticias

DECISIONES TÉCNICAS CLAVE:

1. MODELO BASE: sentence-transformers/all-MiniLM-L6-v2
   ¿Por qué este modelo?
   - Optimizado para generar embeddings semánticos
   - Pequeño (22M parámetros) pero muy efectivo
   - Pre-entrenado con Sentence-BERT para similitud semántica
   - Rápido: Procesa texto 5x más rápido que BERT-base
   
   Alternativas consideradas:
   - BERT-base: Más grande, no optimizado para embeddings
   - USE (Universal Sentence Encoder): Menos flexible para fine-tuning
   - RoBERTa: Excelente pero requiere más recursos

2. ESTRATEGIA DE FINE-TUNING: Contrastive Learning
   ¿Por qué contrastive learning?
   - Aprende a juntar noticias similares (mismo tema)
   - Aprende a separar noticias diferentes (temas distintos)
   - No requiere etiquetas manuales de clusters
   - Genera representaciones más discriminativas
   
   Técnica específica: Multiple Negatives Ranking Loss
   - Trata cada batch como un conjunto de pares
   - Maximiza similitud de pares positivos
   - Minimiza similitud con todos los negativos del batch

3. ALGORITMO DE CLUSTERING: HDBSCAN
   ¿Por qué HDBSCAN y no K-Means?
   - No requiere especificar número de clusters de antemano
   - Maneja clusters de diferentes densidades
   - Identifica outliers (noticias únicas)
   - Más robusto que DBSCAN
   
   K-Means requiere K predefinido - difícil con noticias dinámicas

4. MÉTRICAS: Silhouette Score, Davies-Bouldin Index
   ¿Por qué estas métricas?
   - Silhouette: Mide compactness y separación de clusters
   - Davies-Bouldin: Ratio de dispersión intra/inter cluster
   - No requieren ground truth (clusters son unsupervised)

5. DATA AUGMENTATION: Back-translation para augmentation
   ¿Por qué?
   - Genera variaciones semánticamente similares
   - Mejora robustez del modelo
   - Simula diferentes estilos de escritura de mismo tema
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from collections import Counter

class ClusteringTrainer:
    """
    Fine-tuning de modelo para clustering de noticias
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "2_Models/clustering"
    ):
        """
        Inicializa el clustering trainer.
        
        Args:
            model_name: Modelo base de sentence-transformers
            output_dir: Directorio para guardar modelo
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando device: {self.device}")
        
        # Cargar modelo
        print(f"\nCargando modelo: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # Configuración
        self.config = {
            'batch_size': 16,
            'num_epochs': 3,
            'learning_rate': 2e-5,
            'warmup_steps': 100,
            'evaluation_steps': 500,
            'max_seq_length': 256,  # Más corto que summarization
            'min_cluster_size': 5,  # Para HDBSCAN
            'min_samples': 3,  # Para HDBSCAN
        }
        
        self._log_config()
    
    def _log_config(self):
        """Log configuración"""
        print("\n" + "="*60)
        print("CONFIGURACIÓN DE CLUSTERING")
        print("="*60)
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Carga datos para clustering"""
        df = pd.read_csv(csv_path)
        print(f"Cargados {len(df)} documentos para clustering")
        return df
    
    def create_training_pairs(self, df: pd.DataFrame) -> List[InputExample]:
        """
        Crea pares de entrenamiento para contrastive learning.
        
        ESTRATEGIA:
        - Noticias de la misma fuente → pares positivos (similar)
        - Asumimos que fuentes cubren temas consistentes
        - El modelo aprenderá a juntar noticias temáticamente similares
        
        LIMITACIÓN:
        - No es perfecto (fuentes cubren múltiples temas)
        - Mejor sería: clustering previo + validación manual
        - Para este proyecto es un starting point razonable
        """
        training_examples = []
        
        # Agrupar por fuente
        source_groups = df.groupby('source')['text'].apply(list).to_dict()
        
        print(f"\nCreando pares de entrenamiento...")
        print(f"Fuentes únicas: {len(source_groups)}")
        
        for source, texts in source_groups.items():
            # Solo usar fuentes con suficientes artículos
            if len(texts) < 3:
                continue
            
            # Crear pares dentro de la misma fuente
            for i in range(len(texts)):
                for j in range(i+1, min(i+4, len(texts))):  # Max 3 pares por texto
                    # Score alto = similares (misma fuente)
                    example = InputExample(
                        texts=[texts[i], texts[j]],
                        label=0.9  # Alta similitud
                    )
                    training_examples.append(example)
        
        print(f"✓ Creados {len(training_examples)} pares de entrenamiento")
        return training_examples
    
    def fine_tune(self, training_examples: List[InputExample]):
        """
        Fine-tune con contrastive learning.
        
        LOSS FUNCTION: MultipleNegativesRankingLoss
        ¿Cómo funciona?
        - Batch de N pares (anchor, positive)
        - Anchor: Primera frase
        - Positive: Su par semánticamente similar
        - Negatives: Todos los otros positives del batch
        - Objetivo: Maximizar similitud (anchor, positive)
        -          Minimizar similitud (anchor, negatives)
        """
        print("\n" + "="*60)
        print("INICIANDO FINE-TUNING DE CLUSTERING")
        print("="*60)
        
        # DataLoader
        train_dataloader = DataLoader(
            training_examples,
            shuffle=True,
            batch_size=self.config['batch_size']
        )
        
        # Loss function
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Calcular warmup steps
        num_train_steps = len(train_dataloader) * self.config['num_epochs']
        warmup_steps = self.config['warmup_steps']
        
        print(f"\nPasos totales: {num_train_steps}")
        print(f"Warmup steps: {warmup_steps}")
        
        # Entrenar
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config['num_epochs'],
            warmup_steps=warmup_steps,
            output_path=str(self.output_dir / "fine_tuned_model"),
            show_progress_bar=True,
            evaluation_steps=self.config['evaluation_steps'],
        )
        
        print("\n✓ Fine-tuning completado!")
        
        # Guardar configuración
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Genera embeddings de documentos.
        
        ¿Qué son embeddings?
        - Representaciones vectoriales densas del texto
        - Capturan significado semántico
        - Textos similares → vectores cercanos en espacio
        - Dimensión típica: 384 (este modelo)
        """
        print("\n" + "="*60)
        print("GENERANDO EMBEDDINGS")
        print("="*60)
        
        texts = df['text'].tolist()
        
        # Generar embeddings en batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.config['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ Embeddings generados: {embeddings.shape}")
        
        # Guardar
        np.save(self.output_dir / "embeddings.npy", embeddings)
        
        return embeddings
    
    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Aplica HDBSCAN para clustering.
        
        PARÁMETROS CLAVE:
        - min_cluster_size: Mínimo de puntos para formar cluster
        - min_samples: Conservatividad (más alto = menos outliers)
        - metric: euclidean para embeddings normalizados
        
        HDBSCAN asigna -1 a outliers (noticias únicas)
        """
        print("\n" + "="*60)
        print("EJECUTANDO CLUSTERING")
        print("="*60)
        
        clusterer = HDBSCAN(
            min_cluster_size=self.config['min_cluster_size'],
            min_samples=self.config['min_samples'],
            metric='euclidean',
            cluster_selection_method='eom'  # Excess of Mass
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Estadísticas
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"\n✓ Clusters encontrados: {n_clusters}")
        print(f"✓ Puntos outlier: {n_noise}")
        
        # Distribución
        cluster_counts = Counter(cluster_labels)
        print(f"\nDistribución de clusters:")
        for cluster_id, count in sorted(cluster_counts.items()):
            if cluster_id != -1:
                print(f"  Cluster {cluster_id}: {count} documentos")
        
        return cluster_labels
    
    def evaluate_clustering(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Evalúa calidad del clustering.
        
        MÉTRICAS:
        
        1. Silhouette Score [-1, 1]
           - Mide qué tan bien están separados los clusters
           - 1: Perfecto
           - 0: Clusters overlapping
           - -1: Mal asignados
           
        2. Davies-Bouldin Index [0, ∞]
           - Ratio de similitud intra-cluster vs inter-cluster
           - 0: Perfecto
           - Más bajo = mejor
        
        3. Inertia (optional con K-Means)
           - Suma de distancias al centroid
           - Más bajo = más compacto
        """
        print("\n" + "="*60)
        print("EVALUANDO CLUSTERING")
        print("="*60)
        
        # Filtrar outliers para métricas
        mask = labels != -1
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        
        if len(set(filtered_labels)) < 2:
            print("⚠ Muy pocos clusters para calcular métricas")
            return {}
        
        # Silhouette Score
        silhouette = silhouette_score(filtered_embeddings, filtered_labels)
        print(f"\nSilhouette Score: {silhouette:.4f}")
        print("  (Rango: -1 a 1, más alto = mejor)")
        
        # Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
        print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
        print("  (Rango: 0 a ∞, más bajo = mejor)")
        
        metrics = {
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'n_clusters': len(set(filtered_labels)),
            'n_outliers': int((labels == -1).sum()),
            'n_samples': len(labels)
        }
        
        # Guardar métricas
        with open(self.output_dir / "clustering_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def visualize_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame
    ):
        """
        Visualiza clusters en 2D usando PCA.
        
        ¿Por qué PCA?
        - Reduce de 384D a 2D preservando varianza
        - Permite visualización intuitiva
        - Primera PC captura máxima variabilidad
        
        NOTA: Visualización es aproximación (pérdida de info)
        """
        print("\n" + "="*60)
        print("CREANDO VISUALIZACIONES")
        print("="*60)
        
        # Reducir dimensionalidad
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        # Colores para clusters
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Outliers en negro
                color = 'black'
                marker = 'x'
                label_text = 'Outliers'
            else:
                marker = 'o'
                label_text = f'Cluster {label}'
            
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=label_text,
                marker=marker,
                alpha=0.6,
                s=50
            )
        
        plt.xlabel('Primera Componente Principal')
        plt.ylabel('Segunda Componente Principal')
        plt.title('Clustering de Noticias (PCA 2D)', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Guardar
        plt.savefig(self.output_dir / "clusters_visualization.png", dpi=300)
        print(f"✓ Visualización guardada")
        
        # Análisis de clusters
        self._analyze_clusters(df, labels)
    
    def _analyze_clusters(self, df: pd.DataFrame, labels: np.ndarray):
        """Analiza contenido de cada cluster"""
        df_with_labels = df.copy()
        df_with_labels['cluster'] = labels
        
        print("\n" + "="*60)
        print("ANÁLISIS DE CLUSTERS")
        print("="*60)
        
        analysis = []
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            
            cluster_docs = df_with_labels[df_with_labels['cluster'] == cluster_id]
            
            # Títulos ejemplo
            sample_titles = cluster_docs['title'].head(3).tolist()
            
            # Fuentes principales
            top_sources = cluster_docs['source'].value_counts().head(3)
            
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_docs),
                'sample_titles': sample_titles,
                'top_sources': top_sources.to_dict()
            }
            
            analysis.append(cluster_info)
            
            print(f"\nCluster {cluster_id} ({len(cluster_docs)} docs):")
            print(f"  Títulos ejemplo:")
            for title in sample_titles:
                print(f"    - {title[:80]}...")
            print(f"  Fuentes principales: {', '.join(top_sources.index.tolist())}")
        
        # Guardar análisis
        with open(self.output_dir / "cluster_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)


def main():
    """Pipeline completo de clustering"""
    
    # Inicializar trainer
    trainer = ClusteringTrainer()
    
    # Cargar datos
    df = trainer.load_data("1_Data/processed/clustering_data.csv")
    
    # Crear pares para fine-tuning
    training_pairs = trainer.create_training_pairs(df)
    
    # Fine-tune
    trainer.fine_tune(training_pairs)
    
    # Cargar modelo fine-tuned
    trainer.model = SentenceTransformer(
        str(trainer.output_dir / "fine_tuned_model")
    )
    
    # Generar embeddings
    embeddings = trainer.generate_embeddings(df)
    
    # Clustering
    labels = trainer.cluster_embeddings(embeddings)
    
    # Evaluar
    trainer.evaluate_clustering(embeddings, labels)
    
    # Visualizar
    trainer.visualize_clusters(embeddings, labels, df)
    
    print("\n✓✓✓ Pipeline de clustering completado! ✓✓✓")


if __name__ == "__main__":
    main()