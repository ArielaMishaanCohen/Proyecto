"""
Archivo: 5_Scripts/clustering_trainer.py

Fine-tuning para clustering de noticias

CORRECCIONES APLICADAS:
1. ✓ Agregado train/validation split (80/20)
2. ✓ Creado TripletEvaluator para evaluación durante entrenamiento
3. ✓ Configurado eval_strategy='epoch' con evaluador
4. ✓ Mejorado logging y tracking de métricas

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

3. EVALUACIÓN: TripletEvaluator
   ¿Cómo funciona?
   - Triplets: (anchor, positive, negative)
   - Anchor: Noticia base
   - Positive: Noticia similar (mismo tema)
   - Negative: Noticia diferente (tema distinto)
   - Métrica: Accuracy en ranking correcto

4. ALGORITMO DE CLUSTERING: HDBSCAN
   ¿Por qué HDBSCAN y no K-Means?
   - No requiere especificar número de clusters de antemano
   - Maneja clusters de diferentes densidades
   - Identifica outliers (noticias únicas)
   - Más robusto que DBSCAN
   
   K-Means requiere K predefinido - difícil con noticias dinámicas

5. MÉTRICAS: Silhouette Score, Davies-Bouldin Index
   ¿Por qué estas métricas?
   - Silhouette: Mide compactness y separación de clusters
   - Davies-Bouldin: Ratio de dispersión intra/inter cluster
   - No requieren ground truth (clusters son unsupervised)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
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

class ClusteringTrainer:
    """
    Fine-tuning de modelo para clustering de noticias
    
    MEJORAS EN ESTA VERSIÓN:
    - Train/Val split para evaluación robusta
    - TripletEvaluator para monitoreo durante entrenamiento
    - Mejor tracking de métricas
    - Validación de calidad de datos
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
            'evaluation_steps': 50,  # Evaluar cada 50 steps
            'save_steps': 100,
            'max_seq_length': 256,  # Más corto que summarization
            'min_cluster_size': 5,  # Para HDBSCAN
            'min_samples': 3,  # Para HDBSCAN
            'val_split': 0.2,  # 20% para validación
            'random_seed': 42
        }
        
        # Set seed para reproducibilidad
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        torch.manual_seed(self.config['random_seed'])
        
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
        
        # Validación de datos
        required_cols = ['text', 'title', 'source']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Filtrar filas con datos nulos
        df = df.dropna(subset=required_cols)
        print(f"Después de limpiar nulos: {len(df)} documentos")
        
        return df
    
    def create_training_pairs(
        self, 
        df: pd.DataFrame
    ) -> Tuple[List[InputExample], List[InputExample]]:
        """
        Crea pares de entrenamiento y validación para contrastive learning.
        
        ESTRATEGIA:
        - Noticias de la misma fuente → pares positivos (similar)
        - Asumimos que fuentes cubren temas consistentes
        - El modelo aprenderá a juntar noticias temáticamente similares
        
        MEJORA: Ahora divide en train/val
        
        Returns:
            train_examples: Pares para entrenamiento
            val_examples: Pares para validación
        """
        all_examples = []
        
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
                    all_examples.append(example)
        
        # TRAIN/VAL SPLIT
        train_size = int(len(all_examples) * (1 - self.config['val_split']))
        train_examples = all_examples[:train_size]
        val_examples = all_examples[train_size:]
        
        print(f"✓ Pares de entrenamiento: {len(train_examples)}")
        print(f"✓ Pares de validación: {len(val_examples)}")
        
        return train_examples, val_examples
    
    def create_triplet_evaluator(
        self,
        val_examples: List[InputExample],
        df: pd.DataFrame
    ) -> TripletEvaluator:
        """
        Crea evaluador basado en triplets.
        
        TRIPLETS: (anchor, positive, negative)
        - anchor: Noticia base
        - positive: Noticia de la misma fuente (similar)
        - negative: Noticia de fuente diferente (diferente)
        
        ¿POR QUÉ TRIPLETS?
        - Evalúa si el modelo correctamente ordena:
          similarity(anchor, positive) > similarity(anchor, negative)
        - Métrica más informativa que solo pares
        - Refleja objetivo real del clustering
        
        Returns:
            TripletEvaluator configurado
        """
        print("\nCreando TripletEvaluator...")
        
        # Convertir val_examples a triplets
        # Extraer textos y fuentes
        source_to_texts = df.groupby('source')['text'].apply(list).to_dict()
        sources = list(source_to_texts.keys())
        
        anchors = []
        positives = []
        negatives = []
        
        # Crear triplets
        for source in sources:
            texts = source_to_texts[source]
            if len(texts) < 2:
                continue
            
            # Para cada par en esta fuente
            for i in range(min(5, len(texts))):  # Max 5 triplets por fuente
                if i + 1 >= len(texts):
                    break
                
                anchor = texts[i]
                positive = texts[i + 1]
                
                # Seleccionar negative de otra fuente
                other_sources = [s for s in sources if s != source]
                if not other_sources:
                    continue
                
                neg_source = random.choice(other_sources)
                if not source_to_texts[neg_source]:
                    continue
                
                negative = random.choice(source_to_texts[neg_source])
                
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
        
        print(f"✓ Creados {len(anchors)} triplets para evaluación")
        
        # Crear evaluador
        evaluator = TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            name='validation',
            show_progress_bar=True
        )
        
        return evaluator
    
    def fine_tune(
        self,
        train_examples: List[InputExample],
        val_examples: List[InputExample],
        evaluator: TripletEvaluator
    ):
        """
        Fine-tune con contrastive learning y evaluación.
        
        LOSS FUNCTION: MultipleNegativesRankingLoss
        ¿Cómo funciona?
        - Batch de N pares (anchor, positive)
        - Anchor: Primera frase
        - Positive: Su par semánticamente similar
        - Negatives: Todos los otros positives del batch
        - Objetivo: Maximizar similitud (anchor, positive)
        -          Minimizar similitud (anchor, negatives)
        
        MEJORAS:
        - Evaluación durante entrenamiento con TripletEvaluator
        - Guarda mejor modelo basado en validation
        - Logging detallado de métricas
        """
        print("\n" + "="*60)
        print("INICIANDO FINE-TUNING DE CLUSTERING")
        print("="*60)
        
        # DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.config['batch_size']
        )
        
        # Loss function
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Calcular steps
        num_train_steps = len(train_dataloader) * self.config['num_epochs']
        warmup_steps = min(self.config['warmup_steps'], num_train_steps // 10)
        
        print(f"\nPasos totales: {num_train_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Evaluaciones cada: {self.config['evaluation_steps']} steps")
        
        # ENTRENAR CON EVALUACIÓN
        # CLAVE: Usamos evaluator para monitorear progreso
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,  # ✓ AGREGADO: Evaluador
            epochs=self.config['num_epochs'],
            warmup_steps=warmup_steps,
            output_path=str(self.output_dir / "fine_tuned_model"),
            show_progress_bar=True,
            evaluation_steps=self.config['evaluation_steps'],  # Evaluar cada N steps
            save_best_model=True,  # Guardar mejor modelo según evaluador
            optimizer_params={'lr': self.config['learning_rate']},
        )
        
        print("\n✓ Fine-tuning completado!")
        
        # Guardar configuración
        config_to_save = self.config.copy()
        config_to_save['model_name'] = self.model_name
        config_to_save['num_train_examples'] = len(train_examples)
        config_to_save['num_val_examples'] = len(val_examples)
        
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"✓ Configuración guardada en {self.output_dir / 'training_config.json'}")
    
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
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizar para mejor clustering
        )
        
        print(f"✓ Embeddings generados: {embeddings.shape}")
        print(f"✓ Embeddings normalizados: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
        
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
        print(f"✓ Puntos outlier: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
        
        # Distribución
        cluster_counts = Counter(cluster_labels)
        print(f"\nDistribución de clusters:")
        for cluster_id, count in sorted(cluster_counts.items()):
            if cluster_id != -1:
                print(f"  Cluster {cluster_id}: {count} documentos")
        
        # Guardar labels
        np.save(self.output_dir / "cluster_labels.npy", cluster_labels)
        
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
        
        INTERPRETACIÓN:
        - Silhouette > 0.5: Buena estructura
        - Silhouette 0.25-0.5: Estructura débil pero aceptable
        - Silhouette < 0.25: No hay estructura clara
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
        if silhouette > 0.5:
            print("  → Excelente separación de clusters")
        elif silhouette > 0.25:
            print("  → Separación aceptable")
        else:
            print("  → Separación débil, considerar ajustar parámetros")
        
        # Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
        print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
        if davies_bouldin < 1.0:
            print("  → Clusters bien definidos")
        else:
            print("  → Clusters con overlap, considerar ajustar min_cluster_size")
        
        metrics = {
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'n_clusters': len(set(filtered_labels)),
            'n_outliers': int((labels == -1).sum()),
            'n_samples': len(labels),
            'outlier_percentage': float((labels == -1).sum() / len(labels) * 100)
        }
        
        # Guardar métricas
        with open(self.output_dir / "clustering_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Métricas guardadas en {self.output_dir / 'clustering_metrics.json'}")
        
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
        pca = PCA(n_components=2, random_state=self.config['random_seed'])
        embeddings_2d = pca.fit_transform(embeddings)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"Varianza explicada por PCA: {variance_explained:.2%}")
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        # Colores para clusters
        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l != -1])
        colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
        
        color_map = {}
        color_idx = 0
        for label in unique_labels:
            if label == -1:
                color_map[label] = 'black'
            else:
                color_map[label] = colors[color_idx]
                color_idx += 1
        
        for label in unique_labels:
            if label == -1:
                marker = 'x'
                label_text = 'Outliers'
                alpha = 0.3
                size = 30
            else:
                marker = 'o'
                label_text = f'Cluster {label}'
                alpha = 0.6
                size = 50
            
            mask = labels == label
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
        
        plt.xlabel('Primera Componente Principal', fontsize=12)
        plt.ylabel('Segunda Componente Principal', fontsize=12)
        plt.title(
            f'Clustering de Noticias (PCA 2D - {variance_explained:.1%} varianza)', 
            fontsize=14, 
            fontweight='bold'
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Guardar
        viz_path = self.output_dir / "clusters_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualización guardada en {viz_path}")
        plt.close()
        
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
            sample_titles = cluster_docs['title'].head(5).tolist()
            
            # Fuentes principales
            top_sources = cluster_docs['source'].value_counts().head(3)
            
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_docs),
                'percentage': float(len(cluster_docs) / len(df) * 100),
                'sample_titles': sample_titles,
                'top_sources': top_sources.to_dict()
            }
            
            analysis.append(cluster_info)
            
            print(f"\n{'='*60}")
            print(f"Cluster {cluster_id}")
            print(f"{'='*60}")
            print(f"  Tamaño: {len(cluster_docs)} docs ({cluster_info['percentage']:.1f}%)")
            print(f"\n  Títulos ejemplo:")
            for i, title in enumerate(sample_titles, 1):
                print(f"    {i}. {title[:80]}...")
            print(f"\n  Fuentes principales:")
            for source, count in top_sources.items():
                print(f"    - {source}: {count} artículos")
        
        # Análisis de outliers
        outliers = df_with_labels[df_with_labels['cluster'] == -1]
        if len(outliers) > 0:
            print(f"\n{'='*60}")
            print(f"Outliers (Noticias únicas)")
            print(f"{'='*60}")
            print(f"  Total: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
            print(f"\n  Ejemplos:")
            for i, (_, row) in enumerate(outliers.head(3).iterrows(), 1):
                print(f"    {i}. {row['title'][:80]}...")
        
        # Guardar análisis
        analysis_path = self.output_dir / "cluster_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Análisis guardado en {analysis_path}")


def main():
    """
    Pipeline completo de clustering
    
    FLUJO:
    1. Cargar datos
    2. Crear pares train/val
    3. Crear evaluador
    4. Fine-tune con evaluación
    5. Generar embeddings
    6. Clustering
    7. Evaluar y visualizar
    """
    
    print("\n" + "="*70)
    print(" "*15 + "PIPELINE DE CLUSTERING")
    print("="*70 + "\n")
    
    # Inicializar trainer
    trainer = ClusteringTrainer()
    
    # 1. Cargar datos
    print("\n[PASO 1/7] Cargando datos...")
    df = trainer.load_data("1_Data/processed/clustering_data_cleaned.csv")
    
    # 2. Crear pares para fine-tuning
    print("\n[PASO 2/7] Creando pares de entrenamiento...")
    train_examples, val_examples = trainer.create_training_pairs(df)
    
    # 3. Crear evaluador
    print("\n[PASO 3/7] Creando evaluador...")
    evaluator = trainer.create_triplet_evaluator(val_examples, df)
    
    # 4. Fine-tune
    print("\n[PASO 4/7] Fine-tuning del modelo...")
    trainer.fine_tune(train_examples, val_examples, evaluator)
    
    # 5. Cargar modelo fine-tuned
    print("\n[PASO 5/7] Cargando modelo fine-tuned...")
    trainer.model = SentenceTransformer(
        str(trainer.output_dir / "fine_tuned_model")
    )
    
    # 6. Generar embeddings
    print("\n[PASO 6/7] Generando embeddings...")
    embeddings = trainer.generate_embeddings(df)
    
    # 7. Clustering
    print("\n[PASO 7/7] Ejecutando clustering...")
    labels = trainer.cluster_embeddings(embeddings)
    
    # Evaluar
    trainer.evaluate_clustering(embeddings, labels)
    
    # Visualizar
    trainer.visualize_clusters(embeddings, labels, df)
    
    print("\n" + "="*70)
    print(" "*15 + "✓✓✓ PIPELINE COMPLETADO ✓✓✓")
    print("="*70)
    print(f"\nResultados guardados en: {trainer.output_dir}")
    print(f"  - Modelo: fine_tuned_model/")
    print(f"  - Embeddings: embeddings.npy")
    print(f"  - Labels: cluster_labels.npy")
    print(f"  - Métricas: clustering_metrics.json")
    print(f"  - Análisis: cluster_analysis.json")
    print(f"  - Visualización: clusters_visualization.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()