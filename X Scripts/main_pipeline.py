"""
Archivo: main_pipeline.py

Script principal que ejecuta todo el proyecto de principio a fin

ESTRUCTURA DEL PROYECTO:
1. Setup y configuración
2. Procesamiento de datos
3. Fine-tuning de summarization
4. Fine-tuning de clustering
5. Fine-tuning de sentiment analysis
6. Visualizaciones y comparaciones
7. Reporte final
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configurar paths
sys.path.append('5_Scripts')

# Importar módulos del proyecto
from data_processor import NewsDataProcessor
from summarization_trainer import SummarizationTrainer
from clustering_trainer import ClusteringTrainer
from sentiment_trainer import SentimentTrainer


class ProjectPipeline:
    """
    Orquesta el pipeline completo del proyecto
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        self.output_dir = Path("3_Results")
        self.output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print(" "*20 + "PROYECTO: FINE-TUNING DE LLMs PARA NOTICIAS")
        print("="*80)
        print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
    
    def run_step(self, step_name: str, step_function):
        """
        Ejecuta un paso del pipeline con tracking de tiempo
        """
        print("\n" + "█"*80)
        print(f"█  STEP: {step_name}")
        print("█"*80 + "\n")
        
        step_start = time.time()
        
        try:
            result = step_function()
            step_time = time.time() - step_start
            
            print(f"\n✓ {step_name} completado en {step_time:.2f}s")
            
            self.results[step_name] = {
                'status': 'success',
                'time': step_time,
                'result': result
            }
            
            return result
            
        except Exception as e:
            step_time = time.time() - step_start
            print(f"\n✗ Error en {step_name}: {str(e)}")
            
            self.results[step_name] = {
                'status': 'error',
                'time': step_time,
                'error': str(e)
            }
            
            raise
    
    def step1_process_data(self):
        """STEP 1: Procesar datos raw"""
        processor = NewsDataProcessor()
        sum_df, clust_df, sent_df = processor.process_all()
        
        return {
            'summarization_samples': len(sum_df),
            'clustering_samples': len(clust_df),
            'sentiment_samples': len(sent_df)
        }
    
    def step2_train_summarization(self):
        """STEP 2: Fine-tune modelo de summarization"""
        trainer = SummarizationTrainer()
        
        # Cargar datos
        train_df, val_df, test_df = trainer.load_data(
            "1_Data/processed/summarization_data.csv"
        )
        
        # Entrenar
        trained_model = trainer.train(train_df, val_df)
        
        # Evaluar
        metrics = trainer.evaluate(trained_model, test_df)
        
        # Generar muestras
        trainer.generate_samples(test_df, n_samples=10)
        
        return metrics
    
    def step3_train_clustering(self):
        """STEP 3: Fine-tune modelo de clustering"""
        trainer = ClusteringTrainer()
        
        # Cargar datos
        df = trainer.load_data("1_Data/processed/clustering_data.csv")
        
        # Crear pares
        training_pairs = trainer.create_training_pairs(df)
        
        # Fine-tune
        trainer.fine_tune(training_pairs)
        
        # Cargar modelo fine-tuned
        from sentence_transformers import SentenceTransformer
        trainer.model = SentenceTransformer(
            str(trainer.output_dir / "fine_tuned_model")
        )
        
        # Generar embeddings
        embeddings = trainer.generate_embeddings(df)
        
        # Clustering
        labels = trainer.cluster_embeddings(embeddings)
        
        # Evaluar
        metrics = trainer.evaluate_clustering(embeddings, labels)
        
        # Visualizar
        trainer.visualize_clusters(embeddings, labels, df)
        
        return metrics
    
    def step4_train_sentiment(self):
        """STEP 4: Fine-tune modelo de sentiment"""
        trainer = SentimentTrainer()
        
        # Cargar datos
        train_df, val_df, test_df = trainer.load_data(
            "1_Data/processed/sentiment_data.csv"
        )
        
        # Class weights
        class_weights = trainer.compute_class_weights(train_df)
        
        # Entrenar
        trained_model = trainer.train(train_df, val_df, class_weights)
        
        # Evaluar
        metrics = trainer.evaluate(trained_model, test_df)
        
        # Analizar errores
        trainer.analyze_errors(trained_model, test_df, n_examples=15)
        
        return metrics
    
    def step5_create_visualizations(self):
        """STEP 5: Crear visualizaciones comparativas"""
        print("\nCreando visualizaciones finales...")
        
        # Comparar métricas de todos los modelos
        self._plot_model_comparison()
        
        # Timeline de entrenamiento
        self._plot_training_timeline()
        
        # Recursos utilizados
        self._plot_resources()
        
        return {'visualizations_created': True}
    
    def _plot_model_comparison(self):
        """
        Compara performance de los 3 modelos.
        
        Desafío: Métricas diferentes (ROUGE vs Silhouette vs F1)
        Solución: Gráficos separados pero formato unificado
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Summarization (ROUGE scores)
        sum_metrics = json.load(open("2_Models/summarization/test_metrics.json"))
        rouge_scores = {
            'ROUGE-1': sum_metrics.get('test_rouge1', 0),
            'ROUGE-2': sum_metrics.get('test_rouge2', 0),
            'ROUGE-L': sum_metrics.get('test_rougeL', 0)
        }
        
        axes[0].bar(rouge_scores.keys(), rouge_scores.values(), color='skyblue')
        axes[0].set_title('Summarization Model\n(ROUGE Scores)', fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 2. Clustering (Silhouette & Davies-Bouldin)
        clust_metrics = json.load(open("2_Models/clustering/clustering_metrics.json"))
        
        # Normalizar Davies-Bouldin (invertir y escalar)
        db_normalized = 1 / (1 + clust_metrics.get('davies_bouldin_index', 1))
        
        cluster_scores = {
            'Silhouette\nScore': clust_metrics.get('silhouette_score', 0),
            'DB Index\n(normalized)': db_normalized
        }
        
        axes[1].bar(cluster_scores.keys(), cluster_scores.values(), color='lightcoral')
        axes[1].set_title('Clustering Model\n(Quality Metrics)', fontweight='bold')
        axes[1].set_ylabel('Score')
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis='y', alpha=0.3)
        
        # 3. Sentiment (F1 scores)
        sent_report = json.load(open("2_Models/sentiment/classification_report.json"))
        
        f1_scores = {
            'Negative': sent_report['negative']['f1-score'],
            'Neutral': sent_report['neutral']['f1-score'],
            'Positive': sent_report['positive']['f1-score'],
            'Macro Avg': sent_report['macro avg']['f1-score']
        }
        
        axes[2].bar(f1_scores.keys(), f1_scores.values(), color='lightgreen')
        axes[2].set_title('Sentiment Model\n(F1 Scores)', fontweight='bold')
        axes[2].set_ylabel('Score')
        axes[2].set_ylim(0, 1)
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        print("✓ Comparación de modelos guardada")
    
    def _plot_training_timeline(self):
        """Visualiza timeline de ejecución"""
        steps = []
        times = []
        colors = []
        
        color_map = {
            'success': 'green',
            'error': 'red',
            'warning': 'orange'
        }
        
        for step_name, step_data in self.results.items():
            steps.append(step_name.replace('_', ' ').title())
            times.append(step_data['time'])
            colors.append(color_map.get(step_data['status'], 'gray'))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(steps, times, color=colors, alpha=0.7)
        
        # Añadir valores
        for bar, time_val in zip(bars, times):
            width = bar.get_width()
            ax.text(
                width, bar.get_y() + bar.get_height()/2,
                f' {time_val:.1f}s',
                va='center',
                fontweight='bold'
            )
        
        ax.set_xlabel('Tiempo (segundos)', fontsize=12)
        ax.set_title('Timeline de Ejecución del Proyecto', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_timeline.png", dpi=300, bbox_inches='tight')
        print("✓ Timeline guardado")
    
    def _plot_resources(self):
        """Visualiza recursos de cada modelo"""
        # Información de modelos
        models_info = [
            {
                'name': 'BART-base\n(Summarization)',
                'parameters': 139.4,  # Millones
                'size_mb': 558,
                'training_time': self.results.get('Train Summarization', {}).get('time', 0) / 60
            },
            {
                'name': 'MiniLM-L6\n(Clustering)',
                'parameters': 22.7,
                'size_mb': 91,
                'training_time': self.results.get('Train Clustering', {}).get('time', 0) / 60
            },
            {
                'name': 'DistilBERT\n(Sentiment)',
                'parameters': 66.4,
                'size_mb': 268,
                'training_time': self.results.get('Train Sentiment', {}).get('time', 0) / 60
            }
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        names = [m['name'] for m in models_info]
        
        # Parámetros
        params = [m['parameters'] for m in models_info]
        axes[0].bar(names, params, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0].set_ylabel('Millones de parámetros')
        axes[0].set_title('Tamaño del Modelo', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Tamaño en disco
        sizes = [m['size_mb'] for m in models_info]
        axes[1].bar(names, sizes, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1].set_ylabel('MB')
        axes[1].set_title('Espacio en Disco', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Tiempo de entrenamiento
        times = [m['training_time'] for m in models_info]
        axes[2].bar(names, times, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[2].set_ylabel('Minutos')
        axes[2].set_title('Tiempo de Entrenamiento', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_resources.png", dpi=300, bbox_inches='tight')
        print("✓ Recursos guardados")
    
    def step6_generate_report(self):
        """STEP 6: Generar reporte final"""
        report = {
            'project_name': 'Fine-tuning de LLMs para Análisis de Noticias',
            'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': time.time() - self.start_time,
            'steps': self.results,
            'models': {
                'summarization': {
                    'model': 'facebook/bart-base',
                    'task': 'Summarization',
                    'metrics': json.load(open("2_Models/summarization/test_metrics.json"))
                },
                'clustering': {
                    'model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'task': 'Clustering',
                    'metrics': json.load(open("2_Models/clustering/clustering_metrics.json"))
                },
                'sentiment': {
                    'model': 'distilbert-base-uncased',
                    'task': 'Sentiment Analysis',
                    'metrics': json.load(open("2_Models/sentiment/test_metrics.json"))
                }
            }
        }
        
        # Guardar reporte
        with open(self.output_dir / "final_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Imprimir resumen
        self._print_final_summary(report)
        
        return report
    
    def _print_final_summary(self, report: dict):
        """Imprime resumen ejecutivo"""
        print("\n" + "="*80)
        print(" "*25 + "RESUMEN EJECUTIVO")
        print("="*80)
        
        print(f"\n📊 Tiempo total: {report['total_time_seconds']/60:.2f} minutos")
        
        print("\n" + "-"*80)
        print("MODELO 1: SUMMARIZATION")
        print("-"*80)
        sum_metrics = report['models']['summarization']['metrics']
        print(f"  Modelo base: {report['models']['summarization']['model']}")
        print(f"  ROUGE-1: {sum_metrics.get('test_rouge1', 0):.4f}")
        print(f"  ROUGE-2: {sum_metrics.get('test_rouge2', 0):.4f}")
        print(f"  ROUGE-L: {sum_metrics.get('test_rougeL', 0):.4f}")
        print("\n  ✓ Modelo fine-tuneado específicamente para noticias")
        print("  ✓ Aprende estilo conciso de headlines + descriptions")
        
        print("\n" + "-"*80)
        print("MODELO 2: CLUSTERING")
        print("-"*80)
        clust_metrics = report['models']['clustering']['metrics']
        print(f"  Modelo base: {report['models']['clustering']['model']}")
        print(f"  Silhouette Score: {clust_metrics.get('silhouette_score', 0):.4f}")
        print(f"  Davies-Bouldin Index: {clust_metrics.get('davies_bouldin_index', 0):.4f}")
        print(f"  Clusters encontrados: {clust_metrics.get('n_clusters', 0)}")
        print("\n  ✓ Fine-tuned con contrastive learning")
        print("  ✓ Agrupa noticias por similitud semántica temática")
        
        print("\n" + "-"*80)
        print("MODELO 3: SENTIMENT ANALYSIS")
        print("-"*80)
        sent_metrics = report['models']['sentiment']['metrics']
        print(f"  Modelo base: {report['models']['sentiment']['model']}")
        print(f"  Accuracy: {sent_metrics.get('test_accuracy', 0):.4f}")
        print(f"  F1-Macro: {sent_metrics.get('test_f1_macro', 0):.4f}")
        print(f"  F1-Weighted: {sent_metrics.get('test_f1_weighted', 0):.4f}")
        print("\n  ✓ Fine-tuned con class weighting para balance")
        print("  ✓ Detecta sentimiento positivo/neutral/negativo")
        
        print("\n" + "="*80)
        print("ARCHIVOS GENERADOS:")
        print("="*80)
        print("  📁 2_Models/summarization/final_model/")
        print("  📁 2_Models/clustering/fine_tuned_model/")
        print("  📁 2_Models/sentiment/final_model/")
        print("  📊 3_Results/model_comparison.png")
        print("  📊 3_Results/training_timeline.png")
        print("  📊 3_Results/model_resources.png")
        print("  📄 3_Results/final_report.json")
        
        print("\n" + "="*80)
        print("✓✓✓ PROYECTO COMPLETADO EXITOSAMENTE ✓✓✓")
        print("="*80 + "\n")
    
    def run(self):
        """Ejecuta el pipeline completo"""
        try:
            # STEP 1: Procesar datos
            self.run_step("Process Data", self.step1_process_data)
            
            # STEP 2: Summarization
            self.run_step("Train Summarization", self.step2_train_summarization)
            
            # STEP 3: Clustering
            self.run_step("Train Clustering", self.step3_train_clustering)
            
            # STEP 4: Sentiment
            self.run_step("Train Sentiment", self.step4_train_sentiment)
            
            # STEP 5: Visualizaciones
            self.run_step("Create Visualizations", self.step5_create_visualizations)
            
            # STEP 6: Reporte final
            self.run_step("Generate Report", self.step6_generate_report)
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR CRÍTICO: {str(e)}")
            print(f"{'='*80}\n")
            raise


def main():
    """Punto de entrada principal"""
    pipeline = ProjectPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()