"""
HERRAMIENTA DE DIAGNÓSTICO PARA CLUSTERING
===========================================

Este script te ayuda a entender POR QUÉ el clustering no está funcionando
y QUÉ hacer para mejorarlo.

DIAGNÓSTICOS:
1. Calidad de los datos
2. Diversidad de temas
3. Longitud y contenido de textos
4. Distribución de fuentes
5. Embeddings quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

class ClusteringDiagnostic:
    """
    Herramienta de diagnóstico para identificar problemas
    """
    
    def __init__(self, data_path: str, output_dir: str = "3_Analysis/diagnostic"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.issues = []
        self.recommendations = []
    
    def run_full_diagnostic(self):
        """Ejecuta diagnóstico completo"""
        print("\n" + "="*70)
        print(" "*20 + "🔍 DIAGNÓSTICO DE CLUSTERING")
        print("="*70 + "\n")
        
        # Cargar datos
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Datos cargados: {len(self.df)} documentos\n")
        
        # Ejecutar análisis
        self.analyze_data_quality()
        self.analyze_content_diversity()
        self.analyze_text_characteristics()
        self.analyze_source_distribution()
        self.analyze_similarity_distribution()
        self.visualize_data_distribution()
        
        # Generar reporte
        self.generate_report()
        
        print("\n" + "="*70)
        print(" "*20 + "✅ DIAGNÓSTICO COMPLETADO")
        print("="*70)
        print(f"\nReporte guardado en: {self.output_dir / 'diagnostic_report.txt'}")
    
    def analyze_data_quality(self):
        """Analiza calidad básica de los datos"""
        print("📊 [1/6] Analizando calidad de datos...")
        
        # Valores nulos
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            self.issues.append(f"❌ Valores nulos encontrados: {null_counts.to_dict()}")
            self.recommendations.append("➡️ Eliminar o imputar valores nulos")
        else:
            print("  ✓ No hay valores nulos")
        
        # Duplicados
        duplicates = self.df.duplicated(subset=['title']).sum()
        if duplicates > 0:
            self.issues.append(f"❌ {duplicates} títulos duplicados ({duplicates/len(self.df)*100:.1f}%)")
            self.recommendations.append("➡️ Eliminar duplicados para evitar sesgo")
        else:
            print("  ✓ No hay duplicados")
        
        # Textos muy cortos
        short_texts = (self.df['text'].str.split().str.len() < 20).sum()
        if short_texts > len(self.df) * 0.1:
            self.issues.append(f"⚠️ {short_texts} textos muy cortos (< 20 palabras)")
            self.recommendations.append("➡️ Filtrar textos cortos o combinar con descripción")
        else:
            print("  ✓ Longitud de textos adecuada")
        
        print()
    
    def analyze_content_diversity(self):
        """Analiza diversidad de contenido"""
        print("🎨 [2/6] Analizando diversidad de contenido...")
        
        # TF-IDF para encontrar palabras más importantes
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(self.df['text'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Top palabras
        importance = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = importance.argsort()[-20:][::-1]
        top_words = [(feature_names[i], importance[i]) for i in top_indices]
        
        print(f"\n  📝 Top 10 términos más frecuentes:")
        for word, score in top_words[:10]:
            print(f"    • {word}: {score:.2f}")
        
        # Calcular diversidad
        unique_ratio = len(set(' '.join(self.df['text']).split())) / len(' '.join(self.df['text']).split())
        print(f"\n  📊 Ratio de palabras únicas: {unique_ratio:.3f}")
        
        if unique_ratio < 0.1:
            self.issues.append("❌ Vocabulario muy repetitivo (ratio < 0.1)")
            self.recommendations.append("➡️ Los documentos son muy similares, clustering será difícil")
        elif unique_ratio < 0.2:
            self.issues.append("⚠️ Vocabulario moderadamente repetitivo")
            self.recommendations.append("➡️ Considerar usar más features (bigrams, trigrams)")
        else:
            print("  ✓ Buena diversidad de vocabulario")
        
        print()
    
    def analyze_text_characteristics(self):
        """Analiza características de los textos"""
        print("📏 [3/6] Analizando características de textos...")
        
        # Longitudes
        text_lengths = self.df['text'].str.split().str.len()
        title_lengths = self.df['title'].str.split().str.len()
        
        print(f"\n  📝 Estadísticas de TEXTO:")
        print(f"    • Promedio: {text_lengths.mean():.1f} palabras")
        print(f"    • Mediana: {text_lengths.median():.1f} palabras")
        print(f"    • Min: {text_lengths.min()} palabras")
        print(f"    • Max: {text_lengths.max()} palabras")
        print(f"    • Std: {text_lengths.std():.1f}")
        
        print(f"\n  📰 Estadísticas de TÍTULO:")
        print(f"    • Promedio: {title_lengths.mean():.1f} palabras")
        print(f"    • Mediana: {title_lengths.median():.1f} palabras")
        
        # Variabilidad
        if text_lengths.std() / text_lengths.mean() > 1.0:
            self.issues.append("⚠️ Alta variabilidad en longitud de textos")
            self.recommendations.append("➡️ Considerar normalizar longitudes o usar técnicas adaptativas")
        
        # Textos demasiado cortos para clustering efectivo
        if text_lengths.mean() < 50:
            self.issues.append("❌ Textos muy cortos en promedio (< 50 palabras)")
            self.recommendations.append("➡️ Combinar título + descripción + contenido para más contexto")
        
        print()
    
    def analyze_source_distribution(self):
        """Analiza distribución de fuentes"""
        print("📡 [4/6] Analizando distribución de fuentes...")
        
        source_counts = self.df['source'].value_counts()
        
        print(f"\n  📊 Total de fuentes únicas: {len(source_counts)}")
        print(f"\n  🏆 Top 10 fuentes:")
        for source, count in source_counts.head(10).items():
            percentage = count / len(self.df) * 100
            print(f"    • {source}: {count} ({percentage:.1f}%)")
        
        # Concentración
        top_3_percentage = source_counts.head(3).sum() / len(self.df) * 100
        print(f"\n  📈 Top 3 fuentes representan: {top_3_percentage:.1f}% del total")
        
        if top_3_percentage > 50:
            self.issues.append(f"⚠️ Alta concentración en pocas fuentes ({top_3_percentage:.1f}%)")
            self.recommendations.append("➡️ Datos sesgados hacia pocas fuentes dificultan clustering diverso")
        
        # Fuentes con pocos artículos
        small_sources = (source_counts < 3).sum()
        if small_sources > len(source_counts) * 0.5:
            self.issues.append(f"⚠️ {small_sources} fuentes con < 3 artículos")
            self.recommendations.append("➡️ Muchas fuentes con pocos artículos generan ruido")
        
        print()
    
    def analyze_similarity_distribution(self):
        """Analiza distribución de similitudes"""
        print("🔗 [5/6] Analizando similitudes entre documentos...")
        
        # Sample para eficiencia
        sample_size = min(200, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        # TF-IDF similarity
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sample_df['text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Extraer similitudes (sin diagonal)
        n = similarity_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        similarities = similarity_matrix[mask]
        
        print(f"\n  📊 Estadísticas de similitud (muestra de {sample_size}):")
        print(f"    • Promedio: {similarities.mean():.3f}")
        print(f"    • Mediana: {np.median(similarities):.3f}")
        print(f"    • Min: {similarities.min():.3f}")
        print(f"    • Max: {similarities.max():.3f}")
        print(f"    • Std: {similarities.std():.3f}")
        
        # Distribución
        high_sim = (similarities > 0.5).sum() / len(similarities) * 100
        medium_sim = ((similarities > 0.2) & (similarities <= 0.5)).sum() / len(similarities) * 100
        low_sim = (similarities <= 0.2).sum() / len(similarities) * 100
        
        print(f"\n  📈 Distribución de similitud:")
        print(f"    • Alta (>0.5): {high_sim:.1f}%")
        print(f"    • Media (0.2-0.5): {medium_sim:.1f}%")
        print(f"    • Baja (<0.2): {low_sim:.1f}%")
        
        # Diagnóstico
        if similarities.mean() < 0.1:
            self.issues.append("❌ Similitud promedio muy baja (< 0.1)")
            self.recommendations.append("➡️ Documentos muy diferentes, difícil formar clusters coherentes")
            self.recommendations.append("➡️ Considerar: temas más específicos, mejor preprocesamiento")
        elif similarities.mean() > 0.5:
            self.issues.append("❌ Similitud promedio muy alta (> 0.5)")
            self.recommendations.append("➡️ Documentos muy similares, clustering no aporta mucho valor")
            self.recommendations.append("➡️ Considerar: subtopics más finos, usar más features")
        elif high_sim < 5:
            self.issues.append("⚠️ Muy pocos pares con alta similitud (< 5%)")
            self.recommendations.append("➡️ Pocos grupos naturales en los datos")
            self.recommendations.append("➡️ Clustering podría no ser la técnica adecuada")
        else:
            print("  ✓ Distribución de similitud adecuada para clustering")
        
        # Guardar matriz de similitud (muestra)
        np.save(self.output_dir / "similarity_sample.npy", similarity_matrix)
        
        print()
    
    def visualize_data_distribution(self):
        """Crea visualizaciones de distribución"""
        print("📊 [6/6] Creando visualizaciones...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Diagnóstico de Datos para Clustering', fontsize=16, fontweight='bold')
        
        # 1. Distribución de longitud de textos
        text_lengths = self.df['text'].str.split().str.len()
        axes[0, 0].hist(text_lengths, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Palabras')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].set_title('Distribución de Longitud de Textos')
        axes[0, 0].axvline(text_lengths.mean(), color='red', linestyle='--', label=f'Media: {text_lengths.mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Top fuentes
        top_sources = self.df['source'].value_counts().head(15)
        axes[0, 1].barh(range(len(top_sources)), top_sources.values)
        axes[0, 1].set_yticks(range(len(top_sources)))
        axes[0, 1].set_yticklabels(top_sources.index, fontsize=8)
        axes[0, 1].set_xlabel('Número de artículos')
        axes[0, 1].set_title('Top 15 Fuentes')
        axes[0, 1].invert_yaxis()
        
        # 3. Distribución de similitud (muestra)
        similarity_matrix = np.load(self.output_dir / "similarity_sample.npy")
        n = similarity_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        similarities = similarity_matrix[mask]
        
        axes[0, 2].hist(similarities, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Similitud coseno')
        axes[0, 2].set_ylabel('Frecuencia')
        axes[0, 2].set_title('Distribución de Similitud (TF-IDF)')
        axes[0, 2].axvline(similarities.mean(), color='red', linestyle='--', label=f'Media: {similarities.mean():.3f}')
        axes[0, 2].legend()
        
        # 4. Palabras por título
        title_lengths = self.df['title'].str.split().str.len()
        axes[1, 0].hist(title_lengths, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Palabras')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].set_title('Distribución de Longitud de Títulos')
        axes[1, 0].axvline(title_lengths.mean(), color='red', linestyle='--', label=f'Media: {title_lengths.mean():.1f}')
        axes[1, 0].legend()
        
        # 5. Artículos por fuente (distribución)
        source_counts = self.df['source'].value_counts()
        axes[1, 1].hist(source_counts.values, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Artículos por fuente')
        axes[1, 1].set_ylabel('Número de fuentes')
        axes[1, 1].set_title('Distribución de Artículos por Fuente')
        axes[1, 1].set_yscale('log')
        
        # 6. Heatmap de similitud (muestra pequeña)
        sample_size = min(30, similarity_matrix.shape[0])
        sns.heatmap(
            similarity_matrix[:sample_size, :sample_size],
            ax=axes[1, 2],
            cmap='YlOrRd',
            cbar_kws={'label': 'Similitud'},
            xticklabels=False,
            yticklabels=False
        )
        axes[1, 2].set_title(f'Heatmap de Similitud (muestra {sample_size}x{sample_size})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "data_distribution.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Visualización guardada: data_distribution.png")
        plt.close()
        
        print()
    
    def generate_report(self):
        """Genera reporte textual con diagnóstico y recomendaciones"""
        report = []
        report.append("="*70)
        report.append(" "*15 + "REPORTE DE DIAGNÓSTICO DE CLUSTERING")
        report.append("="*70)
        report.append("")
        
        report.append(f"Dataset: {self.data_path}")
        report.append(f"Total de documentos: {len(self.df)}")
        report.append(f"Fuentes únicas: {self.df['source'].nunique()}")
        report.append("")
        
        # Problemas identificados
        report.append("🔴 PROBLEMAS IDENTIFICADOS:")
        report.append("-"*70)
        if self.issues:
            for issue in self.issues:
                report.append(issue)
        else:
            report.append("✓ No se encontraron problemas críticos")
        report.append("")
        
        # Recomendaciones
        report.append("💡 RECOMENDACIONES:")
        report.append("-"*70)
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                report.append(f"{i}. {rec}")
        else:
            report.append("✓ Los datos están en buenas condiciones para clustering")
        report.append("")
        
        # Recomendaciones específicas para mejorar clustering
        report.append("🎯 ACCIONES ESPECÍFICAS PARA MEJORAR CLUSTERING:")
        report.append("-"*70)
        report.append("1. PREPROCESAMIENTO:")
        report.append("   • Combinar título + descripción + contenido")
        report.append("   • Eliminar duplicados y textos muy cortos")
        report.append("   • Normalizar longitudes si hay mucha variabilidad")
        report.append("")
        report.append("2. FEATURES:")
        report.append("   • Usar TF-IDF con n-grams (1,2) o (1,3)")
        report.append("   • Considerar embeddings contextuales (BERT)")
        report.append("   • Experimentar con diferentes max_features")
        report.append("")
        report.append("3. ALGORITMO:")
        report.append("   • Si similitud baja: Usar K-Means con K grande (15-25)")
        report.append("   • Si similitud alta: HDBSCAN con min_cluster_size bajo (3-5)")
        report.append("   • Probar Agglomerative Clustering (jerárquico)")
        report.append("")
        report.append("4. FINE-TUNING:")
        report.append("   • Más épocas (5-10)")
        report.append("   • Batch size mayor (32-64) para mejores negatives")
        report.append("   • Usar TF-IDF para crear pares de entrenamiento")
        report.append("   • Agregar hard negatives (30-40%)")
        report.append("")
        report.append("5. VALIDACIÓN:")
        report.append("   • Target: Silhouette > 0.3 (aceptable), > 0.5 (bueno)")
        report.append("   • Target: Davies-Bouldin < 1.0")
        report.append("   • Target: Outliers < 10%")
        report.append("")
        
        report.append("="*70)
        
        # Guardar reporte
        report_text = "\n".join(report)
        with open(self.output_dir / "diagnostic_report.txt", 'w') as f:
            f.write(report_text)
        
        # Imprimir también
        print("\n" + report_text)


def main():
    """Ejecuta diagnóstico"""
    diagnostic = ClusteringDiagnostic("1_Data/processed/clustering_data.csv")
    diagnostic.run_full_diagnostic()


if __name__ == "__main__":
    main()