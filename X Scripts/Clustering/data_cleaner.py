"""
LIMPIEZA DE DATOS BASADA EN DIAGNÓSTICO
========================================

PROBLEMAS DETECTADOS EN TUS DATOS:
1. ❌ 51 títulos duplicados (12.7%)
2. ⚠️ 124 fuentes con < 3 artículos (ruido)
3. ❌ Similitud promedio muy baja (documentos muy diferentes)

SOLUCIONES:
- Eliminar duplicados
- Filtrar fuentes con pocos artículos
- Combinar título + descripción + contenido para más contexto
- Filtrar textos muy cortos
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter

class DataCleaner:
    """
    Limpia datos de noticias para mejorar clustering
    """
    
    def __init__(self, input_path: str, output_path: str = None):
        self.input_path = input_path
        
        if output_path is None:
            # Guardar en mismo directorio con sufijo _cleaned
            input_path_obj = Path(input_path)
            self.output_path = input_path_obj.parent / f"{input_path_obj.stem}_cleaned.csv"
        else:
            self.output_path = Path(output_path)
        
        self.df = None
        self.stats = {}
    
    def load_data(self):
        """Carga datos"""
        print("\n" + "="*70)
        print(" "*20 + "🧹 LIMPIEZA DE DATOS")
        print("="*70 + "\n")
        
        self.df = pd.read_csv(self.input_path)
        print(f"📂 Datos cargados: {len(self.df)} documentos")
        
        self.stats['original_count'] = len(self.df)
        
        return self.df
    
    def remove_duplicates(self):
        """Elimina duplicados por título"""
        print("\n[1/6] 🔍 Eliminando duplicados...")
        
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['title'], keep='first')
        after = len(self.df)
        
        removed = before - after
        print(f"  ✓ Eliminados: {removed} duplicados ({removed/before*100:.1f}%)")
        print(f"  ✓ Restantes: {after} documentos")
        
        self.stats['duplicates_removed'] = removed
    
    def filter_short_texts(self, min_words: int = 30):
        """Filtra textos muy cortos"""
        print(f"\n[2/6] 📏 Filtrando textos cortos (< {min_words} palabras)...")
        
        before = len(self.df)
        
        # Calcular longitud
        word_counts = self.df['text'].fillna('').str.split().str.len()
        
        # Filtrar
        self.df = self.df[word_counts >= min_words]
        after = len(self.df)
        
        removed = before - after
        print(f"  ✓ Eliminados: {removed} textos cortos ({removed/before*100:.1f}%)")
        print(f"  ✓ Restantes: {after} documentos")
        
        self.stats['short_texts_removed'] = removed
    
    def filter_low_article_sources(self, min_articles: int = 2):
        """Filtra fuentes con muy pocos artículos"""
        print(f"\n[3/6] 📡 Filtrando fuentes con < {min_articles} artículos...")
        
        before = len(self.df)
        
        # Contar artículos por fuente
        source_counts = self.df['source'].value_counts()
        valid_sources = source_counts[source_counts >= min_articles].index
        
        # Filtrar
        self.df = self.df[self.df['source'].isin(valid_sources)]
        after = len(self.df)
        
        removed = before - after
        removed_sources = len(source_counts) - len(valid_sources)
        
        print(f"  ✓ Fuentes eliminadas: {removed_sources}")
        print(f"  ✓ Documentos eliminados: {removed} ({removed/before*100:.1f}%)")
        print(f"  ✓ Restantes: {after} documentos de {len(valid_sources)} fuentes")
        
        self.stats['low_article_sources_removed'] = removed
        self.stats['sources_after_filter'] = len(valid_sources)
    
    def combine_text_fields(self):
        """
        Combina título + descripción + contenido para más contexto
        
        ¿POR QUÉ ES IMPORTANTE?
        - Más contexto = mejor representación semántica
        - Títulos solos pueden ser ambiguos
        - Descripción añade información clave
        """
        print("\n[4/6] 📝 Combinando campos de texto...")
        
        def combine_fields(row):
            """Combina campos disponibles"""
            parts = []
            
            # Título (peso doble porque es lo más importante)
            if pd.notna(row.get('title')):
                parts.append(row['title'])
                parts.append(row['title'])  # Duplicar para dar más peso
            
            # Descripción
            if pd.notna(row.get('description')):
                parts.append(row['description'])
            
            # Contenido
            if pd.notna(row.get('content')):
                # Limitar contenido a primeras 500 palabras
                content_words = str(row['content']).split()[:500]
                parts.append(' '.join(content_words))
            
            # Full content si existe
            if pd.notna(row.get('full_content')):
                full_content_words = str(row['full_content']).split()[:300]
                parts.append(' '.join(full_content_words))
            
            return ' '.join(parts)
        
        # Crear columna combinada
        self.df['combined_text'] = self.df.apply(combine_fields, axis=1)
        
        # Limpiar
        self.df['combined_text'] = self.df['combined_text'].str.replace(r'[^\w\s\.\,\!\?]', ' ', regex=True)
        self.df['combined_text'] = self.df['combined_text'].str.replace(r'\s+', ' ', regex=True)
        self.df['combined_text'] = self.df['combined_text'].str.strip()
        
        # Estadísticas
        avg_words = self.df['combined_text'].str.split().str.len().mean()
        median_words = self.df['combined_text'].str.split().str.len().median()
        
        print(f"  ✓ Texto combinado creado")
        print(f"  ✓ Promedio: {avg_words:.1f} palabras")
        print(f"  ✓ Mediana: {median_words:.1f} palabras")
        
        # Usar combined_text como 'text'
        self.df['text'] = self.df['combined_text']
        
        self.stats['avg_text_length'] = avg_words
    
    def remove_near_duplicates(self, similarity_threshold: float = 0.95):
        """
        Elimina textos muy similares (casi duplicados)
        
        ¿POR QUÉ?
        Títulos diferentes pero contenido idéntico
        """
        print(f"\n[5/6] 🔎 Eliminando casi-duplicados (similitud > {similarity_threshold})...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        before = len(self.df)
        
        # TF-IDF rápido
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['text'])
        
        # Encontrar duplicados
        to_remove = set()
        
        # Procesar en chunks para eficiencia
        chunk_size = 50
        for i in range(0, len(self.df), chunk_size):
            end_i = min(i + chunk_size, len(self.df))
            
            # Similitud del chunk con todo el dataset
            similarities = cosine_similarity(tfidf_matrix[i:end_i], tfidf_matrix)
            
            for j in range(similarities.shape[0]):
                actual_idx = i + j
                # Encontrar documentos muy similares (excluyendo él mismo)
                similar_indices = np.where(
                    (similarities[j] > similarity_threshold) & 
                    (np.arange(len(self.df)) != actual_idx)
                )[0]
                
                # Marcar para remover (mantener el primero)
                for idx in similar_indices:
                    if idx > actual_idx:  # Solo remover los que vienen después
                        to_remove.add(idx)
        
        # Remover
        if to_remove:
            indices_to_keep = [i for i in range(len(self.df)) if i not in to_remove]
            self.df = self.df.iloc[indices_to_keep].reset_index(drop=True)
        
        after = len(self.df)
        removed = before - after
        
        print(f"  ✓ Casi-duplicados eliminados: {removed} ({removed/before*100:.1f}%)")
        print(f"  ✓ Restantes: {after} documentos")
        
        self.stats['near_duplicates_removed'] = removed
    
    def final_validation(self):
        """Validación final"""
        print("\n[6/6] ✅ Validación final...")
        
        # Verificar que no hay nulos en campos críticos
        null_counts = self.df[['text', 'title', 'source']].isnull().sum()
        
        if null_counts.any():
            print(f"  ⚠️  Nulos encontrados: {null_counts.to_dict()}")
            # Eliminar filas con nulos
            self.df = self.df.dropna(subset=['text', 'title', 'source'])
            print(f"  ✓ Filas con nulos eliminadas")
        else:
            print("  ✓ No hay valores nulos")
        
        # Estadísticas finales
        print(f"\n  📊 ESTADÍSTICAS FINALES:")
        print(f"    • Total documentos: {len(self.df)}")
        print(f"    • Fuentes únicas: {self.df['source'].nunique()}")
        print(f"    • Longitud promedio: {self.df['text'].str.split().str.len().mean():.1f} palabras")
        print(f"    • Longitud mediana: {self.df['text'].str.split().str.len().median():.1f} palabras")
        
        # Top fuentes
        top_sources = self.df['source'].value_counts().head(5)
        print(f"\n  🏆 TOP 5 FUENTES:")
        for source, count in top_sources.items():
            print(f"    • {source}: {count} artículos")
    
    def save_cleaned_data(self):
        """Guarda datos limpios"""
        print(f"\n💾 Guardando datos limpios...")
        
        # Guardar CSV
        self.df.to_csv(self.output_path, index=False)
        print(f"  ✓ Guardado en: {self.output_path}")
        
        # Guardar reporte de limpieza
        report_path = self.output_path.parent / f"{self.output_path.stem}_cleaning_report.txt"
        
        report_lines = [
            "="*70,
            " "*20 + "REPORTE DE LIMPIEZA",
            "="*70,
            "",
            f"Archivo original: {self.input_path}",
            f"Archivo limpio: {self.output_path}",
            "",
            "ESTADÍSTICAS:",
            "-"*70,
            f"Documentos originales: {self.stats['original_count']}",
            f"Documentos finales: {len(self.df)}",
            f"Reducción: {(1 - len(self.df)/self.stats['original_count'])*100:.1f}%",
            "",
            "ELIMINACIONES:",
            "-"*70,
            f"Duplicados: {self.stats.get('duplicates_removed', 0)}",
            f"Textos cortos: {self.stats.get('short_texts_removed', 0)}",
            f"Fuentes con pocos artículos: {self.stats.get('low_article_sources_removed', 0)}",
            f"Casi-duplicados: {self.stats.get('near_duplicates_removed', 0)}",
            "",
            "RESULTADO FINAL:",
            "-"*70,
            f"Documentos: {len(self.df)}",
            f"Fuentes: {self.stats.get('sources_after_filter', 'N/A')}",
            f"Longitud promedio: {self.stats.get('avg_text_length', 0):.1f} palabras",
            "",
            "="*70,
        ]
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✓ Reporte guardado en: {report_path}")
    
    def print_summary(self):
        """Imprime resumen final"""
        print("\n" + "="*70)
        print(" "*20 + "📊 RESUMEN DE LIMPIEZA")
        print("="*70)
        
        reduction = (1 - len(self.df) / self.stats['original_count']) * 100
        
        print(f"\n🔢 ANTES → DESPUÉS:")
        print(f"  Documentos: {self.stats['original_count']} → {len(self.df)} (-{reduction:.1f}%)")
        
        if reduction > 50:
            print(f"\n  ⚠️  ADVERTENCIA: Reducción > 50%")
            print(f"     Considera ajustar filtros si perdiste demasiados datos")
        elif reduction > 30:
            print(f"\n  ✓ Reducción moderada (30-50%)")
        else:
            print(f"\n  ✓ Reducción conservadora (< 30%)")
        
        print("\n✅ LISTO PARA CLUSTERING")
        print(f"   Usa: {self.output_path}")
        print("="*70 + "\n")
    
    def run_full_cleaning(self):
        """Pipeline completo de limpieza"""
        self.load_data()
        self.remove_duplicates()
        self.filter_short_texts(min_words=30)
        self.filter_low_article_sources(min_articles=2)
        self.combine_text_fields()
        self.remove_near_duplicates(similarity_threshold=0.95)
        self.final_validation()
        self.save_cleaned_data()
        self.print_summary()
        
        return self.df


def main():
    """Ejecuta limpieza"""
    cleaner = DataCleaner(
        input_path="1_Data/processed/clustering_data.csv",
        output_path="1_Data/processed/clustering_data_cleaned.csv"
    )
    
    cleaned_df = cleaner.run_full_cleaning()
    
    print("\n🎯 SIGUIENTE PASO:")
    print("   Ejecuta el optimizer con los datos limpios:")
    print("   python clustering_optimizer.py")
    print()


if __name__ == "__main__":
    main()