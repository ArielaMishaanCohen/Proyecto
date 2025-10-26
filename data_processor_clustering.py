"""
Archivo: 5_Scripts/data_processor.py

Procesa los archivos JSON de noticias y prepara los datos para fine-tuning
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import re
from datetime import datetime

class NewsDataProcessor:
    """
    Procesa datos de noticias desde archivos JSON.
    
    ¿Por qué esta clase?
    - Centraliza toda la lógica de procesamiento de datos
    - Facilita el mantenimiento y testing
    - Permite reutilizar el código para diferentes fuentes
    """
    
    def __init__(self, raw_data_path: str = "1_Data/raw"):
        self.raw_data_path = Path(raw_data_path)
        
    def load_json_files(self) -> List[Dict]:
        """
        Carga todos los archivos JSON de la carpeta raw.
        
        Returns:
            Lista de artículos de todos los archivos JSON
        """
        all_articles = []
        json_files = list(self.raw_data_path.glob("*.json"))
        
        print(f"Encontrados {len(json_files)} archivos JSON")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'articles' in data:
                        all_articles.extend(data['articles'])
                print(f"✓ Cargado: {json_file.name} ({len(data.get('articles', []))} artículos)")
            except Exception as e:
                print(f"✗ Error cargando {json_file.name}: {e}")
        
        print(f"\nTotal de artículos cargados: {len(all_articles)}")
        return all_articles
    
    def filter_valid_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Filtra artículos que tienen full_content (requerido para fine-tuning).
        
        ¿Por qué filtrar?
        - Solo podemos entrenar con artículos completos
        - Los artículos parciales generarían resúmenes incorrectos
        - Asegura calidad de datos de entrenamiento
        
        Args:
            articles: Lista de artículos crudos
            
        Returns:
            DataFrame con artículos válidos
        """
        valid_articles = []
        
        for article in articles:
            # Verificar que tenga full_content y no esté vacío
            full_content = article.get('full_content', '')
            
            if (full_content and 
                full_content.strip() and 
                not full_content.startswith('ERROR') and
                len(full_content) > 100):  # Mínimo 100 caracteres
                
                valid_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'full_content': full_content,
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'author': article.get('author', 'Unknown'),
                    'publishedAt': article.get('publishedAt', ''),
                    'url': article.get('url', ''),
                })
        
        df = pd.DataFrame(valid_articles)
        print(f"Artículos válidos con full_content: {len(df)}")
        print(f"Artículos filtrados: {len(articles) - len(df)}")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto de caracteres especiales y formato HTML.
        
        ¿Por qué limpiar?
        - Los modelos aprenden mejor con texto limpio
        - Reduce ruido en los datos
        - Evita que el modelo aprenda artefactos irrelevantes
        """
        if not isinstance(text, str):
            return ""
        
        # Remover HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remover URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remover caracteres especiales pero mantener puntuación básica
        text = re.sub(r'[^\w\s.,!?;:\-\']', ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_summarization_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara datos específicamente para la tarea de summarization.
        
        ¿Por qué este formato?
        - Los modelos de resumen necesitan pares (texto_largo, texto_corto)
        - Usamos full_content como fuente y title+description como resumen objetivo
        - Este formato es estándar en datasets como CNN/DailyMail
        
        Returns:
            DataFrame con columnas 'input_text' y 'target_summary'
        """
        summarization_data = []
        
        for _, row in df.iterrows():
            # Limpiar textos
            input_text = self.clean_text(row['full_content'])
            
            # Crear resumen objetivo combinando título y descripción
            # ¿Por qué combinarlos? Porque juntos dan un resumen más completo
            target_parts = []
            if row['title']:
                target_parts.append(self.clean_text(row['title']))
            if row['description'] and row['description'] != row['title']:
                target_parts.append(self.clean_text(row['description']))
            
            target_summary = '. '.join(target_parts)
            
            # Solo incluir si ambos tienen contenido sustancial
            if len(input_text) > 200 and len(target_summary) > 20:
                summarization_data.append({
                    'input_text': input_text,
                    'target_summary': target_summary,
                    'source': row['source'],
                    'word_count': len(input_text.split()),
                    'summary_length': len(target_summary.split())
                })
        
        return pd.DataFrame(summarization_data)
    
    def prepare_clustering_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara datos para clustering (agrupar noticias similares).
        
        ¿Por qué clustering?
        - Descubrir temas automáticamente en las noticias
        - Identificar tendencias sin etiquetas previas
        - Útil para organizar grandes volúmenes de noticias
        
        Returns:
            DataFrame con texto limpio para clustering
        """
        clustering_data = []
        
        for _, row in df.iterrows():
            # Combinar título, descripción y contenido para mejor representación
            combined_text = ' '.join([
                str(row.get('title', '')),
                str(row.get('description', '')),
                str(row.get('full_content', ''))[:500]  # Primeros 500 chars del contenido
            ])
            
            cleaned_text = self.clean_text(combined_text)
            
            if len(cleaned_text) > 50:
                clustering_data.append({
                    'text': cleaned_text,
                    'source': row['source'],
                    'title': row['title']
                })
        
        return pd.DataFrame(clustering_data)
    
    def add_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade etiquetas de sentimiento basadas en keywords.
        
        NOTA IMPORTANTE: Este es un labeling inicial simple.
        Para production, idealmente usaríamos:
        - Anotación manual de expertos
        - Modelos pre-entrenados de sentiment
        - Active learning
        
        ¿Por qué este approach?
        - Es un starting point rápido para el proyecto
        - Permite entrenar un modelo baseline
        - Puede mejorarse iterativamente
        """
        positive_keywords = ['success', 'win', 'growth', 'improve', 'gain', 
                           'advance', 'positive', 'boost', 'profit', 'rise']
        negative_keywords = ['fail', 'loss', 'decline', 'crisis', 'scandal',
                           'threat', 'risk', 'fall', 'concern', 'warning']
        
        sentiments = []
        
        for _, row in df.iterrows():
            text = (str(row.get('title', '')) + ' ' + 
                   str(row.get('description', ''))).lower()
            
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            
            if pos_count > neg_count:
                sentiment = 'positive'
            elif neg_count > pos_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            sentiments.append(sentiment)
        
        df['sentiment'] = sentiments
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Guarda datos procesados"""
        output_path = Path("1_Data/processed") / filename
        df.to_csv(output_path, index=False)
        print(f"✓ Datos guardados en: {output_path}")
    
    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta todo el pipeline de procesamiento.
        
        Returns:
            Tuple de (summarization_df, clustering_df, sentiment_df)
        """
        print("="*60)
        print("INICIANDO PROCESAMIENTO DE DATOS")
        print("="*60)
        
        # 1. Cargar datos
        print("\n1. Cargando archivos JSON...")
        articles = self.load_json_files()
        
        # 2. Filtrar artículos válidos
        print("\n2. Filtrando artículos válidos...")
        valid_df = self.filter_valid_articles(articles)
        
        # 3. Preparar datos para cada tarea
        print("\n3. Preparando datos para summarization...")
        summarization_df = self.prepare_summarization_data(valid_df)
        
        print("\n4. Preparando datos para clustering...")
        clustering_df = self.prepare_clustering_data(valid_df)
        
        print("\n5. Añadiendo etiquetas de sentimiento...")
        sentiment_df = self.add_sentiment_labels(valid_df.copy())
        
        # 4. Guardar datos procesados
        print("\n6. Guardando datos procesados...")
        self.save_processed_data(summarization_df, 'summarization_data.csv')
        self.save_processed_data(clustering_df, 'clustering_data.csv')
        self.save_processed_data(sentiment_df, 'sentiment_data.csv')
        
        print("\n" + "="*60)
        print("PROCESAMIENTO COMPLETADO")
        print("="*60)
        print(f"Datos de Summarization: {len(summarization_df)} ejemplos")
        print(f"Datos de Clustering: {len(clustering_df)} ejemplos")
        print(f"Datos de Sentiment: {len(sentiment_df)} ejemplos")
        
        return summarization_df, clustering_df, sentiment_df


if __name__ == "__main__":
    # Ejecutar procesamiento
    processor = NewsDataProcessor()
    sum_df, clust_df, sent_df = processor.process_all()
    
    # Mostrar estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE DATOS")
    print("="*60)
    print("\nSummarization Data:")
    print(sum_df.describe())
    print("\nDistribución de sentiment:")
    print(sent_df['sentiment'].value_counts())
