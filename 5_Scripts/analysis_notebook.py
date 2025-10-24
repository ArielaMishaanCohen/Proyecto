"""
Archivo: 4_Notebooks/interactive_analysis.ipynb

Notebook para análisis interactivo de los modelos fine-tuneados

CONVIERTE ESTE ARCHIVO A .ipynb con:
jupyter nbconvert --to notebook analysis_notebook.py --output interactive_analysis.ipynb
"""

# %% [markdown]
# # 📊 Análisis Interactivo de Modelos Fine-tuneados
# 
# Este notebook permite interactuar con los tres modelos entrenados:
# 1. Summarization
# 2. Clustering  
# 3. Sentiment Analysis

# %%
# Imports
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

print("✓ Librerías cargadas")

# %% [markdown]
# ## 1. Cargar Modelos Fine-tuneados

# %%
# Summarization Model
print("Cargando modelo de Summarization...")
summarization_tokenizer = BartTokenizer.from_pretrained("2_Models/summarization/final_model")
summarization_model = BartForConditionalGeneration.from_pretrained("2_Models/summarization/final_model")
summarization_model.eval()

# Sentiment Model
print("Cargando modelo de Sentiment...")
sentiment_tokenizer = DistilBertTokenizer.from_pretrained("2_Models/sentiment/final_model")
sentiment_model = DistilBertForSequenceClassification.from_pretrained("2_Models/sentiment/final_model")
sentiment_model.eval()

# Clustering Model
print("Cargando modelo de Clustering...")
clustering_model = SentenceTransformer("2_Models/clustering/fine_tuned_model")

print("\n✓ Todos los modelos cargados")

# %%
# Configurar device
device = "cuda" if torch.cuda.is_available() else "cpu"
summarization_model.to(device)
sentiment_model.to(device)
clustering_model.to(device)
print(f"Usando device: {device}")

# %% [markdown]
# ## 2. Funciones de Predicción

# %%
def summarize_text(text, max_length=128):
    """Genera resumen de texto"""
    inputs = summarization_tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = summarization_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    summary = summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def analyze_sentiment(text):
    """Analiza sentimiento de texto"""
    inputs = sentiment_tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    pred_label_id = torch.argmax(probs).item()
    
    return {
        'sentiment': label_map[pred_label_id],
        'confidence': float(probs[pred_label_id]),
        'probabilities': {
            label_map[i]: float(probs[i])
            for i in range(len(probs))
        }
    }


def get_embedding(text):
    """Obtiene embedding de texto"""
    return clustering_model.encode(text, convert_to_numpy=True)

# %% [markdown]
# ## 3. Demo Interactiva

# %%
# Ejemplos de noticias para probar
example_news = [
    {
        'title': 'Tech Company Reports Record Profits',
        'text': 'Major technology company announced record-breaking quarterly profits today, exceeding analyst expectations by 15%. The company attributed success to strong cloud services growth and increased adoption of AI technologies across enterprise customers. Stock prices surged 8% in after-hours trading.'
    },
    {
        'title': 'Natural Disaster Causes Widespread Damage',
        'text': 'A powerful earthquake struck the region early this morning, causing extensive damage to infrastructure and buildings. Emergency services are working to rescue trapped individuals and provide aid to affected communities. The disaster has displaced thousands of residents.'
    },
    {
        'title': 'New Study on Climate Change Published',
        'text': 'Researchers have published a comprehensive study examining long-term climate trends. The study analyzed data from multiple sources and presents findings on temperature variations and their potential impacts on ecosystems. Scientists call for continued monitoring and policy action.'
    }
]

# %%
# Analizar cada ejemplo
print("="*80)
print("ANÁLISIS DE EJEMPLOS")
print("="*80)

for i, news in enumerate(example_news, 1):
    print(f"\n{'='*80}")
    print(f"EJEMPLO {i}: {news['title']}")
    print(f"{'='*80}")
    
    # Texto original
    print(f"\n📰 Texto original:")
    print(news['text'])
    
    # Summarization
    summary = summarize_text(news['text'])
    print(f"\n📝 Resumen generado:")
    print(summary)
    
    # Sentiment
    sentiment = analyze_sentiment(news['text'])
    print(f"\n😊 Análisis de sentimiento:")
    print(f"  Sentimiento: {sentiment['sentiment'].upper()}")
    print(f"  Confianza: {sentiment['confidence']:.2%}")
    print(f"  Probabilidades:")
    for label, prob in sentiment['probabilities'].items():
        print(f"    {label}: {prob:.2%}")
    
    # Embedding (visualizar dimensiones)
    embedding = get_embedding(news['text'])
    print(f"\n🔢 Embedding generado:")
    print(f"  Dimensiones: {embedding.shape}")
    print(f"  Primeros 5 valores: {embedding[:5]}")

# %% [markdown]
# ## 4. Análisis Comparativo de Múltiples Noticias

# %%
# Cargar datos procesados
df = pd.read_csv("1_Data/processed/clustering_data.csv")
print(f"Datos cargados: {len(df)} noticias")
print(f"\nPrimeras fuentes: {df['source'].value_counts().head()}")

# %%
# Seleccionar muestra para análisis
sample_size = 100
sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

print(f"Analizando muestra de {len(sample_df)} noticias...")

# %%
# Generar embeddings para toda la muestra
embeddings = []
for text in sample_df['text']:
    emb = get_embedding(text)
    embeddings.append(emb)

embeddings = np.array(embeddings)
print(f"✓ Embeddings generados: {embeddings.shape}")

# %% [markdown]
# ## 5. Visualización con t-SNE

# %%
# Reducir dimensionalidad con t-SNE
print("Ejecutando t-SNE (puede tardar un minuto)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)
print("✓ t-SNE completado")

# %%
# Visualizar
plt.figure(figsize=(14, 10))

# Colorear por fuente
sources = sample_df['source'].values
unique_sources = list(set(sources))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sources)))
source_to_color = {source: color for source, color in zip(unique_sources, colors)}

for source in unique_sources:
    mask = sources == source
    plt.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=[source_to_color[source]],
        label=source,
        alpha=0.6,
        s=100
    )

plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.title('Visualización de Embeddings de Noticias (t-SNE)', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("3_Results/visualizations/tsne_embeddings.png", dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualización guardada")

# %% [markdown]
# ## 6. Análisis de Similitud

# %%
def find_similar_news(query_text, df, embeddings, top_k=5):
    """Encuentra noticias similares a una query"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Embedding de query
    query_emb = get_embedding(query_text).reshape(1, -1)
    
    # Calcular similitudes
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Top K
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'title': df.iloc[idx]['title'],
            'source': df.iloc[idx]['source'],
            'similarity': similarities[idx]
        })
    
    return results

# %%
# Probar búsqueda de similares
query = "artificial intelligence and machine learning in healthcare"

print(f"Query: {query}")
print("\nNoticias similares encontradas:\n")

similar = find_similar_news(query, sample_df, embeddings, top_k=5)

for i, result in enumerate(similar, 1):
    print(f"{i}. {result['title']}")
    print(f"   Fuente: {result['source']}")
    print(f"   Similitud: {result['similarity']:.4f}\n")

# %% [markdown]
# ## 7. Análisis de Distribución de Sentimientos

# %%
# Analizar sentimiento de la muestra
print("Analizando sentimientos (puede tardar)...")
sentiments = []

for text in sample_df['text'][:50]:  # Limitar a 50 para velocidad
    sent = analyze_sentiment(text)
    sentiments.append(sent['sentiment'])

# Crear DataFrame
sentiment_df = pd.DataFrame({
    'sentiment': sentiments,
    'source': sample_df['source'][:50].values
})

# %%
# Visualizar distribución
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distribución general
sentiment_counts = sentiment_df['sentiment'].value_counts()
axes[0].bar(sentiment_counts.index, sentiment_counts.values, 
            color=['red', 'gray', 'green'])
axes[0].set_title('Distribución de Sentimientos', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Frecuencia')
axes[0].grid(axis='y', alpha=0.3)

# Por fuente
sentiment_by_source = pd.crosstab(
    sentiment_df['source'], 
    sentiment_df['sentiment']
)
sentiment_by_source.plot(kind='bar', stacked=True, ax=axes[1], 
                         color=['red', 'gray', 'green'])
axes[1].set_title('Sentimientos por Fuente', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frecuencia')
axes[1].set_xlabel('Fuente')
axes[1].legend(title='Sentimiento')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("3_Results/visualizations/sentiment_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

print("✓ Distribución de sentimientos visualizada")

# %% [markdown]
# ## 8. Comparación de Resúmenes

# %%
# Comparar resúmenes generados vs originales
comparison_samples = sample_df.sample(n=3, random_state=42)

print("="*80)
print("COMPARACIÓN DE RESÚMENES")
print("="*80)

for idx, row in comparison_samples.iterrows():
    print(f"\n{'='*80}")
    print(f"NOTICIA: {row['title']}")
    print(f"{'='*80}")
    
    # Texto original (primeros 500 chars)
    original_text = row['text'][:500] + "..."
    print(f"\n📰 Texto original:")
    print(original_text)
    
    # Resumen generado
    summary = summarize_text(row['text'], max_length=100)
    print(f"\n📝 Resumen del modelo:")
    print(summary)
    
    # Título original (como "resumen humano")
    print(f"\n✍️ Título original (referencia):")
    print(row['title'])
    
    # Calcular longitud
    print(f"\n📊 Estadísticas:")
    print(f"  Texto original: {len(row['text'].split())} palabras")
    print(f"  Resumen generado: {len(summary.split())} palabras")
    print(f"  Ratio de compresión: {len(summary.split())/len(row['text'].split()):.2%}")

# %% [markdown]
# ## 9. Matriz de Similitud entre Noticias

# %%
from sklearn.metrics.pairwise import cosine_similarity

# Tomar subset pequeño para visualización
n_sample = 20
subset_indices = sample_df.sample(n=n_sample, random_state=42).index
subset_embeddings = embeddings[subset_indices - sample_df.index[0]]
subset_titles = sample_df.loc[subset_indices, 'title'].values

# Calcular similitudes
similarity_matrix = cosine_similarity(subset_embeddings)

# %%
# Visualizar matriz
plt.figure(figsize=(14, 12))
sns.heatmap(
    similarity_matrix,
    cmap='viridis',
    xticklabels=[t[:30] + "..." for t in subset_titles],
    yticklabels=[t[:30] + "..." for t in subset_titles],
    cbar_kws={'label': 'Similitud Coseno'},
    vmin=0,
    vmax=1
)
plt.title('Matriz de Similitud entre Noticias', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("3_Results/visualizations/similarity_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

print("✓ Matriz de similitud visualizada")

# %% [markdown]
# ## 10. Testing Interactivo

# %%
# Función para análisis completo de una noticia
def analyze_news_complete(text):
    """Análisis completo de una noticia"""
    print("="*80)
    print("ANÁLISIS COMPLETO")
    print("="*80)
    
    # Resumen
    print("\n📝 RESUMEN:")
    summary = summarize_text(text, max_length=128)
    print(summary)
    
    # Sentiment
    print("\n😊 SENTIMIENTO:")
    sentiment = analyze_sentiment(text)
    print(f"  {sentiment['sentiment'].upper()} ({sentiment['confidence']:.2%} confianza)")
    
    # Embedding
    print("\n🔢 EMBEDDING:")
    embedding = get_embedding(text)
    print(f"  Dimensiones: {embedding.shape}")
    
    # Similar news
    print("\n🔍 NOTICIAS SIMILARES:")
    similar = find_similar_news(text, sample_df, embeddings, top_k=3)
    for i, sim in enumerate(similar, 1):
        print(f"  {i}. {sim['title'][:60]}... ({sim['similarity']:.3f})")
    
    print("\n" + "="*80)

# %%
# Probar con texto custom
custom_text = """
Breaking news: A major technology company has announced a groundbreaking 
development in artificial intelligence. The new AI system demonstrates 
unprecedented capabilities in natural language understanding and generation. 
Industry experts are calling this a significant milestone that could 
revolutionize how we interact with computers. The company's stock price 
surged following the announcement.
"""

analyze_news_complete(custom_text)

# %% [markdown]
# ## 11. Métricas y Performance

# %%
# Cargar métricas guardadas
import json

summarization_metrics = json.load(open("2_Models/summarization/test_metrics.json"))
clustering_metrics = json.load(open("2_Models/clustering/clustering_metrics.json"))
sentiment_metrics = json.load(open("2_Models/sentiment/test_metrics.json"))

print("="*80)
print("RESUMEN DE MÉTRICAS")
print("="*80)

print("\n📊 SUMMARIZATION (BART):")
print(f"  ROUGE-1: {summarization_metrics.get('test_rouge1', 0):.4f}")
print(f"  ROUGE-2: {summarization_metrics.get('test_rouge2', 0):.4f}")
print(f"  ROUGE-L: {summarization_metrics.get('test_rougeL', 0):.4f}")

print("\n📊 CLUSTERING (MiniLM):")
print(f"  Silhouette Score: {clustering_metrics.get('silhouette_score', 0):.4f}")
print(f"  Davies-Bouldin Index: {clustering_metrics.get('davies_bouldin_index', 0):.4f}")
print(f"  Clusters: {clustering_metrics.get('n_clusters', 0)}")

print("\n📊 SENTIMENT (DistilBERT):")
print(f"  Accuracy: {sentiment_metrics.get('test_accuracy', 0):.4f}")
print(f"  F1-Macro: {sentiment_metrics.get('test_f1_macro', 0):.4f}")
print(f"  F1-Weighted: {sentiment_metrics.get('test_f1_weighted', 0):.4f}")

# %%
# Visualizar comparación
fig, ax = plt.subplots(figsize=(12, 6))

models = ['Summarization\n(ROUGE-L)', 'Clustering\n(Silhouette)', 'Sentiment\n(F1-Macro)']
scores = [
    summarization_metrics.get('test_rougeL', 0),
    clustering_metrics.get('silhouette_score', 0),
    sentiment_metrics.get('test_f1_macro', 0)
]
colors = ['skyblue', 'lightcoral', 'lightgreen']

bars = ax.bar(models, scores, color=colors, alpha=0.7, edgecolor='black')

# Añadir valores
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparación de Performance de Modelos', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("3_Results/visualizations/model_scores_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Comparación de métricas completada")

# %% [markdown]
# ## 12. Conclusiones y Próximos Pasos

# %%
print("="*80)
print("CONCLUSIONES DEL PROYECTO")
print("="*80)

print("""
✅ LOGROS:
1. Fine-tuning exitoso de 3 modelos diferentes
2. Summarization: Genera resúmenes concisos y relevantes
3. Clustering: Agrupa noticias similares efectivamente
4. Sentiment: Clasifica sentimiento con buena precisión

📈 MEJORAS POSIBLES:
1. Más datos de entrenamiento (actualmente limitado)
2. Hyperparameter tuning más exhaustivo
3. Data augmentation para sentiment
4. Anotación manual para mejor ground truth
5. Modelos más grandes (BART-large, RoBERTa-large)

🚀 PRÓXIMOS PASOS:
1. Deployment como API REST
2. Interface web para usuarios
3. Pipeline en tiempo real con streaming
4. Monitoreo y reentrenamiento periódico
5. A/B testing en producción

💡 LECCIONES APRENDIDAS:
1. Fine-tuning requiere balance entre learning rate y epochs
2. Class weighting es crucial para datos desbalanceados
3. Contrastive learning efectivo sin labels manuales
4. Evaluación cualitativa tan importante como métricas
5. Visualizaciones ayudan a entender embeddings
""")

print("="*80)
print("✓✓✓ ANÁLISIS COMPLETO ✓✓✓")
print("="*80)

# %%
# Guardar este notebook
print("\n💾 Para guardar este análisis:")
print("   File > Download as > Notebook (.ipynb)")
print("   O usar: jupyter nbconvert --to notebook --execute interactive_analysis.ipynb")