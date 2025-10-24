# 🚀 Fine-tuning de LLMs para Análisis de Noticias

Proyecto completo de fine-tuning de tres modelos de Hugging Face para tareas de NLP en noticias: **Summarization**, **Clustering** y **Sentiment Analysis**.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Modelos y Decisiones Técnicas](#-modelos-y-decisiones-técnicas)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Decisiones de Fine-tuning Explicadas](#-decisiones-de-fine-tuning-explicadas)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un pipeline completo de fine-tuning de modelos de lenguaje para analizar noticias de múltiples fuentes. Los objetivos son:

1. **Summarization**: Generar resúmenes concisos de artículos
2. **Clustering**: Agrupar noticias por temas similares
3. **Sentiment Analysis**: Clasificar sentimiento (positivo/neutral/negativo)

### ¿Por qué este proyecto?

- **Aprendizaje profundo**: Entender fine-tuning desde cero
- **Aplicación práctica**: Análisis automatizado de noticias
- **Portfolio**: Proyecto completo de ML/NLP
- **Escalabilidad**: Fácilmente adaptable a otros dominios

---

## 📁 Estructura del Proyecto

```
proyecto/
│
├── 1_Data/
│   ├── raw/                    # JSONs originales de News API
│   ├── processed/              # Datos procesados (CSV)
│   └── train_test_splits/      # Splits de datos
│
├── 2_Models/
│   ├── summarization/          # BART fine-tuned
│   ├── clustering/             # MiniLM fine-tuned
│   └── sentiment/              # DistilBERT fine-tuned
│
├── 3_Results/
│   ├── metrics/                # Métricas de evaluación
│   ├── visualizations/         # Gráficos y plots
│   └── sample_outputs/         # Ejemplos de predicciones
│
├── 5_Scripts/
│   ├── data_processor.py       # Procesamiento de datos
│   ├── summarization_trainer.py
│   ├── clustering_trainer.py
│   ├── sentiment_trainer.py
│   └── main_pipeline.py        # Script principal
│
├── requirements.txt
└── README.md
```

---

## 🤖 Modelos y Decisiones Técnicas

### 1. Summarization: BART-base

**Modelo**: `facebook/bart-base` (139M parámetros)

**¿Por qué BART?**
- Arquitectura encoder-decoder diseñada para seq2seq
- Pre-entrenado con denoising (reconstruir texto corrupto)
- Excelente para summarization, mejor que GPT-2 o T5-small

**Fine-tuning**:
- **Loss**: CrossEntropyLoss estándar
- **Learning rate**: 2e-5 con warmup
- **Batch size**: 4 + gradient accumulation (simula 16)
- **Epochs**: 3 (típico para fine-tuning)
- **Max input**: 1024 tokens
- **Max output**: 128 tokens (resúmenes concisos)

**Cambios específicos para noticias**:
1. **Formato de datos**: `full_content → title + description`
2. **Tokenización**: Truncation para artículos largos
3. **Métricas**: ROUGE-1, ROUGE-2, ROUGE-L
4. **Generación**: Beam search con 4 beams

**¿Por qué estos cambios?**
- Noticias tienen estructura específica (headline + body)
- ROUGE es estándar en summarization de noticias
- Beam search mejora coherencia vs greedy decoding

---

### 2. Clustering: MiniLM-L6-v2

**Modelo**: `sentence-transformers/all-MiniLM-L6-v2` (22M parámetros)

**¿Por qué MiniLM?**
- Optimizado para embeddings semánticos
- 5x más rápido que BERT-base
- Pequeño pero muy efectivo

**Fine-tuning**:
- **Técnica**: Contrastive Learning
- **Loss**: MultipleNegativesRankingLoss
- **Learning rate**: 2e-5
- **Epochs**: 3
- **Strategy**: Pares de noticias de misma fuente = similares

**Cambios específicos para noticias**:
1. **Training pairs**: Noticias de misma fuente como positivos
2. **Clustering**: HDBSCAN (no requiere K predefinido)
3. **Métricas**: Silhouette Score, Davies-Bouldin Index
4. **Visualización**: PCA para 2D

**¿Por qué estos cambios?**
- Contrastive learning sin labels manuales
- HDBSCAN maneja densidades variables (mejor que K-Means)
- Métricas unsupervised (no tenemos ground truth)

---

### 3. Sentiment Analysis: DistilBERT

**Modelo**: `distilbert-base-uncased` (66M parámetros)

**¿Por qué DistilBERT?**
- 40% más pequeño, 60% más rápido que BERT
- Retiene 97% del performance
- Perfecto para clasificación

**Fine-tuning**:
- **Arquitectura**: Clasificación con [CLS] token pooling
- **Loss**: CrossEntropyLoss con class weights
- **Learning rate**: 3e-5 con cosine annealing
- **Batch size**: 16
- **Epochs**: 5
- **Classes**: positive, neutral, negative

**Cambios específicos para noticias**:
1. **Class weighting**: Compensa desbalance de clases
2. **Labeling**: Keywords + heurísticas (mejorable)
3. **Métricas**: Accuracy, F1-macro, F1-weighted
4. **Early stopping**: Patience=3 en validation loss

**¿Por qué estos cambios?**
- Class weights críticos (noticias mayormente neutrales)
- F1-macro trata clases igual (mejor que accuracy)
- Early stopping previene overfitting

---

## 🔧 Instalación

### Prerrequisitos

- Python 3.8+
- GPU (recomendado) o CPU
- 16GB RAM mínimo

### Pasos

```bash
# 1. Clonar repositorio
git clone <tu-repo>
cd proyecto-news-llm

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Instalación con GPU (CUDA)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Uso

### Opción 1: Pipeline Completo (Recomendado)

```bash
python main_pipeline.py
```

Esto ejecuta:
1. Procesamiento de datos
2. Fine-tuning de summarization
3. Fine-tuning de clustering
4. Fine-tuning de sentiment
5. Generación de visualizaciones
6. Reporte final

**Tiempo estimado**: 2-4 horas (con GPU)

---

### Opción 2: Módulos Individuales

```bash
# Solo procesar datos
python 5_Scripts/data_processor.py

# Solo summarization
python 5_Scripts/summarization_trainer.py

# Solo clustering
python 5_Scripts/clustering_trainer.py

# Solo sentiment
python 5_Scripts/sentiment_trainer.py
```

---

## 📊 Resultados

### Métricas Esperadas

| Modelo | Métrica Principal | Valor Típico |
|--------|------------------|--------------|
| Summarization | ROUGE-L | 0.30 - 0.45 |
| Clustering | Silhouette Score | 0.20 - 0.40 |
| Sentiment | F1-Macro | 0.60 - 0.75 |

### Visualizaciones Generadas

1. **model_comparison.png**: Compara performance de los 3 modelos
2. **training_timeline.png**: Timeline de ejecución
3. **model_resources.png**: Recursos (parámetros, tamaño, tiempo)
4. **clusters_visualization.png**: Clusters en 2D (PCA)
5. **confusion_matrix.png**: Matriz de confusión de sentiment

---

## 🧠 Decisiones de Fine-tuning Explicadas

### ¿Por qué Fine-tuning y no Pre-trained?

**Pre-trained models** son generales. **Fine-tuning** adapta al dominio específico:

- **Vocabulario**: Aprende jerga de noticias
- **Estilo**: Aprende formato de headlines
- **Distribución**: Se ajusta a tus datos específicos

### Cambios Clave en Fine-tuning

#### 1. Learning Rate

```python
# Pre-training: 1e-3 o mayor
# Fine-tuning: 2e-5 (100x menor)
```

**¿Por qué más bajo?**
- Modelo ya sabe lenguaje
- Solo ajustamos ligeramente
- Evita "olvidar" conocimiento pre-entrenado (catastrophic forgetting)

---

#### 2. Epochs

```python
# Pre-training: 100+ epochs
# Fine-tuning: 3-5 epochs
```

**¿Por qué menos epochs?**
- Menos datos que pre-training
- Riesgo de overfitting
- Convergencia más rápida

---

#### 3. Batch Size + Gradient Accumulation

```python
batch_size = 4
gradient_accumulation_steps = 4
# Effective batch size = 16
```

**¿Por qué este truco?**
- GPUs consumer tienen memoria limitada
- Batches grandes mejoran estabilidad
- Gradient accumulation simula batches grandes sin usar más memoria

---

#### 4. Warmup Steps

```python
warmup_steps = 500
# o warmup_ratio = 0.1 (10% de pasos)
```

**¿Por qué warmup?**
- Learning rate aumenta gradualmente
- Previene inestabilidad inicial
- Da tiempo al optimizer de "explorar"

---

#### 5. Weight Decay

```python
weight_decay = 0.01
```

**¿Por qué?**
- Regularización L2
- Penaliza pesos grandes
- Previene overfitting

---

### Técnicas Específicas por Tarea

#### Summarization

**Cambio**: Beam Search en generación
```python
num_beams = 4
```
**Impacto**: +5-10% en ROUGE vs greedy decoding

---

#### Clustering

**Cambio**: Contrastive Learning
```python
loss = MultipleNegativesRankingLoss()
```
**Impacto**: Embeddings más discriminativos, mejor separación

---

#### Sentiment

**Cambio**: Class Weights
```python
class_weights = compute_class_weight('balanced', ...)
```
**Impacto**: +10-15% F1 en clases minoritarias

---

## 📈 Mejoras Futuras

1. **Labeling de Sentiment**: Usar anotadores humanos o modelos pre-entrenados
2. **Data Augmentation**: Back-translation, paraphrasing
3. **Ensemble**: Combinar múltiples modelos
4. **Hyperparameter Tuning**: Grid search o Bayesian optimization
5. **Modelos más grandes**: Probar BART-large, RoBERTa-large
6. **Active Learning**: Iterar con ejemplos difíciles

---

## 🤝 Contribuciones

Pull requests son bienvenidos. Para cambios mayores, abre un issue primero.

---

## 📝 Licencia

MIT License - Libre para uso académico y comercial

---

## 👤 Autor

Tu Nombre - [GitHub](https://github.com/tu-usuario)

---

## 🙏 Agradecimientos

- Hugging Face por los modelos pre-entrenados
- News API por los datos
- Comunidad de PyTorch y Transformers

---

**¿Preguntas?** Abre un issue o contacta directamente.