# Análisis de Sentimiento - Comparación de Múltiples Modelos

## 1. Librerías y configuraciones
```python
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# Reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
OUTPUT_DIR = Path("2_Modelos/sentiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CSV = "1_Data/processed/sentiment_data.csv"  

# Device (MPS aware)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("DEVICE:", DEVICE)
```

## 2. Hiperparámetros

```python
# Definir 3 modelos a comparar
MODELS_TO_COMPARE = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "bert": "bert-base-uncased"
}

# Definir 3 configuraciones diferentes para cada modelo
CONFIGS = {
    "config_1": {
        "MAX_LENGTH": 128,
        "BATCH_SIZE": 16,
        "NUM_EPOCHS": 3,
        "LEARNING_RATE": 5e-5,
        "WARMUP_RATIO": 0.1,
        "WEIGHT_DECAY": 0.01,
    },
    "config_2": {
        "MAX_LENGTH": 256,
        "BATCH_SIZE": 16,
        "NUM_EPOCHS": 5,
        "LEARNING_RATE": 3e-5,
        "WARMUP_RATIO": 0.15,
        "WEIGHT_DECAY": 0.02,
    },
    "config_3": {
        "MAX_LENGTH": 256,
        "BATCH_SIZE": 8,
        "NUM_EPOCHS": 5,
        "LEARNING_RATE": 2e-5,
        "WARMUP_RATIO": 0.2,
        "WEIGHT_DECAY": 0.01,
    }
}

# Parámetros comunes
GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 3
EVAL_STEPS = 200
SAVE_STEPS = 200
LOGGING_STEPS = 50

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
```

## 3. Carga y preparación de datos

```python
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
import re

stop_words = set(stopwords.words("english"))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # solo letras y espacios
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
```

```python
df = pd.read_csv(DATA_CSV)
df = df.rename(columns={"full_content": "text"})
df.head()
```

```python
df['text'] = df['text'].apply(clean_text)
```

### Mapear labels

```python
if "sentiment" in df.columns:
    df["label"] = df["sentiment"].map(LABEL2ID)
else:
    df["label"] = df["label"].astype(int)

df.head()
```

```python
print("Distribución:")
print(df["sentiment"].value_counts())
print(df["sentiment"].value_counts(normalize=True))
```

### Split train, val y test

```python
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

## 4. Pesos de clase

```python
labels = train_df["label"].values
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)
```

## 5. Trainer personalizado (weighted loss)

```python
class WeightedTrainer(Trainer):
    def __init__(self, class_weights_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        class_weights = self._class_weights.to(device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

## 6. Métricas

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weight = f1_score(labels, preds, average='weighted')

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weight,
    }

    # Añadir f1 por clase
    f1s = f1_score(labels, preds, average=None)
    for i, val in enumerate(f1s):
        metrics[f"f1_{ID2LABEL[i]}"] = float(val)

    return metrics
```

## 7. Función de preparación de datasets

```python
def prepare_datasets(tokenizer, max_length, train_df, val_df, test_df):
    """Prepara datasets tokenizados para un modelo específico"""
    
    def preprocess_texts(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
    
    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(preprocess_texts, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(preprocess_texts, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(preprocess_texts, batched=True, remove_columns=["text"])

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    return train_ds, val_ds, test_ds
```

## 8. Función de entrenamiento de un modelo

```python
def train_model(model_name, model_key, config_name, config, train_ds, val_ds):
    """
    Entrena un modelo con una configuración específica
    
    Args:
        model_name: Nombre del modelo (ej: "distilbert-base-uncased")
        model_key: Clave del modelo (ej: "distilbert")
        config_name: Nombre de la configuración (ej: "config_1")
        config: Dict con hiperparámetros
        train_ds, val_ds: Datasets de entrenamiento y validación
    
    Returns:
        model, trainer, metrics
    """
    print(f"\n{'='*80}")
    print(f"Entrenando: {model_key} - {config_name}")
    print(f"{'='*80}\n")
    
    # Crear directorio específico para este experimento
    exp_dir = OUTPUT_DIR / f"{model_key}_{config_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(exp_dir),
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        learning_rate=config["LEARNING_RATE"],
        per_device_train_batch_size=config["BATCH_SIZE"],
        per_device_eval_batch_size=config["BATCH_SIZE"],
        num_train_epochs=config["NUM_EPOCHS"],
        warmup_ratio=config["WARMUP_RATIO"],
        weight_decay=config["WEIGHT_DECAY"],
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        max_grad_norm=GRAD_CLIP,
        lr_scheduler_type="cosine",
    )
    
    # Trainer
    trainer = WeightedTrainer(
        class_weights_tensor=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    
    # Entrenar
    train_result = trainer.train()
    
    # Guardar modelo y métricas
    trainer.save_model(str(exp_dir / "final_model"))
    with open(exp_dir / "training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    # Evaluar en validación
    val_metrics = trainer.evaluate()
    
    return model, trainer, val_metrics
```

## 9. ENTRENAMIENTO DE TODOS LOS MODELOS Y CONFIGURACIONES

```python
# Diccionario para almacenar resultados
all_results = {}

# Iterar sobre cada modelo
for model_key, model_name in MODELS_TO_COMPARE.items():
    print(f"\n{'#'*80}")
    print(f"MODELO: {model_key.upper()} ({model_name})")
    print(f"{'#'*80}\n")
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Iterar sobre cada configuración
    for config_name, config in CONFIGS.items():
        
        # Preparar datasets con el max_length de esta config
        train_ds, val_ds, test_ds_prepared = prepare_datasets(
            tokenizer, 
            config["MAX_LENGTH"], 
            train_df, 
            val_df, 
            test_df
        )
        
        # Entrenar modelo
        try:
            model, trainer, val_metrics = train_model(
                model_name, 
                model_key, 
                config_name, 
                config,
                train_ds, 
                val_ds
            )
            
            # Guardar resultados
            experiment_key = f"{model_key}_{config_name}"
            all_results[experiment_key] = {
                "model_key": model_key,
                "model_name": model_name,
                "config_name": config_name,
                "config": config,
                "val_metrics": val_metrics,
                "model_path": str(OUTPUT_DIR / f"{model_key}_{config_name}" / "final_model"),
                "tokenizer": tokenizer,
                "test_ds": test_ds_prepared,
            }
            
            print(f"\n✅ {experiment_key} completado")
            print(f"Val F1 Macro: {val_metrics['eval_f1_macro']:.4f}")
            print(f"Val Accuracy: {val_metrics['eval_accuracy']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error en {experiment_key}: {str(e)}")
            continue

print("\n" + "="*80)
print("ENTRENAMIENTO COMPLETADO PARA TODOS LOS MODELOS")
print("="*80)
```

## 10. EVALUACIÓN EN TEST DE TODOS LOS MODELOS

```python
# Evaluar cada modelo en test
test_results = {}

print("\n" + "="*80)
print("EVALUANDO TODOS LOS MODELOS EN TEST SET")
print("="*80)

for exp_key, exp_data in all_results.items():
    print(f"\nEvaluando: {exp_key}")
    
    try:
        # Cargar modelo entrenado
        model = AutoModelForSequenceClassification.from_pretrained(exp_data["model_path"])
        model.to(DEVICE)
        
        # Crear trainer para evaluación
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "temp"),
            per_device_eval_batch_size=16,
        )
        
        trainer = WeightedTrainer(
            class_weights_tensor=class_weights,
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
        )
        
        # Predecir en test
        predictions = trainer.predict(exp_data["test_ds"])
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        # Guardar resultados
        test_results[exp_key] = {
            "metrics": predictions.metrics,
            "predictions": preds,
            "labels": labels,
            "probabilities": torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy(),
            "model_key": exp_data["model_key"],
            "config_name": exp_data["config_name"],
        }
        
        print(f"Test F1 Macro: {predictions.metrics['test_f1_macro']:.4f}")
        print(f"Test Accuracy: {predictions.metrics['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Error evaluando {exp_key}: {str(e)}")
        continue

print("\n" + "="*80)
print("EVALUACIÓN COMPLETADA")
print("="*80)
```

## 11. COMPARACIÓN Y SELECCIÓN DEL MEJOR MODELO

```python
# Crear DataFrame con resultados comparativos
comparison_data = []

for exp_key, results in test_results.items():
    comparison_data.append({
        "experiment": exp_key,
        "model": results["model_key"],
        "config": results["config_name"],
        "test_f1_macro": results["metrics"]["test_f1_macro"],
        "test_accuracy": results["metrics"]["test_accuracy"],
        "test_f1_weighted": results["metrics"]["test_f1_weighted"],
        "test_f1_negative": results["metrics"]["test_f1_negative"],
        "test_f1_neutral": results["metrics"]["test_f1_neutral"],
        "test_f1_positive": results["metrics"]["test_f1_positive"],
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values("test_f1_macro", ascending=False)

print("\n" + "="*80)
print("COMPARACIÓN DE TODOS LOS MODELOS")
print("="*80 + "\n")
print(comparison_df.to_string(index=False))

# Identificar el mejor modelo
best_experiment = comparison_df.iloc[0]["experiment"]
best_model_data = test_results[best_experiment]

print(f"\n{'='*80}")
print(f"🏆 MEJOR MODELO: {best_experiment}")
print(f"{'='*80}")
print(f"F1 Macro: {best_model_data['metrics']['test_f1_macro']:.4f}")
print(f"Accuracy: {best_model_data['metrics']['test_accuracy']:.4f}")
print(f"F1 Weighted: {best_model_data['metrics']['test_f1_weighted']:.4f}")

# Guardar comparación
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
```

## 12. VISUALIZACIÓN DE RESULTADOS

```python
# Gráfico de comparación de F1 Macro
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# F1 Macro por modelo y configuración
ax1 = axes[0, 0]
comparison_df_pivot = comparison_df.pivot(index="config", columns="model", values="test_f1_macro")
comparison_df_pivot.plot(kind="bar", ax=ax1)
ax1.set_title("F1 Macro Score por Modelo y Configuración")
ax1.set_ylabel("F1 Macro")
ax1.set_xlabel("Configuración")
ax1.legend(title="Modelo")
ax1.grid(axis='y', alpha=0.3)

# Accuracy por modelo y configuración
ax2 = axes[0, 1]
comparison_df_pivot_acc = comparison_df.pivot(index="config", columns="model", values="test_accuracy")
comparison_df_pivot_acc.plot(kind="bar", ax=ax2)
ax2.set_title("Accuracy por Modelo y Configuración")
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Configuración")
ax2.legend(title="Modelo")
ax2.grid(axis='y', alpha=0.3)

# F1 por clase del mejor modelo
ax3 = axes[1, 0]
f1_by_class = {
    "Negative": best_model_data["metrics"]["test_f1_negative"],
    "Neutral": best_model_data["metrics"]["test_f1_neutral"],
    "Positive": best_model_data["metrics"]["test_f1_positive"],
}
ax3.bar(f1_by_class.keys(), f1_by_class.values(), color=['red', 'gray', 'green'])
ax3.set_title(f"F1 Score por Clase - {best_experiment}")
ax3.set_ylabel("F1 Score")
ax3.grid(axis='y', alpha=0.3)

# Comparación general de métricas del mejor modelo
ax4 = axes[1, 1]
metrics_comparison = {
    "F1 Macro": best_model_data["metrics"]["test_f1_macro"],
    "F1 Weighted": best_model_data["metrics"]["test_f1_weighted"],
    "Accuracy": best_model_data["metrics"]["test_accuracy"],
}
ax4.bar(metrics_comparison.keys(), metrics_comparison.values(), color=['blue', 'orange', 'purple'])
ax4.set_title(f"Métricas Generales - {best_experiment}")
ax4.set_ylabel("Score")
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_plots.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Gráficos guardados en: {OUTPUT_DIR / 'model_comparison_plots.png'}")
```

## 13. MATRIZ DE CONFUSIÓN DEL MEJOR MODELO

```python
def plot_confusion_matrix(true_labels, pred_labels, exp_name, out_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=list(LABEL2ID.keys()), 
                yticklabels=list(LABEL2ID.keys()),
                cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - {exp_name}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f'✅ Matriz de confusión guardada en {out_path}')

# Generar matriz de confusión del mejor modelo
plot_confusion_matrix(
    best_model_data["labels"],
    best_model_data["predictions"],
    best_experiment,
    OUTPUT_DIR / "best_model_confusion_matrix.png"
)
```

## 14. CLASSIFICATION REPORT DEL MEJOR MODELO

```python
print("\n" + "="*80)
print(f"CLASSIFICATION REPORT - {best_experiment}")
print("="*80 + "\n")

print(classification_report(
    best_model_data["labels"], 
    best_model_data["predictions"], 
    target_names=list(LABEL2ID.keys())
))
```

## 15. EXPORTAR PREDICCIONES DEL MEJOR MODELO

```python
# Preparar dataframe con resultados del mejor modelo
pred_probs = best_model_data["probabilities"]
pred_labels = best_model_data["predictions"]
true_labels = best_model_data["labels"]

results_df = test_df.reset_index(drop=True).copy()
results_df["true_label"] = [ID2LABEL[i] for i in true_labels]
results_df["pred_label"] = [ID2LABEL[i] for i in pred_labels]

# Agregar las probabilidades de cada clase
for i, label_name in ID2LABEL.items():
    results_df[f"prob_{label_name}"] = pred_probs[:, i]

# Agregar información del modelo usado
results_df["model_used"] = best_experiment
results_df["model_f1_macro"] = best_model_data["metrics"]["test_f1_macro"]
results_df["model_accuracy"] = best_model_data["metrics"]["test_accuracy"]

# Guardar a CSV y Parquet
results_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
results_df.to_parquet(OUTPUT_DIR / "test_predictions.parquet", index=False)

print(f"\n✅ Predicciones exportadas:")
print(f"   - CSV: {OUTPUT_DIR / 'test_predictions.csv'}")
print(f"   - Parquet: {OUTPUT_DIR / 'test_predictions.parquet'}")
print(f"\n📊 Shape: {results_df.shape}")
print(f"📋 Columnas: {list(results_df.columns)}")
```

## 16. RESUMEN FINAL

```python
print("\n" + "="*80)
print("RESUMEN FINAL DEL EXPERIMENTO")
print("="*80 + "\n")

print(f"Total de experimentos: {len(all_results)}")
print(f"Modelos evaluados: {len(MODELS_TO_COMPARE)}")
print(f"Configuraciones por modelo: {len(CONFIGS)}")
print(f"\n🏆 Mejor modelo: {best_experiment}")
print(f"   - F1 Macro: {best_model_data['metrics']['test_f1_macro']:.4f}")
print(f"   - Accuracy: {best_model_data['metrics']['test_accuracy']:.4f}")
print(f"   - F1 Negative: {best_model_data['metrics']['test_f1_negative']:.4f}")
print(f"   - F1 Neutral: {best_model_data['metrics']['test_f1_neutral']:.4f}")
print(f"   - F1 Positive: {best_model_data['metrics']['test_f1_positive']:.4f}")

print(f"\n📁 Archivos generados:")
print(f"   - Comparación de modelos: {OUTPUT_DIR / 'model_comparison.csv'}")
print(f"   - Gráficos comparativos: {OUTPUT_DIR / 'model_comparison_plots.png'}")
print(f"   - Matriz de confusión: {OUTPUT_DIR / 'best_model_confusion_matrix.png'}")
print(f"   - Predicciones finales: {OUTPUT_DIR / 'test_predictions.csv'}")
print(f"   - Modelos entrenados: {OUTPUT_DIR / '<modelo>_<config>/final_model/'}")

print("\n" + "="*80)
print("✅ EXPERIMENTO COMPLETADO EXITOSAMENTE")
print("="*80)
```