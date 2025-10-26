# %%
"""
Notebook-friendly script (use in Jupyter / Jupytext).
Run cell-by-cell. Hecho para ejecutar paso a paso sin crear muchas funciones.
Instala dependencias si hace falta:
!pip install transformers datasets torch scikit-learn matplotlib seaborn
"""

# %%
# 1) IMPORTS Y CONFIG
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
OUTPUT_DIR = Path("2_Models/sentiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CSV = "1_Data/processed/sentiment_data.csv"  # ajusta si hace falta

# Device (MPS aware)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("DEVICE:", DEVICE)

# %%
# 2) HYPERPARAMS (modifica aquí)
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 3
EVAL_STEPS = 200
SAVE_STEPS = 200
LOGGING_STEPS = 50

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# %%
# 3) CARGA Y PREPARACIÓN DE DATOS (CSV)
df = pd.read_csv(DATA_CSV)
# Crear campo 'text' si tienes 'title' y 'description'
if "text" not in df.columns:
    df["text"] = df.get("title", "").fillna("") + ". " + df.get("description", "").fillna("")

# Mapear labels
if "sentiment" in df.columns:
    df["label"] = df["sentiment"].map(LABEL2ID)
else:
    # suponer que ya hay columna 'label' con ids
    df["label"] = df["label"].astype(int)

print("Distribución:")
print(df["sentiment"].value_counts())
print(df["sentiment"].value_counts(normalize=True))

# Split estratificado
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# %%
# 4) PESOS DE CLASE
labels = train_df["label"].values
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)

# Enviar a device luego dentro del Trainer (no aquí necesariamente)

# %%
# 5) TOKENIZER Y DATASETS
print("Cargando tokenizer y tokenizando...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Función simple de tokenización para map
def preprocess_texts(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )

# Crear datasets de HuggingFace (cada uno con columnas 'input_ids','attention_mask','label')
train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

train_ds = train_ds.map(preprocess_texts, batched=True, remove_columns=["text"])
val_ds = val_ds.map(preprocess_texts, batched=True, remove_columns=["text"])
test_ds = test_ds.map(preprocess_texts, batched=True, remove_columns=["text"])

# Asegurar formato
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print("Tokenización completa")

# %%
# 6) CARGAR MODELO
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# Mover modelo a device
model.to(DEVICE)

# %%
# 7) TRAINER PERSONALIZADO (Weighted Loss)
# Lo dejamos simple y robusto: acepta kwargs para compatibilidad con diferentes versiones

class WeightedTrainer(Trainer):
    def __init__(self, class_weights_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Guardar pesos y forzarlos al device del modelo cuando se use
        self._class_weights = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device

        # Mover tensores de inputs al device
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Asegurarse que los pesos estén en el mismo device
        class_weights = self._class_weights.to(device)

        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# %%
# 8) MÉTRICAS

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

# %%
# 9) TRAINING ARGUMENTS Y ENTRENAMIENTO (ejecutar cuando quieras)
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    max_grad_norm=GRAD_CLIP,
    lr_scheduler_type="cosine",
)

trainer = WeightedTrainer(
    class_weights_tensor=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

# Ejecutar entrenamiento con: train_result = trainer.train()
# Guarda modelo con: trainer.save_model(str(OUTPUT_DIR/"final_model"))

# %%
# 10) EVALUAR EN TEST (después de entrenar y cargar el mejor modelo)
# Asumiendo que ejecutaste `train_result = trainer.train()` y `trainer.save_model(...)`

# Si quieres forzar usar el mejor modelo guardado en disk, descomenta:
# model = DistilBertForSequenceClassification.from_pretrained(str(OUTPUT_DIR/"final_model"))
# model.to(DEVICE)
# trainer.model = model

# Predict
# predictions = trainer.predict(test_ds)
# preds = np.argmax(predictions.predictions, axis=1)
# labels = predictions.label_ids

# Imprimir métricas y classification report
# metrics = predictions.metrics
# print(metrics)
# print(classification_report(labels, preds, target_names=list(LABEL2ID.keys())))

# %%
# 11) MATRIZ DE CONFUSIÓN Y ANÁLISIS DE ERRORES (celdas separadas)

def plot_confusion_matrix(true_labels, pred_labels, out_path=OUTPUT_DIR/"confusion_matrix.png"):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(LABEL2ID.keys()), yticklabels=list(LABEL2ID.keys()))
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print('Confusion matrix guardada en', out_path)


def error_analysis_df(test_df_local, true_labels, pred_labels, predictions_probs):
    # test_df_local debe corresponder al test_df original alineado con test_ds
    rows = []
    for i, (t, p, probs) in enumerate(zip(true_labels, pred_labels, predictions_probs)):
        if t != p:
            rows.append({
                'text': test_df_local.iloc[i]['text'][:300],
                'true_label': ID2LABEL[int(t)],
                'pred_label': ID2LABEL[int(p)],
                'pred_confidence': float(np.max(probs)),
                'pred_probs': probs.tolist(),
            })
    return pd.DataFrame(rows)

# %%
# 12) PREDICCIÓN RÁPIDA

def predict_text(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    return {"text": text, "sentiment": ID2LABEL[pred], "confidence": float(probs[pred]), "probs": probs}

# Ejemplos (puedes ejecutar):
# print(predict_text("Company stocks soar to record highs after strong earnings report"))

# %%
# 13) GUARDAR RESULTADOS MANUALES
# Para guardar métricas o análisis después de evaluar:
# with open(OUTPUT_DIR/"training_metrics.json", "w") as f:
#     json.dump(train_result.metrics, f, indent=2)

# train_result tiene info retornada por trainer.train()

# Fin del script para notebook. Ejecuta celda por celda y verifica outputs.
