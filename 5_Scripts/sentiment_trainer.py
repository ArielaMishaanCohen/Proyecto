"""
Archivo: 5_Scripts/sentiment_trainer.py

Fine-tuning para sentiment analysis de noticias

DECISIONES TÉCNICAS CLAVE:

1. MODELO BASE: distilbert-base-uncased
   ¿Por qué DistilBERT?
   - 40% más pequeño que BERT, 60% más rápido
   - Retiene 97% del performance de BERT
   - Destilación: Estudiante aprende de BERT (teacher)
   - Excelente para clasificación de texto
   
   Alternativas consideradas:
   - BERT-base: Más grande, más lento, mínima mejora
   - RoBERTa: Excelente pero overkill para 3 clases
   - TinyBERT: Demasiado pequeño, pierde capacidad

2. ARQUITECTURA: Clasificación con pooling de [CLS]
   ¿Cómo funciona?
   - Token [CLS] al inicio: Representa toda la secuencia
   - Pooling del hidden state de [CLS]
   - Linear layer: 768 dims → 3 classes
   - Softmax: Probabilidades para (positive, negative, neutral)

3. LOSS FUNCTION: CrossEntropyLoss con class weights
   ¿Por qué class weights?
   - Datos probablemente desbalanceados (más neutrales)
   - Weights compensan: Penaliza más errores en clases minoritarias
   - Evita bias hacia clase mayoritaria

4. MÉTRICAS: Accuracy, F1-Score (macro/weighted), Confusion Matrix
   ¿Por qué estas métricas?
   - Accuracy: Métrica general, intuitiva
   - F1-macro: Promedio no ponderado, trata clases igual
   - F1-weighted: Pondera por frecuencia de clase
   - Confusion Matrix: Muestra errores específicos (pos→neg, etc)

5. TÉCNICAS DE MEJORA:
   - Learning rate scheduling: Decae con cosine annealing
   - Gradient clipping: Previene exploding gradients
   - Dropout: 0.1 para regularización
   - Early stopping: Para en validation loss mínimo
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import json

class SentimentTrainer:
    """
    Fine-tuning para análisis de sentimiento
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dir: str = "2_Models/sentiment"
    ):
        """
        Inicializa sentiment trainer.
        
        Args:
            model_name: Modelo base de HuggingFace
            output_dir: Directorio para guardar modelo
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando device: {self.device}")
        
        # Labels
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Cargar modelo y tokenizer
        print(f"\nCargando modelo: {model_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # 3 clases de sentimiento
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)
        
        # Configuración
        self.config = {
            'max_length': 256,  # Más corto, suficiente para sentiment
            'batch_size': 16,
            'num_epochs': 5,  # Más epochs para clasificación
            'learning_rate': 3e-5,  # Ligeramente más alto
            'warmup_ratio': 0.1,  # 10% de steps
            'weight_decay': 0.01,
            'gradient_clip': 1.0,  # Clip gradients
            'early_stopping_patience': 3,
            'eval_steps': 200,
            'save_steps': 200,
            'logging_steps': 50,
        }
        
        self._log_config()
    
    def _log_config(self):
        """Log configuración"""
        print("\n" + "="*60)
        print("CONFIGURACIÓN DE SENTIMENT ANALYSIS")
        print("="*60)
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
    
    def load_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carga y divide datos con estratificación.
        
        ¿Por qué estratificación?
        - Mantiene proporción de clases en train/val/test
        - Crítico con datos desbalanceados
        - Asegura representación en validation/test
        """
        df = pd.read_csv(csv_path)
        
        # Combinar texto para clasificación
        df['text'] = df['title'].fillna('') + '. ' + df['description'].fillna('')
        
        # Convertir sentiments a IDs
        df['label'] = df['sentiment'].map(self.label2id)
        
        # Mostrar distribución
        print("\nDistribución de sentiments:")
        print(df['sentiment'].value_counts())
        print(df['sentiment'].value_counts(normalize=True))
        
        # Split estratificado
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['label']
        )
        
        print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def compute_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """
        Calcula class weights para loss balanceado.
        
        ¿Cómo funciona?
        - Weight inversamente proporcional a frecuencia
        - Clase rara → weight alto → penalización mayor por error
        - Formula: n_samples / (n_classes * n_samples_class)
        """
        labels = train_df['label'].values
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        print("\nClass weights:")
        for label_id, weight in enumerate(class_weights):
            label_name = self.id2label[label_id]
            print(f"  {label_name}: {weight:.4f}")
        
        return class_weights_tensor
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Tokeniza ejemplos.
        
        PARÁMETROS CLAVE:
        - truncation: Corta textos largos (no falla)
        - padding: Todos mismo length en batch (eficiencia)
        - max_length: 256 es suficiente para captar sentiment
        """
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length']
        )
    
    def compute_metrics(self, eval_pred) -> Dict:
        """
        Calcula métricas de clasificación.
        
        MÉTRICAS EXPLICADAS:
        
        1. Accuracy: % correctas
           - Simple, intuitiva
           - Puede ser engañosa con clases desbalanceadas
        
        2. F1-Score Macro:
           - Promedio de F1 por clase
           - Trata todas las clases igual
           - Mejor para clases desbalanceadas
        
        3. F1-Score Weighted:
           - F1 ponderado por frecuencia
           - Refleja performance en datos reales
        
        4. Per-class metrics:
           - Precision: De los predichos X, cuántos son X
           - Recall: De los reales X, cuántos detectamos
           - F1: Balance de precision y recall
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Métricas generales
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        # Per-class F1
        f1_per_class = f1_score(labels, predictions, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
        }
        
        # Añadir F1 por clase
        for label_id, f1 in enumerate(f1_per_class):
            label_name = self.id2label[label_id]
            metrics[f'f1_{label_name}'] = f1
        
        return metrics
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        class_weights: torch.Tensor
    ):
        """
        Ejecuta fine-tuning con loss ponderado.
        
        COMPONENTES DEL TRAINING:
        
        1. Custom Trainer con weighted loss
           - Sobrescribe compute_loss
           - Aplica class weights al CrossEntropyLoss
        
        2. Learning rate scheduler
           - Cosine annealing: LR decae suavemente
           - Warmup: Aumenta gradualmente al inicio
           - Evita overfitting al final
        
        3. Early stopping
           - Monitorea validation loss
           - Para si no mejora en N epochs
           - Previene overfitting
        
        4. Gradient clipping
           - Limita norma de gradients
           - Previene exploding gradients
           - Estabiliza entrenamiento
        """
        print("\n" + "="*60)
        print("INICIANDO FINE-TUNING DE SENTIMENT")
        print("="*60)
        
        # Convertir a Dataset
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
        
        # Tokenizar
        print("\nTokenizando datos...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['text']
        )
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Custom Trainer con weighted loss
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
            
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Weighted cross entropy loss
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits, labels)
                
                return (loss, outputs) if return_outputs else loss
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_strategy="steps",
            save_steps=self.config['save_steps'],
            learning_rate=self.config['learning_rate'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            num_train_epochs=self.config['num_epochs'],
            warmup_ratio=self.config['warmup_ratio'],
            weight_decay=self.config['weight_decay'],
            logging_steps=self.config['logging_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            max_grad_norm=self.config['gradient_clip'],
            lr_scheduler_type="cosine",  # Cosine annealing
        )
        
        # Trainer
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['early_stopping_patience']
                )
            ]
        )
        
        # Entrenar
        print("\nIniciando entrenamiento...")
        train_result = trainer.train()
        
        # Guardar
        print("\nGuardando modelo...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        # Guardar métricas
        with open(self.output_dir / "training_metrics.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        print("\n✓ Fine-tuning completado!")
        return trainer
    
    def evaluate(self, trainer, test_df: pd.DataFrame) -> Dict:
        """
        Evaluación completa en test set.
        
        Incluye:
        - Métricas numéricas
        - Classification report detallado
        - Confusion matrix
        """
        print("\n" + "="*60)
        print("EVALUACIÓN EN TEST SET")
        print("="*60)
        
        # Preparar test dataset
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Predecir
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Métricas
        metrics = predictions.metrics
        
        print("\nMétricas en Test Set:")
        for key, value in metrics.items():
            if key.startswith('test_'):
                print(f"{key}: {value:.4f}")
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(
            true_labels,
            pred_labels,
            target_names=list(self.label2id.keys()),
            digits=4
        )
        print(report)
        
        # Guardar report
        report_dict = classification_report(
            true_labels,
            pred_labels,
            target_names=list(self.label2id.keys()),
            output_dict=True
        )
        
        with open(self.output_dir / "classification_report.json", 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Confusion matrix
        self._plot_confusion_matrix(true_labels, pred_labels)
        
        # Guardar métricas
        with open(self.output_dir / "test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _plot_confusion_matrix(self, true_labels: np.ndarray, pred_labels: np.ndarray):
        """
        Grafica confusion matrix.
        
        ¿Qué muestra?
        - Filas: Etiquetas reales
        - Columnas: Etiquetas predichas
        - Diagonal: Predicciones correctas
        - Off-diagonal: Errores específicos
        
        Análisis útil:
        - ¿Confunde positive con neutral?
        - ¿Bias hacia alguna clase?
        - ¿Errores simétricos o asimétricos?
        """
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.label2id.keys()),
            yticklabels=list(self.label2id.keys()),
            cbar_kws={'label': 'Frecuencia'}
        )
        plt.xlabel('Predicción', fontsize=12)
        plt.ylabel('Real', fontsize=12)
        plt.title('Confusion Matrix - Sentiment Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300)
        print("\n✓ Confusion matrix guardada")
    
    def analyze_errors(self, trainer, test_df: pd.DataFrame, n_examples: int = 10):
        """
        Analiza ejemplos mal clasificados.
        
        ¿Por qué analizar errores?
        - Identifica patrones de fallo
        - Guía mejoras (más datos, features, etc)
        - Detecta bias o problemas sistemáticos
        """
        print("\n" + "="*60)
        print("ANÁLISIS DE ERRORES")
        print("="*60)
        
        # Preparar dataset
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Predecir
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Probabilidades
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
        
        # Encontrar errores
        errors = []
        for idx, (true, pred, prob) in enumerate(zip(true_labels, pred_labels, probs)):
            if true != pred:
                errors.append({
                    'text': test_df.iloc[idx]['text'],
                    'true_label': self.id2label[true],
                    'pred_label': self.id2label[pred],
                    'confidence': float(prob[pred]),
                    'true_prob': float(prob[true])
                })
        
        # Ordenar por confianza (errores con alta confianza son peores)
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nTotal de errores: {len(errors)} / {len(test_df)} ({100*len(errors)/len(test_df):.2f}%)")
        print(f"\nTop {n_examples} errores con mayor confianza:")
        
        for i, error in enumerate(errors[:n_examples], 1):
            print(f"\n{i}. Texto: {error['text'][:100]}...")
            print(f"   Real: {error['true_label']}, Predicho: {error['pred_label']}")
            print(f"   Confianza: {error['confidence']:.4f}")
        
        # Guardar análisis completo
        with open(self.output_dir / "error_analysis.json", 'w') as f:
            json.dump(errors[:50], f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Análisis completo guardado (top 50 errores)")
    
    def predict_sentiment(self, text: str) -> Dict:
        """
        Predice sentiment de un texto nuevo.
        
        Útil para:
        - Testing interactivo
        - Deployment en producción
        - Validación manual
        """
        # Tokenizar
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config['max_length'],
            return_tensors="pt"
        ).to(self.device)
        
        # Predecir
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
        
        # Resultado
        pred_label_id = torch.argmax(probs).item()
        pred_label = self.id2label[pred_label_id]
        confidence = probs[pred_label_id].item()
        
        result = {
            'text': text,
            'sentiment': pred_label,
            'confidence': confidence,
            'probabilities': {
                self.id2label[i]: float(probs[i])
                for i in range(len(probs))
            }
        }
        
        return result


def main():
    """Pipeline completo de sentiment analysis"""
    
    # Inicializar trainer
    trainer = SentimentTrainer()
    
    # Cargar datos
    train_df, val_df, test_df = trainer.load_data(
        "1_Data/processed/sentiment_data.csv"
    )
    
    # Calcular class weights
    class_weights = trainer.compute_class_weights(train_df)
    
    # Entrenar
    trained_model = trainer.train(train_df, val_df, class_weights)
    
    # Evaluar
    trainer.evaluate(trained_model, test_df)
    
    # Analizar errores
    trainer.analyze_errors(trained_model, test_df, n_examples=15)
    
    # Ejemplo de predicción
    print("\n" + "="*60)
    print("EJEMPLO DE PREDICCIÓN")
    print("="*60)
    
    example_texts = [
        "Company stocks soar to record highs after strong earnings report",
        "Devastating earthquake causes massive destruction and casualties",
        "The weather today is partly cloudy with mild temperatures"
    ]
    
    for text in example_texts:
        result = trainer.predict_sentiment(text)
        print(f"\nTexto: {text}")
        print(f"Sentiment: {result['sentiment']} (confianza: {result['confidence']:.4f})")
    
    print("\n✓✓✓ Pipeline de sentiment analysis completado! ✓✓✓")


if __name__ == "__main__":
    main()