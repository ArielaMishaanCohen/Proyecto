"""
Archivo: 5_Scripts/summarization_trainer.py

Fine-tuning de modelo para summarization de noticias

DECISIONES TÉCNICAS CLAVE:

1. MODELO BASE: BART-base (facebook/bart-base)
   ¿Por qué BART?
   - Diseñado específicamente para tareas seq2seq como summarization
   - Pre-entrenado con técnica de denoising (reconstruir texto corrupto)
   - Excelente balance entre calidad y velocidad
   - Tamaño manejable (140M parámetros) para fine-tuning local
   
   Alternativas consideradas:
   - T5: Más grande, requiere más recursos
   - PEGASUS: Específico para resumen pero menos flexible
   - GPT-2: Decoder-only, menos efectivo para summarization

2. TÉCNICA DE FINE-TUNING: Full Fine-tuning con LoRA opcional
   ¿Por qué?
   - Full fine-tuning: Actualiza todos los parámetros, mejor para dominio específico
   - LoRA (opcional): Reduce memoria, útil si recursos son limitados
   
3. OPTIMIZADOR: AdamW
   ¿Por qué?
   - Maneja weight decay correctamente
   - Estándar en fine-tuning de transformers
   - Convergencia estable

4. LEARNING RATE: 2e-5 (con warmup)
   ¿Por qué?
   - Típico para fine-tuning (no pre-training)
   - Warmup previene inestabilidad inicial
   - Permite adaptación gradual al dominio de noticias

5. BATCH SIZE: 4 (con gradient accumulation)
   ¿Por qué?
   - Balance entre memoria y estabilidad
   - Gradient accumulation simula batches más grandes
   - Permite entrenamiento en GPUs consumer
"""

import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import evaluate
from typing import Dict, List
import json

class SummarizationTrainer:
    """
    Clase para fine-tuning de modelo de summarization
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        output_dir: str = "2_Models/summarization",
        use_lora: bool = False
    ):
        """
        Inicializa el trainer.
        
        Args:
            model_name: Modelo base de HuggingFace
            output_dir: Directorio para guardar el modelo
            use_lora: Si usar LoRA para fine-tuning eficiente
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.use_lora = use_lora
        
        # Configurar device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando device: {self.device}")
        
        # Cargar modelo y tokenizer
        print(f"\nCargando modelo: {model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        if self.use_lora:
            self._setup_lora()
        
        self.model.to(self.device)
        
        # Métricas
        self.rouge = evaluate.load("rouge")
        
        # Configuración de entrenamiento
        self.config = {
            'max_input_length': 1024,  # Máximo de BART
            'max_target_length': 128,  # Resúmenes concisos
            'num_epochs': 3,  # Típico para fine-tuning
            'learning_rate': 2e-5,  # Estándar para transformers
            'batch_size': 4,
            'gradient_accumulation_steps': 4,  # Simula batch de 16
            'warmup_steps': 500,  # Warmup gradual
            'weight_decay': 0.01,  # Regularización
            'save_steps': 500,
            'eval_steps': 500,
            'logging_steps': 100,
        }
        
        self._log_config()
    
    def _setup_lora(self):
        """
        Configura LoRA para fine-tuning eficiente.
        
        ¿Qué es LoRA?
        - Low-Rank Adaptation: Añade matrices pequeñas adaptables
        - Solo entrena ~0.1% de parámetros
        - Reduce memoria dramáticamente
        - Mínima pérdida de calidad
        """
        from peft import get_peft_model, LoraConfig, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,  # Rank de matrices LoRA
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]  # Aplicar a attention
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("✓ LoRA configurado")
        self.model.print_trainable_parameters()
    
    def _log_config(self):
        """Registra configuración de entrenamiento"""
        print("\n" + "="*60)
        print("CONFIGURACIÓN DE FINE-TUNING")
        print("="*60)
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
    
    def load_data(self, csv_path: str) -> tuple:
        """
        Carga y divide datos.
        
        ¿Por qué 80/10/10?
        - 80% train: Suficiente para aprender patrones
        - 10% validation: Monitorear overfitting
        - 10% test: Evaluación final imparcial
        """
        df = pd.read_csv(csv_path)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        n = len(df)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Tokeniza ejemplos para el modelo.
        
        ¿Por qué estos parámetros?
        - truncation: Artículos largos se cortan, no fallan
        - padding: Batches eficientes
        - max_length: Límite del modelo BART
        """
        inputs = self.tokenizer(
            examples['input_text'],
            max_length=self.config['max_input_length'],
            truncation=True,
            padding='max_length'
        )
        
        targets = self.tokenizer(
            examples['target_summary'],
            max_length=self.config['max_target_length'],
            truncation=True,
            padding='max_length'
        )
        
        inputs['labels'] = targets['input_ids']
        return inputs
    
    def compute_metrics(self, eval_pred) -> Dict:
        """
        Calcula métricas ROUGE.
        
        ¿Qué es ROUGE?
        - Recall-Oriented Understudy for Gisting Evaluation
        - Mide overlap entre resumen generado y referencia
        - ROUGE-1: Unigrams (palabras individuales)
        - ROUGE-2: Bigrams (pares de palabras)
        - ROUGE-L: Longest common subsequence
        
        ¿Por qué ROUGE?
        - Estándar en evaluación de summarization
        - Correlaciona bien con evaluación humana
        - Rápido de calcular
        """
        predictions, labels = eval_pred
        
        # Decode predictions
        """
        
        for i, pred in enumerate(predictions):
            try:
                text = self.tokenizer.decode(pred, skip_special_tokens=True)
                decoded_preds.append(text)
            except Exception as e:
                print(f"❌ Error en índice {i}: {e}")
                print(f"Pred tokens: {pred}")"""

        #decoded_preds = self.tokenizer.batch_decode(
        #    predictions, skip_special_tokens=True
        #)
        
        decoded_preds = []
        for pred in predictions:
            # Filtrar -100 y None
            clean_pred = [t for t in pred if t is not None and t != -100]
            decoded_preds.append(
                self.tokenizer.decode(clean_pred, skip_special_tokens=True)
            )

        # Replace -100 in labels (padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        
        # Compute ROUGE
        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        return {
            'rouge1': result['rouge1'],
            'rouge2': result['rouge2'],
            'rougeL': result['rougeL']
        }
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Ejecuta el fine-tuning.
        
        PIPELINE DE ENTRENAMIENTO:
        1. Convertir DataFrames a Datasets de HuggingFace
        2. Tokenizar datos
        3. Configurar TrainingArguments
        4. Inicializar Trainer
        5. Entrenar con early stopping
        6. Guardar mejor modelo
        """
        print("\n" + "="*60)
        print("INICIANDO FINE-TUNING")
        print("="*60)
        
        # Convertir a Dataset
        train_dataset = Dataset.from_pandas(train_df[['input_text', 'target_summary']])
        val_dataset = Dataset.from_pandas(val_df[['input_text', 'target_summary']])
        
        # Tokenizar
        print("\nTokenizando datos...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['input_text', 'target_summary']
        )
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['input_text', 'target_summary']
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_strategy="steps",
            save_steps=self.config['save_steps'],
            learning_rate=self.config['learning_rate'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            num_train_epochs=self.config['num_epochs'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_steps=self.config['logging_steps'],
            predict_with_generate=True,  # Para métricas ROUGE
            generation_max_length=self.config['max_target_length'],
            load_best_model_at_end=True,  # Cargar mejor checkpoint
            metric_for_best_model="rougeL",  # Métrica para selección
            greater_is_better=True,
            save_total_limit=2,  # Solo 2 mejores checkpoints
            fp16=torch.cuda.is_available(),  # Mixed precision si hay GPU
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Entrenar
        print("\nIniciando entrenamiento...")
        train_result = trainer.train()
        
        # Guardar modelo final
        print("\nGuardando modelo...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        # Guardar métricas
        metrics = train_result.metrics
        with open(self.output_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("\n✓ Fine-tuning completado!")
        return trainer
    
    def evaluate(self, trainer, test_df: pd.DataFrame) -> Dict:
        """Evalúa en test set"""
        print("\n" + "="*60)
        print("EVALUACIÓN EN TEST SET")
        print("="*60)
        
        test_dataset = Dataset.from_pandas(test_df[['input_text', 'target_summary']])
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['input_text', 'target_summary']
        )
        
        metrics = trainer.evaluate(test_dataset)
        
        print("\nMétricas en Test Set:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Guardar
        with open(self.output_dir / "test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def generate_samples(self, test_df: pd.DataFrame, n_samples: int = 5):
        """
        Genera resúmenes de muestra para inspección cualitativa.
        
        ¿Por qué inspección manual?
        - ROUGE no captura todo (coherencia, factualidad)
        - Permite detectar problemas específicos
        - Valida que el modelo aprende el dominio
        """
        print("\n" + "="*60)
        print("GENERANDO MUESTRAS")
        print("="*60)
        
        self.model.to("cpu")
        samples = []
        
        for i in range(min(n_samples, len(test_df))):
            input_text = test_df.iloc[i]['input_text']
            target = test_df.iloc[i]['target_summary']
            
            # Generar resumen
            inputs = self.tokenizer(
                input_text,
                max_length=self.config['max_input_length'],
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=self.config['max_target_length'],
                num_beams=4,  # Beam search para mejor calidad
                early_stopping=True
            )
                        
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            sample = {
                'input': input_text[:500] + "...",
                'target_summary': target,
                'generated_summary': generated
            }
            samples.append(sample)
            
            print(f"\n--- Ejemplo {i+1} ---")
            print(f"Target: {target}")
            print(f"Generated: {generated}")
        
        # Guardar
        with open(self.output_dir / "sample_outputs.json", 'w') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        return samples


def main():
    """Pipeline completo de fine-tuning"""
    
    # Inicializar trainer
    trainer = SummarizationTrainer(
        model_name="facebook/bart-base",
        use_lora=False  # Cambiar a True si recursos limitados
    )
    
    # Cargar datos
    train_df, val_df, test_df = trainer.load_data(
        "1_Data/processed/summarization_data.csv"
    )
    
    # Entrenar
    trained_model = trainer.train(train_df, val_df)
    
    # Evaluar
    trainer.evaluate(trained_model, test_df)
    
    # Generar muestras
    trainer.generate_samples(test_df, n_samples=10)
    
    print("\n✓✓✓ Pipeline de summarization completado! ✓✓✓")


if __name__ == "__main__":
    main()