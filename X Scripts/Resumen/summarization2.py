"""
Archivo: 5_Scripts/summarization_trainer.py

Fine-tuning de modelo para summarization de noticias
Versión final adaptada para BART sin prompts en el input.
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
from typing import Dict
import json

class SummarizationTrainer:
    """Clase para fine-tuning de modelo de summarization"""

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        output_dir: str = "2_Models/summarization",
        use_lora: bool = False
    ):
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
            'max_input_length': 1024,
            'max_target_length': 128,
            'num_epochs': 3,
            'learning_rate': 2e-5,
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'save_steps': 500,
            'eval_steps': 500,
            'logging_steps': 100,
        }

        self._log_config()

    def _setup_lora(self):
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.model, lora_config)
        print("✓ LoRA configurado")
        self.model.print_trainable_parameters()

    def _log_config(self):
        print("\n" + "="*60)
        print("CONFIGURACIÓN DE FINE-TUNING")
        print("="*60)
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")

    def load_data(self, csv_path: str) -> tuple:
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokeniza ejemplos para el modelo, sin prompt."""
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
        predictions, labels = eval_pred
        decoded_preds = []
        for pred in predictions:
            clean_pred = [t for t in pred if t is not None and t != -100]
            decoded_preds.append(self.tokenizer.decode(clean_pred, skip_special_tokens=True))

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

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
        print("\n" + "="*60)
        print("INICIANDO FINE-TUNING")
        print("="*60)

        train_dataset = Dataset.from_pandas(train_df[['input_text', 'target_summary']])
        val_dataset = Dataset.from_pandas(val_df[['input_text', 'target_summary']])

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
            predict_with_generate=True,
            generation_max_length=self.config['max_target_length'],
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        print("\nIniciando entrenamiento...")
        train_result = trainer.train()

        print("\nGuardando modelo...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))

        metrics = train_result.metrics
        with open(self.output_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        print("\n✓ Fine-tuning completado!")
        return trainer

    def evaluate(self, trainer, test_df: pd.DataFrame) -> Dict:
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

        with open(self.output_dir / "test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def generate_samples(self, test_df: pd.DataFrame, n_samples: int = 5):
        print("\n" + "="*60)
        print("GENERANDO MUESTRAS")
        print("="*60)

        self.model.to("cpu")
        samples = []

        for i in range(min(n_samples, len(test_df))):
            input_text = test_df.iloc[i]['input_text']
            target = test_df.iloc[i]['target_summary']

            inputs = self.tokenizer(
                input_text,
                max_length=self.config['max_input_length'],
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=self.config['max_target_length'],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2
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

        with open(self.output_dir / "sample_outputs.json", 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        return samples


def main():
    trainer = SummarizationTrainer(
        model_name="facebook/bart-base",
        use_lora=False
    )

    train_df, val_df, test_df = trainer.load_data(
        "1_Data/processed/summarization_data.csv"
    )

    trained_model = trainer.train(train_df, val_df)
    trainer.evaluate(trained_model, test_df)
    trainer.generate_samples(test_df, n_samples=10)

    print("\n✓✓✓ Pipeline de summarization completado! ✓✓✓")


if __name__ == "__main__":
    main() 