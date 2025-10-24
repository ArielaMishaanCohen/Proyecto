"""
Proyecto: Fine-tuning de LLMs para Análisis de Noticias
Archivo: setup_project.py

Este script crea la estructura de carpetas y archivos necesarios
"""

import os
from pathlib import Path

def create_project_structure():
    """Crea la estructura de carpetas del proyecto"""
    
    folders = [
        "1_Data/raw",
        "1_Data/processed",
        "1_Data/train_test_splits",
        "2_Models/summarization",
        "2_Models/clustering",
        "2_Models/sentiment",
        "2_Models/checkpoints",
        "3_Results/metrics",
        "3_Results/visualizations",
        "3_Results/sample_outputs",
        "4_Notebooks",
        "5_Scripts",
        "6_Configs",
        "logs"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Creada carpeta: {folder}")
    
    print("\n✓ Estructura de proyecto creada exitosamente!")

if __name__ == "__main__":
    create_project_structure()
