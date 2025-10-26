"""
SCRIPT DE DEBUG PARA EL OPTIMIZER
==================================

Este script prueba cada paso del optimizer para ver dónde falla
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Prueba 1: Imports"""
    print("\n" + "="*70)
    print("[TEST 1] Verificando imports...")
    print("="*70)
    
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except Exception as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"✓ sentence-transformers")
    except Exception as e:
        print(f"❌ sentence-transformers: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except Exception as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        from sklearn.cluster import HDBSCAN
        print(f"✓ hdbscan")
    except Exception as e:
        print(f"❌ hdbscan: {e}")
        print(f"   → Instala: pip install hdbscan")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print(f"✓ sklearn")
    except Exception as e:
        print(f"❌ sklearn: {e}")
        return False
    
    print("\n✅ Todos los imports OK")
    return True


def test_data_loading():
    """Prueba 2: Carga de datos"""
    print("\n" + "="*70)
    print("[TEST 2] Verificando carga de datos...")
    print("="*70)
    
    data_path = "1_Data/processed/clustering_data_cleaned.csv"
    
    if not Path(data_path).exists():
        print(f"❌ Archivo no encontrado: {data_path}")
        print(f"\n💡 SOLUCIÓN:")
        print(f"   1. Ejecuta primero: python data_cleaner.py")
        print(f"   2. O verifica la ruta del archivo")
        return False, None
    
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"✓ Datos cargados: {len(df)} documentos")
        
        # Verificar columnas
        required_cols = ['text', 'title', 'source']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"❌ Columnas faltantes: {missing}")
            print(f"   Columnas disponibles: {df.columns.tolist()}")
            return False, None
        
        print(f"✓ Columnas requeridas presentes")
        
        # Verificar que no hay nulos
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            print(f"⚠️  Valores nulos: {null_counts.to_dict()}")
        else:
            print(f"✓ No hay valores nulos")
        
        print(f"\n✅ Datos OK")
        return True, df
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        traceback.print_exc()
        return False, None


def test_model_loading():
    """Prueba 3: Carga de modelo"""
    print("\n" + "="*70)
    print("[TEST 3] Verificando carga de modelo...")
    print("="*70)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Cargando modelo: {model_name}")
        
        model = SentenceTransformer(model_name)
        print(f"✓ Modelo cargado")
        
        # Test encoding
        test_text = "This is a test sentence"
        embedding = model.encode([test_text])
        print(f"✓ Embedding generado: shape {embedding.shape}")
        
        print(f"\n✅ Modelo OK")
        return True, model
        
    except Exception as e:
        print(f"❌ Error con modelo: {e}")
        traceback.print_exc()
        return False, None


def test_tfidf_similarity(df):
    """Prueba 4: TF-IDF similarity"""
    print("\n" + "="*70)
    print("[TEST 4] Verificando TF-IDF similarity...")
    print("="*70)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Sample pequeño para rapidez
        sample_df = df.head(50)
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        print(f"Calculando TF-IDF para {len(sample_df)} documentos...")
        tfidf_matrix = vectorizer.fit_transform(sample_df['text'])
        print(f"✓ TF-IDF matrix: {tfidf_matrix.shape}")
        
        print(f"Calculando similitudes...")
        similarity_matrix = cosine_similarity(tfidf_matrix)
        print(f"✓ Similarity matrix: {similarity_matrix.shape}")
        
        # Estadísticas
        import numpy as np
        mask = ~np.eye(len(sample_df), dtype=bool)
        similarities = similarity_matrix[mask]
        
        print(f"\n📊 Estadísticas de similitud:")
        print(f"  Promedio: {similarities.mean():.3f}")
        print(f"  Mediana: {np.median(similarities):.3f}")
        print(f"  Max: {similarities.max():.3f}")
        
        if similarities.mean() < 0.05:
            print(f"\n⚠️  ADVERTENCIA: Similitud muy baja ({similarities.mean():.3f})")
            print(f"   Clustering será difícil con datos tan diversos")
        
        print(f"\n✅ TF-IDF OK")
        return True, similarity_matrix
        
    except Exception as e:
        print(f"❌ Error en TF-IDF: {e}")
        traceback.print_exc()
        return False, None


def test_training_pairs(df):
    """Prueba 5: Creación de pares"""
    print("\n" + "="*70)
    print("[TEST 5] Verificando creación de pares...")
    print("="*70)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import InputExample
        import numpy as np
        
        # Sample
        sample_df = df.head(50).reset_index(drop=True)
        
        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sample_df['text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Crear pares positivos
        threshold = 0.3
        positive_pairs = []
        
        for i in range(len(sample_df)):
            similar_indices = np.where(
                (similarity_matrix[i] > threshold) & 
                (similarity_matrix[i] < 0.99)
            )[0]
            
            for j in similar_indices[:2]:  # Max 2 pares
                if i < j:
                    positive_pairs.append((i, j, similarity_matrix[i, j]))
        
        print(f"✓ Pares positivos: {len(positive_pairs)}")
        
        if len(positive_pairs) == 0:
            print(f"\n⚠️  NO SE ENCONTRARON PARES POSITIVOS")
            print(f"   Threshold muy alto ({threshold}) o documentos muy diferentes")
            print(f"   Probando con threshold más bajo...")
            
            # Probar threshold más bajo
            threshold = 0.15
            for i in range(len(sample_df)):
                similar_indices = np.where(
                    (similarity_matrix[i] > threshold) & 
                    (similarity_matrix[i] < 0.99)
                )[0]
                
                for j in similar_indices[:2]:
                    if i < j:
                        positive_pairs.append((i, j, similarity_matrix[i, j]))
            
            print(f"   Con threshold={threshold}: {len(positive_pairs)} pares")
            
            if len(positive_pairs) == 0:
                print(f"\n❌ PROBLEMA CRÍTICO: Documentos demasiado diferentes")
                print(f"   Clustering no será efectivo con estos datos")
                return False, None
        
        # Crear InputExamples
        examples = []
        for i, j, sim in positive_pairs[:10]:  # Solo 10 para test
            example = InputExample(
                texts=[sample_df.iloc[i]['text'], sample_df.iloc[j]['text']],
                label=float(sim)
            )
            examples.append(example)
        
        print(f"✓ InputExamples creados: {len(examples)}")
        
        print(f"\n✅ Pares OK")
        return True, examples
        
    except Exception as e:
        print(f"❌ Error creando pares: {e}")
        traceback.print_exc()
        return False, None


def test_training(model, examples):
    """Prueba 6: Training (mini)"""
    print("\n" + "="*70)
    print("[TEST 6] Verificando training...")
    print("="*70)
    
    try:
        from sentence_transformers import losses
        from torch.utils.data import DataLoader
        
        print(f"Creando dataloader con {len(examples)} ejemplos...")
        dataloader = DataLoader(examples, batch_size=4, shuffle=True)
        print(f"✓ DataLoader creado")
        
        print(f"Configurando loss...")
        train_loss = losses.MultipleNegativesRankingLoss(model)
        print(f"✓ Loss configurado")
        
        print(f"\n⚠️  No ejecutaremos training real (toma tiempo)")
        print(f"   Pero la configuración es correcta")
        
        print(f"\n✅ Training setup OK")
        return True
        
    except Exception as e:
        print(f"❌ Error en training setup: {e}")
        traceback.print_exc()
        return False


def main():
    """Ejecuta todos los tests"""
    print("\n" + "="*70)
    print(" "*15 + "🔍 DIAGNÓSTICO DEL OPTIMIZER")
    print("="*70)
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n❌ FALLO EN IMPORTS - Instala dependencias faltantes")
        return
    
    # Test 2: Data
    results['data'], df = test_data_loading()
    if not results['data']:
        print("\n❌ FALLO EN DATOS - Ejecuta data_cleaner.py primero")
        return
    
    # Test 3: Model
    results['model'], model = test_model_loading()
    if not results['model']:
        print("\n❌ FALLO EN MODELO - Verifica instalación de sentence-transformers")
        return
    
    # Test 4: TF-IDF
    results['tfidf'], sim_matrix = test_tfidf_similarity(df)
    if not results['tfidf']:
        print("\n❌ FALLO EN TF-IDF")
        return
    
    # Test 5: Pairs
    results['pairs'], examples = test_training_pairs(df)
    if not results['pairs']:
        print("\n❌ FALLO EN PARES - Datos muy diversos para clustering")
        return
    
    # Test 6: Training
    results['training'] = test_training(model, examples)
    
    # Resumen
    print("\n" + "="*70)
    print(" "*20 + "📊 RESUMEN DE TESTS")
    print("="*70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name.upper()}")
    
    if all_passed:
        print("\n" + "="*70)
        print("✅✅✅ TODOS LOS TESTS PASARON ✅✅✅")
        print("="*70)
        print("\nEl optimizer DEBERÍA funcionar.")
        print("Si no muestra output, puede ser:")
        print("  1. El script está corriendo pero muy lento")
        print("  2. Hay un error silencioso (ver logs)")
        print("  3. Output está siendo redirigido")
        print("\n💡 PRUEBA EJECUTAR:")
        print("   python clustering_optimizer.py 2>&1 | tee optimizer_log.txt")
    else:
        print("\n" + "="*70)
        print("❌ ALGUNOS TESTS FALLARON")
        print("="*70)
        print("\nRevisa los errores arriba y corrígelos antes de ejecutar el optimizer")


if __name__ == "__main__":
    main()