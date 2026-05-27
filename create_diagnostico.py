import json

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source.split('\n')]
    }

cells = []

# CELDA 1
cells.append(code_cell("""import os, sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws, regexp_replace, trim, lower, explode, split, length
from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

spark = SparkSession.builder \\
    .appName("Diagnostico") \\
    .config("spark.driver.memory", "8g") \\
    .getOrCreate()

df = spark.read.csv("diabetes_v1_.csv", header=True, inferSchema=True)
print(f"Filas: {df.count()}, Columnas: {len(df.columns)}")"""))

# CELDA 2
cells.append(code_cell("""# Preparar features base (igual que el notebook principal)
cat_cols = ['race', 'gender', 'age', 'A1Cresult', 'max_glu_serum', 'change',
            'insulin', 'medical_specialty', 'readmitted', 'metformin',
            'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
            'glimepiride', 'repaglinide', 'nateglinide']
cat_cols = list(set(cat_cols))
num_cols = ['time_in_hospital', 'num_medications', 'num_lab_procedures',
            'num_procedures', 'number_diagnoses',
            'number_inpatient', 'number_outpatient', 'number_emergency']

df_clean = df
for c in cat_cols:
    df_clean = df_clean.withColumn(c, when((col(c)=='?')|col(c).isNull(),'Unknown').otherwise(col(c)))

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols]
assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in cat_cols] + num_cols,
    outputCol="raw_features", handleInvalid="skip")
scaler_mm = MinMaxScaler(inputCol="raw_features", outputCol="scaledFeatures")

pipe = Pipeline(stages=indexers + encoders + [assembler, scaler_mm])
pipe_model = pipe.fit(df_clean)
df_scaled = pipe_model.transform(df_clean).cache()

# Probar PCA con k=5,10,15,20,30
print("Varianza acumulada por número de componentes PCA:")
pca_test = PCA(k=30, inputCol="scaledFeatures", outputCol="pca_test")
pca_model = pca_test.fit(df_scaled)
var_exp = np.cumsum(pca_model.explainedVariance.toArray())
for k in [5, 10, 15, 20, 25, 30]:
    print(f"  k={k:2d} componentes → varianza acumulada = {var_exp[k-1]:.4f}")

plt.figure(figsize=(10,4))
plt.plot(range(1,31), var_exp, marker='o')
plt.axhline(0.85, color='r', linestyle='--', label='85%')
plt.axhline(0.70, color='orange', linestyle='--', label='70%')
plt.title("Varianza Explicada Acumulada — MinMaxScaler")
plt.xlabel("Componentes PCA"); plt.ylabel("Varianza acumulada")
plt.legend(); plt.grid(); plt.show()"""))

# CELDA 3
cells.append(code_cell("""scaler_std = StandardScaler(inputCol="raw_features", outputCol="scaledFeatures", withStd=True, withMean=False)
pipe_std = Pipeline(stages=indexers + encoders + [assembler, scaler_std])
pipe_std_model = pipe_std.fit(df_clean)
df_scaled_std = pipe_std_model.transform(df_clean).cache()

pca_std = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures")
df_pca_mm  = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures").fit(df_scaled).transform(df_scaled).cache()
df_pca_std = pca_std.fit(df_scaled_std).transform(df_scaled_std).cache()

evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="pcaFeatures", metricName="silhouette")

results = []
for scaler_name, df_pca in [("MinMaxScaler", df_pca_mm), ("StandardScaler", df_pca_std)]:
    for k in [2, 3, 4, 5]:
        km = KMeans(featuresCol="pcaFeatures", predictionCol="prediction", k=k, seed=42)
        sil = evaluator.evaluate(km.fit(df_pca).transform(df_pca))
        results.append({"Scaler": scaler_name, "K": k, "Silhouette": round(sil,4)})
        print(f"{scaler_name} | K={k} | Silhouette={sil:.4f}")

print(pd.DataFrame(results).to_string(index=False))"""))

# CELDA 4
cells.append(code_cell("""# Solo variables numéricas, sin OHE
assembler_num = VectorAssembler(inputCols=num_cols, outputCol="raw_features", handleInvalid="skip")
scaler_num = MinMaxScaler(inputCol="raw_features", outputCol="scaledFeatures")
pipe_num = Pipeline(stages=[assembler_num, scaler_num])
df_num = pipe_num.fit(df_clean).transform(df_clean).cache()

# Sin PCA directo
evaluator_num = ClusteringEvaluator(predictionCol="prediction", featuresCol="scaledFeatures", metricName="silhouette")
print("Solo features numéricas (sin PCA):")
for k in [2, 3, 4, 5]:
    km = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=k, seed=42)
    sil = evaluator_num.evaluate(km.fit(df_num).transform(df_num))
    print(f"  K={k} | Silhouette={sil:.4f}")"""))

# CELDA 5
cells.append(code_cell("""# Reproducir el error exacto del notebook principal
# El evaluator tiene featuresCol="pcaFeatures" pero dataset_15 tiene "pcaFeatures_15"
# Este es el bug: el evaluator no se reconfigura al cambiar de columna
print("BUG IDENTIFICADO:")
print("El ClusteringEvaluator en el notebook principal tiene featuresCol='pcaFeatures'")
print("pero dataset_15 solo tiene la columna 'pcaFeatures_15'.")
print("Solución: crear un nuevo evaluator con featuresCol='pcaFeatures_15' para el fallback.")
print()

# Probar el fallback correcto
pca_15 = PCA(k=15, inputCol="scaledFeatures", outputCol="pcaFeatures_15")
df_pca_15 = pca_15.fit(df_scaled).transform(df_scaled).cache()

evaluator_15 = ClusteringEvaluator(predictionCol="prediction", featuresCol="pcaFeatures_15", metricName="silhouette")
print("Silueta con PCA k=15 (evaluator corregido):")
for k in [2, 3, 4, 5]:
    km = KMeans(featuresCol="pcaFeatures_15", predictionCol="prediction", k=k, seed=42)
    sil = evaluator_15.evaluate(km.fit(df_pca_15).transform(df_pca_15))
    print(f"  K={k} | Silhouette={sil:.4f}")"""))

# CELDA 6
cells.append(code_cell("""print("Búsqueda exhaustiva: Scaler x k_pca x K_clusters")
best = {"sil": 0}
resultados = []

for scaler_name, df_s in [("MinMax", df_scaled), ("Standard", df_scaled_std)]:
    for k_pca in [5, 10, 15, 20]:
        pca_exp = PCA(k=k_pca, inputCol="scaledFeatures", outputCol="pcaF")
        df_pca_exp = pca_exp.fit(df_s).transform(df_s).cache()
        ev = ClusteringEvaluator(predictionCol="prediction", featuresCol="pcaF", metricName="silhouette")
        for k_cl in [2, 3, 4, 5, 6]:
            km = KMeans(featuresCol="pcaF", predictionCol="prediction", k=k_cl, seed=42)
            sil = ev.evaluate(km.fit(df_pca_exp).transform(df_pca_exp))
            resultados.append({"Scaler": scaler_name, "k_PCA": k_pca, "K_clusters": k_cl, "Silhouette": round(sil,4)})
            if sil > best["sil"]:
                best = {"sil": sil, "scaler": scaler_name, "k_pca": k_pca, "k_cl": k_cl}
        df_pca_exp.unpersist()

df_res = pd.DataFrame(resultados).sort_values("Silhouette", ascending=False)
print(df_res.head(15).to_string(index=False))
print(f"\\n>>> MEJOR COMBINACIÓN: {best}")"""))

# CELDA 7
cells.append(code_cell("""print("=" * 60)
print("RESUMEN DE DIAGNÓSTICO")
print("=" * 60)
print(\"\"\"
1. VARIANZA PCA: Revisar cuánta varianza captura k=10 vs k=15.
   Si con k=10 solo se captura <50% de varianza, los clusters
   están comprimidos en espacio insuficiente.

2. BUG CONFIRMADO en notebook principal: el evaluator del fallback
   PCA k=15 usa featuresCol='pcaFeatures' pero el dataset tiene
   'pcaFeatures_15' → IllegalArgumentException.
   FIX: crear evaluator_15 con featuresCol='pcaFeatures_15'.

3. MEJOR CONFIGURACIÓN ENCONTRADA (ver Celda 6):
   Usar los valores del dict 'best' de arriba para actualizar
   el notebook principal con el scaler, k_pca y K_clusters óptimos.

4. NOTA SOBRE 0.6: En datasets clínicos tabulares con muchas
   variables categóricas codificadas con OHE, siluetas de 0.3-0.4
   son normales. Si la búsqueda exhaustiva no supera 0.6, documentar
   la mejor configuración encontrada y justificarlo en el notebook.
\"\"\")"""))

# Cleanup trailling newlines to be exact
for c in cells:
    c['source'][-1] = c['source'][-1].rstrip('\n')

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("diagnostico.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("diagnostico.ipynb creado exitosamente.")
