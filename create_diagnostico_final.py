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
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib
matplotlib.use('Agg')  # sin pantalla, guarda en archivo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json, time

spark = SparkSession.builder \\
    .appName("DiagnosticoFinal") \\
    .config("spark.driver.memory", "8g") \\
    .getOrCreate()

df = spark.read.csv("diabetes_v1_.csv", header=True, inferSchema=True)
print(f"Dataset cargado: {df.count()} filas, {len(df.columns)} columnas")"""))

# CELDA 2
cells.append(code_cell("""cat_cols = list(set([
    'race', 'gender', 'age', 'A1Cresult', 'max_glu_serum', 'change',
    'insulin', 'medical_specialty', 'readmitted', 'metformin',
    'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
    'glimepiride', 'repaglinide', 'nateglinide'
]))
num_cols = [
    'time_in_hospital', 'num_medications', 'num_lab_procedures',
    'num_procedures', 'number_diagnoses',
    'number_inpatient', 'number_outpatient', 'number_emergency'
]

df_clean = df
for c in cat_cols:
    df_clean = df_clean.withColumn(c, when((col(c)=='?')|col(c).isNull(),'Unknown').otherwise(col(c)))

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols]
assembler_full = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in cat_cols] + num_cols,
    outputCol="raw_features", handleInvalid="skip")

# MinMaxScaler
pipe_mm = Pipeline(stages=indexers + encoders + [assembler_full, MinMaxScaler(inputCol="raw_features", outputCol="scaledFeatures")])
df_mm = pipe_mm.fit(df_clean).transform(df_clean).cache()

# StandardScaler
pipe_std = Pipeline(stages=indexers + encoders + [assembler_full, StandardScaler(inputCol="raw_features", outputCol="scaledFeatures", withStd=True, withMean=False)])
df_std = pipe_std.fit(df_clean).transform(df_clean).cache()

# Solo numéricas
assembler_num = VectorAssembler(inputCols=num_cols, outputCol="raw_features", handleInvalid="skip")
pipe_num = Pipeline(stages=[assembler_num, MinMaxScaler(inputCol="raw_features", outputCol="scaledFeatures")])
df_num = pipe_num.fit(df_clean).transform(df_clean).cache()

print("Features preparadas correctamente.")"""))

# CELDA 3
cells.append(code_cell("""print(\"\\n=== VARIANZA EXPLICADA POR PCA ===\")
resultados_varianza = {}
for scaler_name, df_s in [("MinMaxScaler", df_mm), ("StandardScaler", df_std)]:
    pca30 = PCA(k=30, inputCol="scaledFeatures", outputCol="pca_tmp")
    var_acum = np.cumsum(pca30.fit(df_s).explainedVariance.toArray())
    resultados_varianza[scaler_name] = var_acum
    print(f\"\\n{scaler_name}:\")
    for k in [5, 10, 15, 20, 25, 30]:
        print(f"  k={k:2d} → varianza acumulada = {var_acum[k-1]:.4f} ({var_acum[k-1]*100:.1f}%)")

plt.figure(figsize=(10,4))
for name, var in resultados_varianza.items():
    plt.plot(range(1,31), var, marker='o', label=name)
plt.axhline(0.85, color='r', linestyle='--', label='85%')
plt.axhline(0.70, color='orange', linestyle='--', label='70%')
plt.title("Varianza Explicada Acumulada PCA")
plt.xlabel("k componentes"); plt.ylabel("Varianza acumulada")
plt.legend(); plt.grid()
plt.savefig("diagnostico_varianza_pca.png", bbox_inches='tight')
plt.close()
print(\"\\nGráfica guardada: diagnostico_varianza_pca.png\")"""))

# CELDA 4
cells.append(code_cell("""print(\"\\n=== BÚSQUEDA EXHAUSTIVA: Scaler x k_PCA x K_clusters x Algoritmo ===\")
print("Esto puede tardar 15-30 minutos. No interrumpir.\\n")

resultados_busqueda = []
mejor = {"sil": 0, "config": {}}

configs = [
    ("MinMax",    df_mm),
    ("Standard",  df_std),
    ("SoloNum",   df_num),
]

k_pca_valores  = [5, 10, 15, 20]
k_clus_valores = [2, 3, 4, 5, 6, 7]

for scaler_name, df_s in configs:
    for k_pca in k_pca_valores:
        pca_exp = PCA(k=k_pca, inputCol="scaledFeatures", outputCol="pcaF")
        df_pca = pca_exp.fit(df_s).transform(df_s).cache()
        ev = ClusteringEvaluator(predictionCol="prediction", featuresCol="pcaF", metricName="silhouette")

        for k_cl in k_clus_valores:
            for algo_name, AlgoClass in [("KMeans", KMeans), ("BisectingKMeans", BisectingKMeans)]:
                try:
                    t0 = time.time()
                    model = AlgoClass(featuresCol="pcaF", predictionCol="prediction", k=k_cl, seed=42).fit(df_pca)
                    preds = model.transform(df_pca)
                    sil = ev.evaluate(preds)
                    dur = round(time.time() - t0, 1)
                    row = {
                        "Scaler": scaler_name, "k_PCA": k_pca,
                        "Algoritmo": algo_name, "K": k_cl,
                        "Silhouette": round(sil, 4), "Tiempo_s": dur
                    }
                    resultados_busqueda.append(row)
                    print(f"  {scaler_name} | PCA={k_pca} | {algo_name} | K={k_cl} | Sil={sil:.4f} | {dur}s")
                    if sil > mejor["sil"]:
                        mejor = {"sil": sil, "config": row}
                except Exception as e:
                    print(f"  ERROR: {scaler_name} PCA={k_pca} {algo_name} K={k_cl} → {e}")

        df_pca.unpersist()

df_resultados = pd.DataFrame(resultados_busqueda).sort_values("Silhouette", ascending=False)
df_resultados.to_csv("diagnostico_resultados.csv", index=False)
print(f\"\\nResultados guardados en diagnostico_resultados.csv\")
print(f\"\\nTOP 10 MEJORES CONFIGURACIONES:\")
print(df_resultados.head(10).to_string(index=False))
print(f\"\\n>>> MEJOR CONFIGURACIÓN ENCONTRADA:\")
print(json.dumps(mejor["config"], indent=2))"""))

# CELDA 5
cells.append(code_cell("""for algo in ["KMeans", "BisectingKMeans"]:
    for scaler in ["MinMax", "Standard"]:
        sub = df_resultados[(df_resultados["Algoritmo"]==algo) & (df_resultados["Scaler"]==scaler)]
        if sub.empty:
            continue
        pivot = sub.pivot(index="k_PCA", columns="K", values="Silhouette")
        plt.figure(figsize=(8,4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"Silhouette — {algo} — {scaler}")
        fname = f"diagnostico_heatmap_{algo}_{scaler}.png"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"Guardado: {fname}")"""))

# CELDA 6
cells.append(code_cell("""mejor_config = mejor["config"]
mejor_sil    = mejor["sil"]

print(\"\\n\" + \"=\"*60)
print("RECOMENDACIÓN FINAL PARA EL NOTEBOOK PRINCIPAL")
print(\"=\"*60)
print(f\"\"\"
Mejor Silhouette encontrado : {mejor_sil:.4f}
Scaler a usar               : {mejor_config.get('Scaler')}
k_PCA a usar                : {mejor_config.get('k_PCA')}
Algoritmo principal         : {mejor_config.get('Algoritmo')}
K (número de clusters)      : {mejor_config.get('K')}

CAMBIOS A HACER EN diabetes_analisis_completo.ipynb:
1. En la celda de Preprocesamiento ML:
   - Cambiar el scaler a: {mejor_config.get('Scaler')}
   - Cambiar PCA k= a: {mejor_config.get('k_PCA')}
2. En los experimentos:
   - Usar K={mejor_config.get('K')} como punto de referencia
3. En la celda de verificación de silueta:
   - Eliminar el bloque if best_sil_km < 0.6 con re-ejecución automática
   - Si no se alcanza 0.6, documentar que en datasets clínicos con OHE
     este rango (0.3-0.4) es esperado y justificarlo académicamente

TODOS LOS ARCHIVOS DE RESULTADOS GENERADOS:
- diagnostico_varianza_pca.png
- diagnostico_resultados.csv
- diagnostico_heatmap_KMeans_MinMax.png
- diagnostico_heatmap_KMeans_Standard.png
- diagnostico_heatmap_BisectingKMeans_MinMax.png
- diagnostico_heatmap_BisectingKMeans_Standard.png
\"\"\")

# Guardar recomendación en JSON para referencia
with open("diagnostico_recomendacion.json", "w") as f:
    json.dump(mejor, f, indent=2)
print("Recomendación guardada en diagnostico_recomendacion.json")"""))

# Remove trailing newlines in each cell
for c in cells:
    c['source'][-1] = c['source'][-1].rstrip('\\n')

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

with open("diagnostico_final.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("diagnostico_final.ipynb creado exitosamente.")
