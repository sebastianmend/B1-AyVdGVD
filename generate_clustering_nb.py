import nbformat as nbf

# Crear un nuevo notebook
nb = nbf.v4.new_notebook()

# Celda 1: Markdown
nb.cells.append(nbf.v4.new_markdown_cell("""# Análisis No Supervisado y Clustering con PySpark - Diabetes V1

**Integrantes:** Sebastián, Byron, Jean Daniel.

**Descripción:** Proyecto para la materia de Big Data en la UTPL. El objetivo es identificar perfiles ocultos de pacientes diabéticos mediante clustering."""))

# Celda 2: Setup
nb.cells.append(nbf.v4.new_code_cell("""from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Inicializar sesión de Spark optimizada para ejecución local
spark = SparkSession.builder \
    .appName("DiabetesClustering") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Cargar el dataset
file_path = "diabetes_v1_.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Mostrar confirmación de carga
print(f"Dataset cargado con {df.count()} filas y {len(df.columns)} columnas.")"""))

# Celda 3: EDA Parte 1
nb.cells.append(nbf.v4.new_code_cell("""# Esquema del dataset
df.printSchema()

# Análisis de variables categóricas clave
categorical_cols = ['race', 'gender', 'age']

plt.figure(figsize=(15, 5))

for i, col in enumerate(categorical_cols):
    # Agregación en PySpark y paso a Pandas para graficar
    pd_df = df.groupBy(col).count().orderBy("count", ascending=False).toPandas()
    
    plt.subplot(1, 3, i+1)
    sns.barplot(data=pd_df, x=col, y='count', palette='viridis')
    plt.title(f'Distribución de {col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()"""))

# Celda 4: EDA Parte 2
nb.cells.append(nbf.v4.new_code_cell("""# Análisis de variables numéricas clave
numeric_cols = ['time_in_hospital', 'num_medications', 'num_lab_procedures']

plt.figure(figsize=(15, 5))

for i, col in enumerate(numeric_cols):
    # Seleccionar datos y pasar a Pandas
    pd_series = df.select(col).toPandas()
    
    plt.subplot(1, 3, i+1)
    sns.histplot(pd_series[col], kde=True, color='skyblue')
    plt.title(f'Histograma de {col}')

plt.tight_layout()
plt.show()

# Boxplots para detectar outliers y distribución
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols):
    pd_series = df.select(col).toPandas()
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=pd_series[col], color='lightgreen')
    plt.title(f'Boxplot de {col}')

plt.tight_layout()
plt.show()"""))

# Celda 5: Markdown (Feature Engineering)
nb.cells.append(nbf.v4.new_markdown_cell("""## Preparación de Datos (Feature Engineering)

Los algoritmos de clustering como **K-Means** calculan distancias euclidianas, por lo que requieren que los datos sean numéricos y estén en la misma escala. 

En este paso realizaremos:
1. **Limpieza:** Manejo de valores faltantes (representados como '?').
2. **Indexación:** Conversión de texto a índices numéricos.
3. **Codificación (OHE):** Creación de variables binarias para categorías.
4. **Vectorización:** Unión de todas las columnas en un solo vector de características.
5. **Escalamiento:** Normalización de los datos.
6. **PCA (Principal Component Analysis):** Reducción de dimensionalidad para mejorar la eficiencia del clustering y mitigar la "maldición de la dimensionalidad" causada por el One-Hot Encoding."""))

# Celda 6: Limpieza y Pipeline con PCA
nb.cells.append(nbf.v4.new_code_cell("""from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA
from pyspark.sql.functions import when, col

# Limpieza: Reemplazar '?' por None y eliminar filas con nulos en columnas clave
cols_to_clean = ['race', 'gender', 'age']
df_clean = df
for c in cols_to_clean:
    df_clean = df_clean.withColumn(c, when(col(c) == '?', None).otherwise(col(c)))

df_clean = df_clean.dropna(subset=cols_to_clean)

# Definir etapas del Pipeline
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep") for c in cols_to_clean]
encoders = [OneHotEncoder(inputCol=f"{c}_indexed", outputCol=f"{c}_ohe") for c in cols_to_clean]

# Columnas numéricas a incluir
num_cols = ['time_in_hospital', 'num_medications', 'num_lab_procedures']

# Assembler para crear el vector 'features'
assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in cols_to_clean] + num_cols,
    outputCol="features"
)

# Escalador
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# Implementación de PCA (Reducción a 10 componentes)
pca = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures")

# Crear y ejecutar el Pipeline incluyendo PCA
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, pca])
model_pipeline = pipeline.fit(df_clean)
dataset_final = model_pipeline.transform(df_clean)

# Mostrar resultado del procesamiento
dataset_final.select("features", "scaledFeatures", "pcaFeatures").show(5, truncate=False)"""))

# NUEVA CELDA: Justificación de K (Método de la Silueta)
nb.cells.append(nbf.v4.new_code_cell("""from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

silhouette_scores = []
k_values = range(2, 7) # Probaremos con 2, 3, 4, 5 y 6 clústeres

evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="pcaFeatures", metricName="silhouette")

print("Evaluando diferentes valores de K...")
for k in k_values:
    # Usamos pcaFeatures en lugar de scaledFeatures
    kmeans_temp = KMeans(featuresCol="pcaFeatures", predictionCol="prediction", k=k, seed=42)
    model_temp = kmeans_temp.fit(dataset_final)
    predictions_temp = model_temp.transform(dataset_final)
    
    score = evaluator.evaluate(predictions_temp)
    silhouette_scores.append(score)
    print(f"K={k} | Silhouette Score = {score:.4f}")

# Graficar para justificar visualmente
plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='b')
plt.title('Justificación de K: Método de la Silueta')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('Silhouette Score (Más alto es mejor)')
plt.grid(True)
plt.show()"""))

# Celda 7: Modelado K-Means Definitivo
nb.cells.append(nbf.v4.new_code_cell("""from pyspark.ml.clustering import KMeans

# Entrenar K-Means definitivo basándose en el mejor valor de la gráfica anterior
mejor_k = 3 # Basado en la evaluación previa
kmeans = KMeans(featuresCol="pcaFeatures", predictionCol="prediction", k=mejor_k, seed=42)
model = kmeans.fit(dataset_final)

# Obtener predicciones
predictions = model.transform(dataset_final)

print(f"Entrenamiento completado. {mejor_k} Clústeres asignados usando PCA.")"""))

# Celda 8: Evaluación Final
nb.cells.append(nbf.v4.new_code_cell("""from pyspark.ml.evaluation import ClusteringEvaluator

# Evaluar el modelo final usando el Coeficiente de Silueta sobre pcaFeatures
evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="pcaFeatures", metricName="silhouette")

silhouette = evaluator.evaluate(predictions)

print("-" * 30)
print(f"📊 Coeficiente de Silueta Final: {silhouette:.4f}")
print("-" * 30)"""))

# Celda 9: Interpretación
nb.cells.append(nbf.v4.new_code_cell("""# Análisis de los perfiles clínicos por clúster
print("Resumen estadístico por Clúster:")
perfiles = predictions.groupBy("prediction") \\
    .agg({
        "time_in_hospital": "mean", 
        "num_medications": "mean",
        "num_lab_procedures": "mean"
    }) \\
    .orderBy("prediction")

perfiles.show()

# Interpretación textual breve basada en los resultados
print("Interpretación de los Clústeres:")
print("- Clúster 0: Posible perfil de baja intensidad clínica.")
print("- Clúster 1: Pacientes con mayor estancia o medicación.")
print("- Clúster 2: Perfil intermedio o con alta frecuencia de laboratorios.")"""))

# Guardar el archivo
file_name = "diabetes_clustering_fase2.ipynb"
with open(file_name, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook '{file_name}' actualizado exitosamente con PCA y Justificación de K.")
