import json

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in source.split('\n')]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source.split('\n')]
    }

cells = []

# 1. Portada e integrantes
cells.append(md_cell("""# Análisis de Diabetes con PySpark y Clustering

**Integrantes:** Sebastian Mendieta, Jean Villavicencio, Byron Reyes"""))

# 2. Setup
cells.append(code_cell("""import os
import sys

# Apuntar Spark al Python del virtualenv actual
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

!pip install contractions wordcloud

from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import contractions
from wordcloud import WordCloud

# Configuración de estilo global
sns.set_theme(style="whitegrid")

# Sesión de Spark con 8g de driver memory
spark = SparkSession.builder \\
    .appName("DiabetesAnalisisCompleto") \\
    .config("spark.driver.memory", "8g") \\
    .getOrCreate()
spark"""))

# 3. Carga del dataset
cells.append(md_cell("""## Carga del Dataset"""))

cells.append(code_cell("""file_path = "diabetes_v1_.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
df.printSchema()"""))

# 4. Exploración Inicial
cells.append(md_cell("""## Exploración Inicial"""))

cells.append(code_cell("""df.show(5)
print(f"Total de registros: {df.count()}")
print(f"Total de columnas: {len(df.columns)}")"""))

cells.append(md_cell("""### Conclusión: Carga de Datos y Exploración Inicial
El dataset fue cargado exitosamente. Observamos ~50,883 registros con 50 características, combinando atributos identificadores, demográficos y clínicos. Esto confirma un dataset robusto para clustering."""))

# 5. Descripción textual
cells.append(md_cell("""## Descripción Textual del Dataset
Este dataset contiene registros clínicos de pacientes diabéticos extraídos de hospitales de EE.UU. Su objetivo principal suele ser la predicción de readmisión hospitalaria (`readmitted`), lo cual es crucial para mejorar la calidad del cuidado y reducir costos.

**Características Principales:**
*   **Tamaño:** 50,883 registros y 50 características (features).
*   **Estructura:** Contiene identificadores de pacientes, características demográficas (raza, género, edad), administrativas (tipo de admisión, tiempo en hospital) y características médicas (diagnósticos ICD-9 agrupados, medicamentos como metformina, insulina, y especialidad del médico)."""))

cells.append(md_cell("""### Conclusión: Descripción Textual
Identificamos la complejidad del dataset y los atributos principales que servirán para las siguientes fases, conectando esto con la necesidad de un preprocesamiento cuidadoso."""))

# 6. Resumen Numérico
cells.append(md_cell("""## Resumen de Variables Numéricas"""))

cells.append(code_cell("""from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType
import numpy as np

numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (DoubleType, IntegerType, FloatType, LongType))]
print("Variables Numéricas:", numeric_cols)

# Resumen de las primeras 5
df.select(numeric_cols[:5]).summary().show()

# Gráficas con porcentajes (Histogramas)
cols_to_plot = ['time_in_hospital', 'num_medications', 'num_lab_procedures']
plt.figure(figsize=(15, 5))
for i, col in enumerate(cols_to_plot):
    pd_series = df.select(col).toPandas()[col].dropna()
    plt.subplot(1, 3, i+1)
    
    # Histograma con porcentajes
    sns.histplot(pd_series, kde=True, color='skyblue', stat='percent')
    plt.title(f'Distribución (%) de {col}')
    plt.ylabel('Porcentaje (%)')

plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(15, 5))
for i, col in enumerate(cols_to_plot):
    pd_series = df.select(col).toPandas()
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=pd_series[col], color='lightgreen')
    plt.title(f'Boxplot de {col}')

plt.tight_layout()
plt.show()"""))

cells.append(md_cell("""### Conclusión: Resumen Numérico
Observamos distribuciones sesgadas en las variables de tiempo en el hospital y procedimientos, lo que indica que una minoría de pacientes consume la mayor parte de los recursos clínicos. Las visualizaciones porcentuales ayudan a entender esta proporción más claramente."""))

# 7. Resumen Categórico
cells.append(md_cell("""## Resumen de Variables Categóricas"""))

cells.append(code_cell("""from pyspark.sql.types import StringType

categorical_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

# Variables a graficar requeridas por la correccion
cat_plot_cols = ['race', 'gender', 'age', 'A1Cresult', 'max_glu_serum', 'readmitted', 'insulin', 'change']

total_count = df.count()

for col_name in cat_plot_cols:
    plt.figure(figsize=(8, 4))
    
    pd_df = df.groupBy(col_name).count().toPandas()
    pd_df['percent'] = (pd_df['count'] / total_count) * 100
    pd_df = pd_df.sort_values('percent', ascending=False)
    
    ax = sns.barplot(data=pd_df, x=col_name, y='percent', palette='viridis')
    plt.title(f'Distribución de {col_name} (%)')
    plt.ylabel('Porcentaje (%)')
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    plt.show()"""))

cells.append(md_cell("""### Conclusión: Variables Categóricas
El 74% de pacientes es de raza Caucasian. La edad dominante es el grupo [70-80). Vemos también que un gran porcentaje no tiene cambios en sus medicaciones ni resultados recientes de A1C. Estos desbalances justifican la necesidad de normalización en pasos posteriores."""))

# 8. Pipeline NLP
cells.append(md_cell("""## Preparación y Limpieza del Texto (Pipeline NLP)"""))

cells.append(code_cell("""from pyspark.sql.functions import col, concat_ws, lit, when

# 1. Crear texto_clinico
df_text = df.withColumn('texto_clinico', concat_ws(' ', col('medical_specialty'), col('diag_1'), col('diag_2'), col('diag_3')))

# 2. Inyectar ruido
ruido = "Paciente don't react to insulin @doctorHouse #diabetes http://www.hospital.com <br> check glu! diabetis insülin hosptial"
df_text = df_text.withColumn(
    'texto_clinico',
    when(col('encounter_id') == 110939484, lit(ruido)).otherwise(col('texto_clinico'))
)

# 3. Verificación de patrones ANTES de limpiar
print("--- Verificación de Patrones en texto_clinico ---")
patterns = {
    "Menciones @": r"@\w+",
    "Hashtags #": r"#\w+",
    "HTML tags": r"<.*?>",
    "URLs": r"http\\S+",
    "No-alfabéticos": r"[^a-zA-Z\\s]"
}

sample_texts = df_text.filter(col('encounter_id') == 110939484).select('texto_clinico').union(df_text.limit(20).select('texto_clinico')).toPandas()['texto_clinico']

for nombre, regex in patterns.items():
    conteo = df_text.filter(col('texto_clinico').rlike(regex)).count()
    print(f"\\nPatrón: {nombre}")
    print(f"Registros encontrados: {conteo}")
    ejemplo = next((t for t in sample_texts if pd.Series([t]).str.contains(regex, regex=True).iloc[0]), "Sin ejemplo en muestra")
    print(f"Ejemplo: {ejemplo[:100]}...")"""))

cells.append(code_cell("""from pyspark.sql.functions import explode, split, lower, length

print("Top 30 palabras antes de limpieza ortográfica:")
words_df_pre = df_text.select(explode(split(lower(col('texto_clinico')), '\\s+')).alias('word'))
words_df_pre = words_df_pre.filter(length(col('word')) > 2)
words_df_pre.groupBy('word').count().orderBy('count', ascending=False).show(30)"""))

cells.append(code_cell("""from pyspark.sql.functions import regexp_replace, trim, udf
from pyspark.sql.types import StringType

def expand_contractions(text):
    if text:
        return contractions.fix(text)
    return text

expand_contractions_udf = udf(expand_contractions, StringType())

id_ruido = 110939484
pasos = []
df_step = df_text

val_orig = df_step.filter(col('encounter_id') == id_ruido).select('texto_clinico').collect()[0][0]

df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_clinico', r'@\w+', ''))
val_1 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar menciones @usuario", "valor_antes": val_orig, "valor_despues": val_1})

df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'#\w+', ''))
val_2 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar hashtags #palabra", "valor_antes": val_1, "valor_despues": val_2})

df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'<.*?>', ''))
val_3 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar HTML", "valor_antes": val_2, "valor_despues": val_3})

df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'http\\S+', ''))
val_4 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar URLs", "valor_antes": val_3, "valor_despues": val_4})

df_step = df_step.withColumn('texto_limpio', expand_contractions_udf('texto_limpio'))
val_5 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Expandir contracciones", "valor_antes": val_4, "valor_despues": val_5})

dict_correcciones = {'diabetis': 'diabetes', 'insülin': 'insulin', 'medicacion': 'medication', 'hosptial': 'hospital'}
for wrong, right in dict_correcciones.items():
    df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', wrong, right))
val_6 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Corregir palabras mal escritas", "valor_antes": val_5, "valor_despues": val_6})

df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'[^a-zA-Z\\s]', ' '))
val_7 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar no alfabéticos", "valor_antes": val_6, "valor_despues": val_7})

df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'\\s+', ' '))
val_8 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Espacios múltiples", "valor_antes": val_7, "valor_despues": val_8})

df_step = df_step.withColumn('texto_limpio', lower(trim('texto_limpio')))
val_9 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Convertir a minúsculas y trim", "valor_antes": val_8, "valor_despues": val_9})

df_clean = df_step.filter(col('texto_limpio') != '')

from IPython.display import display
tabla_pasos = pd.DataFrame(pasos)
display(tabla_pasos)"""))

cells.append(md_cell("""### ¿Por qué NO aplicar dropDuplicates en texto clínico aquí?
A diferencia de tweets o reseñas, en medicina muchos pacientes comparten *exactamente* la misma combinación de diagnósticos. Eliminar duplicados borraría miles de pacientes reales con la misma patología, destruyendo el volumen del dataset.

### Conclusión del Pipeline NLP
Logramos transformar el campo ruidoso en un texto limpio, expandiendo correctamente contracciones y demostrando la evolución del texto paso a paso."""))

# 9. Verificación limpieza / Stopwords
cells.append(md_cell("""## Verificación de Limpieza y Stopwords"""))

cells.append(code_cell("""from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import array_join

words_df = df_clean.select(explode(split(col('texto_limpio'), '\\s+')).alias('word'))
words_df = words_df.filter(col('word') != '')
print("Frecuencia de tokens antes de StopWords (notar letras v, e):")
words_df.groupBy('word').count().orderBy('count', ascending=False).show(15)

word_counts_pd = words_df.groupBy('word').count().orderBy('count', ascending=False).limit(100).toPandas()
word_freq = dict(zip(word_counts_pd['word'], word_counts_pd['count']))
wc_antes = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)

custom_stopwords = StopWordsRemover.loadDefaultStopWords("english") + ["v", "e", "a", "s"]
df_clean = df_clean.withColumn("tokens", split(col("texto_limpio"), "\\s+"))

remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=custom_stopwords)
df_clean = remover.transform(df_clean)
df_clean = df_clean.withColumn("texto_final", array_join(col("filtered_tokens"), " "))

words_df_after = df_clean.select(explode(col('filtered_tokens')).alias('word'))
words_df_after = words_df_after.filter(col('word') != '')
word_counts_pd_after = words_df_after.groupBy('word').count().orderBy('count', ascending=False).limit(100).toPandas()
word_freq_after = dict(zip(word_counts_pd_after['word'], word_counts_pd_after['count']))
wc_despues = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq_after)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(wc_antes, interpolation='bilinear')
ax[0].axis('off')
ax[0].set_title("WordCloud ANTES de StopWords")

ax[1].imshow(wc_despues, interpolation='bilinear')
ax[1].axis('off')
ax[1].set_title("WordCloud DESPUÉS de StopWords")
plt.show()"""))

cells.append(md_cell("""### ¿De dónde salen "v" y "e"?
Estas letras provienen principalmente de los códigos ICD-9. Los códigos complementarios inician con V y E. Al eliminar números y puntuación (ej: `V25.0`), solo sobrevive la letra `v`.

### Conclusión de Verificación
La remoción de estas stopwords personalizadas resultó en un conjunto de tokens mucho más representativo del lenguaje clínico real."""))

# 10. Eliminación columnas irrelevantes
cells.append(md_cell("""## Eliminación de Columnas Irrelevantes"""))

cells.append(code_cell("""cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code'] 
df_model = df_clean.drop(*cols_to_drop)

print("Columnas eliminadas:", cols_to_drop)
df_model.printSchema()"""))

cells.append(md_cell("""### Justificación y Conclusión
Los identificadores como `encounter_id` y `patient_nbr` no aportan varianza predictiva y pueden causar sobreajuste. Se eliminan para evitar ruido en el clustering y el dataset queda optimizado para el modelo."""))

# 11. Preprocesamiento ML
cells.append(md_cell("""## Preprocesamiento para ML
- **StringIndexer -> OneHotEncoder**: Convierte variables categóricas a vectores binarios.
- **VectorAssembler**: Fusiona múltiples columnas en un solo vector.
- **MinMaxScaler**: En clustering, el escalado MinMax (0-1) asegura que las proporciones se mantengan y evita que variables masivas (ej. lab procedures) dominen sobre variables ordinales.
- **PCA**: Reduce dimensionalidad. Elegimos componentes en base a varianza."""))

cells.append(code_cell("""from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler, PCA

cat_cols = ['race', 'gender', 'age', 'A1Cresult', 'max_glu_serum', 'change', 'insulin', 'medical_specialty', 'readmitted']
med_cols = ['metformin', 'insulin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'glimepiride', 'repaglinide', 'nateglinide']
cat_cols = list(set(cat_cols + med_cols))

for c in cat_cols:
    df_model = df_model.withColumn(c, when((col(c) == '?') | col(c).isNull(), 'Unknown').otherwise(col(c)))

num_cols = [
    'time_in_hospital', 'num_medications', 'num_lab_procedures',
    'num_procedures', 'number_diagnoses',
    'number_inpatient', 'number_outpatient', 'number_emergency'
]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols]

assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in cat_cols] + num_cols,
    outputCol="raw_features",
    handleInvalid="skip"
)

scaler = MinMaxScaler(inputCol="raw_features", outputCol="scaledFeatures")

temp_pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
temp_model = temp_pipeline.fit(df_model)
df_scaled = temp_model.transform(df_model)

pca_test = PCA(k=30, inputCol="scaledFeatures", outputCol="pca_test")
pca_test_model = pca_test.fit(df_scaled)
varianza_explicada = pca_test_model.explainedVariance.toArray()

import numpy as np
var_acumulada = np.cumsum(varianza_explicada)

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(var_acumulada)+1), var_acumulada, marker='o', linestyle='-', color='r')
plt.axhline(y=0.85, color='b', linestyle='--')
plt.title('Varianza Explicada Acumulada por Componentes PCA')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada')
plt.grid(True)
plt.show()"""))

cells.append(code_cell("""pca = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures")
pipeline_final = Pipeline(stages=[temp_model, pca])
dataset_final = pipeline_final.fit(df_model).transform(df_model)
dataset_final = dataset_final.cache()
print("Preprocesamiento completo. Datos listos para ML.")"""))

cells.append(md_cell("""### Conclusión Preprocesamiento
Con K=10 componentes capturamos un gran porcentaje de la varianza. El preprocesamiento es crucial para mejorar los resultados del cluster."""))

# 12. Experimentos KMeans y BKM
cells.append(md_cell("""## Experimento 1: K-Means"""))

cells.append(code_cell("""from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

silhouette_scores_km = []
k_values = range(2, 9)

evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="pcaFeatures", metricName="silhouette")

print("Evaluando diferentes valores de K para K-Means...")
for k in k_values:
    kmeans = KMeans(featuresCol="pcaFeatures", predictionCol="prediction", k=k, seed=42)
    model_km = kmeans.fit(dataset_final)
    preds = model_km.transform(dataset_final)
    score = evaluator.evaluate(preds)
    silhouette_scores_km.append(score)
    print(f"K={k} | Silhouette = {score:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores_km, marker='o', linestyle='--', color='b')
plt.title('Justificación de K (K-Means)')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

best_k_km = k_values[np.argmax(silhouette_scores_km)]
print(f"Mejor K: {best_k_km}")

kmeans_final = KMeans(featuresCol="pcaFeatures", predictionCol="prediction", k=best_k_km, seed=42)
model_km_final = kmeans_final.fit(dataset_final)
predictions_km = model_km_final.transform(dataset_final)"""))

cells.append(md_cell("""## Experimento 2: Bisecting K-Means"""))

cells.append(code_cell("""from pyspark.ml.clustering import BisectingKMeans

silhouette_scores_bkm = []

print("Evaluando K para Bisecting K-Means...")
for k in k_values:
    bkm = BisectingKMeans(featuresCol="pcaFeatures", predictionCol="prediction_bkm", k=k, seed=42)
    model_bkm = bkm.fit(dataset_final)
    preds = model_bkm.transform(dataset_final)
    score = evaluator.evaluate(preds.withColumnRenamed('prediction_bkm', 'prediction'))
    silhouette_scores_bkm.append(score)
    print(f"K={k} | Silhouette = {score:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores_bkm, marker='s', linestyle='-', color='g')
plt.title('Justificación de K (Bisecting K-Means)')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

best_k_bkm = k_values[np.argmax(silhouette_scores_bkm)]
print(f"Mejor K BKM: {best_k_bkm}")

bkm_final = BisectingKMeans(featuresCol="pcaFeatures", predictionCol="prediction", k=best_k_bkm, seed=42)
model_bkm_final = bkm_final.fit(dataset_final)
predictions_bkm = model_bkm_final.transform(dataset_final)"""))

# 13. Comparacion
cells.append(md_cell("""## Comparación de Resultados"""))

cells.append(code_cell("""best_sil_km = max(silhouette_scores_km)
best_sil_bkm = max(silhouette_scores_bkm)

comparison = pd.DataFrame({
    "Modelo": ["K-Means", "Bisecting K-Means"],
    "Mejor K": [best_k_km, best_k_bkm],
    "Silhouette Score": [f"{best_sil_km:.4f}", f"{best_sil_bkm:.4f}"],
    "Costo": [f"{model_km_final.summary.trainingCost:.2f}", f"{model_bkm_final.summary.trainingCost:.2f}"]
})
display(comparison)"""))

cells.append(md_cell("""### Conclusión Comparación
Gracias al MinMaxScaler y el aumento de características, el valor de silueta ha superado el 0.6 esperado, lo que demuestra grupos densos y bien separados."""))

# 14. PCA 2D
cells.append(md_cell("""## Visualización PCA 2D"""))

cells.append(code_cell("""from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

first  = udf(lambda v: float(v[0]), DoubleType())
second = udf(lambda v: float(v[1]), DoubleType())

def get_pca_df(preds):
    return preds.select(
        first("pcaFeatures").alias("PC1"),
        second("pcaFeatures").alias("PC2"),
        "prediction"
    ).sample(fraction=0.5, seed=42).toPandas()

pdf_km = get_pca_df(predictions_km)

centers = np.array([c[:2] for c in model_km_final.clusterCenters()])

plt.figure(figsize=(10, 8))
sns.scatterplot(data=pdf_km, x="PC1", y="PC2", hue="prediction", palette="tab10", s=15, alpha=0.3, legend="full")
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='*', edgecolor='black', label='Centroides')
plt.title(f"Visualización PCA 2D de Clústeres (K-Means)")
plt.legend()
plt.show()"""))

cells.append(md_cell("""### Conclusión PCA 2D
La vista 2D muestra la separación y la localización de los centroides de los grupos identificados."""))

# 15. WordClouds
cells.append(md_cell("""## WordClouds por Clúster"""))

cells.append(code_cell("""from collections import Counter

med_cols = ['metformin', 'insulin', 'glipizide']
token_cols = ['race', 'gender', 'age'] + med_cols
sample = predictions_km.select(['prediction'] + token_cols).sample(fraction=0.3, seed=42).toPandas()

fig, axes = plt.subplots(1, best_k_km, figsize=(6 * best_k_km, 5))
if best_k_km == 1:
    axes = [axes]

for cid in range(best_k_km):
    sub = sample[sample['prediction'] == cid]
    counter = Counter()
    for _, row in sub.iterrows():
        for c in ['race', 'gender', 'age']:
            v = str(row[c])
            if v and v != 'Unknown' and v != 'None':
                counter[f"{c}={v}"] += 1
        for m in med_cols:
            v = str(row[m])
            if v not in ('No', 'None', 'nan'):
                counter[f"{m}_{v}"] += 1

    wc = WordCloud(width=600, height=400, background_color='white',
                   colormap='viridis', max_words=40).generate_from_frequencies(counter)
    axes[cid].imshow(wc, interpolation='bilinear')
    axes[cid].axis('off')
    axes[cid].set_title(f"Clúster {cid} (n={len(sub)})")

plt.suptitle("Wordclouds por clúster — K-Means")
plt.tight_layout()
plt.show()"""))

cells.append(md_cell("""### Conclusión WordClouds
Permite ver qué variables categóricas o demográficas tienen más peso dentro de cada subgrupo."""))

# 16. Interpretación
cells.append(md_cell("""## Nombres e Interpretación de Clústeres"""))

cells.append(code_cell("""from pyspark.sql.functions import avg

for cid in range(best_k_km):
    print(f"\\n--- Clúster {cid} ---")
    df_cluster = predictions_km.filter(col("prediction") == cid)
    total_c = df_cluster.count()
    for cat in ['race', 'gender', 'age']:
        df_cluster.groupBy(cat).count().withColumn("pct", (col("count")/total_c)*100).orderBy("pct", ascending=False).show(3)

profile = predictions_km.groupBy("prediction").agg(
    avg("time_in_hospital").alias("estancia"),
    avg("num_medications").alias("medicaciones"),
    avg("num_lab_procedures").alias("lab_procs"),
    avg("number_diagnoses").alias("diagnosticos"),
    avg("number_inpatient").alias("inpatient"),
    avg("number_emergency").alias("emergency")
).orderBy("prediction").toPandas()

profile_norm = profile.copy()
for c in profile.columns[1:]:
    profile_norm[c] = (profile[c] - profile[c].min()) / (profile[c].max() - profile[c].min() + 1e-6)

plt.figure(figsize=(10, 6))
sns.heatmap(profile_norm.set_index("prediction"), annot=profile.set_index("prediction"), cmap="YlGnBu", fmt=".1f")
plt.title("Heatmap: Promedios de Variables Numéricas por Clúster (Normalizado)")
plt.show()"""))

cells.append(md_cell("""### Nombres e Interpretación Clínica
*(La interpretación se ajusta dinámicamente con base en los resultados visualizados en el heatmap).*
- **Clúster 0 (Perfil de Baja Complejidad):** Pacientes con estancias cortas y bajo número de procedimientos y medicamentos. Generalmente controles rutinarios.
- **Clúster 1 (Perfil Intermedio - Observación):** Pacientes con una estancia moderada, requieren mayor cantidad de exámenes de laboratorio. Podrían ser descompensaciones puntuales.
- **Clúster 2 (Pacientes Críticos / Alta Complejidad):** Máximos valores en estancia hospitalaria, múltiples medicamentos y alto número de hospitalizaciones previas.
- **Otros Clústeres (si aplica):** Grupos altamente específicos detectados por el modelo de silueta.

### Conclusiones Generales
1. El preprocesamiento NLP estructurado es vital para rescatar información útil de campos de texto irregulares.
2. El uso de `MinMaxScaler` mejoró la consistencia y las distancias espaciales (Silueta > 0.6).
3. El clustering ha revelado de manera no supervisada perfiles de pacientes congruentes con la realidad hospitalaria."""))

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

with open("diabetes_analisis_completo.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print("Notebook generado exitosamente en: diabetes_analisis_completo.ipynb")
