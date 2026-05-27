import nbformat as nbf

file_path = "analisis_diabetes.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Section 4
cell1 = nbf.v4.new_markdown_cell("""## 4. Descripción Textual del Dataset

Este dataset contiene registros clínicos de pacientes diabéticos extraídos de hospitales de EE.UU. Su objetivo principal suele ser la predicción de readmisión hospitalaria (`readmitted`), lo cual es crucial para mejorar la calidad del cuidado y reducir costos.

**Características Principales:**
*   **Tamaño:** 50,883 registros y 50 características (features).
*   **Estructura:** Contiene identificadores de pacientes, características demográficas (raza, género, edad), administrativas (tipo de admisión, tiempo en hospital) y características médicas (diagnósticos ICD-9 agrupados, medicamentos como metformina, insulina, y especialidad del médico).
*   **Tipos de datos:** Mayoritariamente variables categóricas representadas como strings (`race`, `gender`, `diag_1`), y algunas variables numéricas discretas (`time_in_hospital`, `num_lab_procedures`).
""")

# Section 5
cell2 = nbf.v4.new_markdown_cell("""## 5. Gráficas Resumen de Variables (Visualización)""")

cell3 = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_theme(style="whitegrid")

# Gráfico Cualitativo: Distribución de Raza
race_pd = df.groupBy('race').count().toPandas()

plt.figure(figsize=(10, 5))
sns.barplot(data=race_pd.sort_values('count', ascending=False), x='race', y='count', palette='viridis')
plt.title('Distribución por Raza')
plt.xticks(rotation=45)
plt.show()

# Gráfico Cuantitativo: Días en el hospital por género
# Convertimos las columnas necesarias a Pandas para graficar
sample_df = df.select('gender', 'time_in_hospital').toPandas()
# Filtramos géneros desconocidos
sample_df = sample_df[sample_df['gender'].isin(['Male', 'Female'])]

plt.figure(figsize=(8, 5))
sns.boxplot(data=sample_df, x='gender', y='time_in_hospital', palette='Set2')
plt.title('Distribución de Días en el Hospital por Género')
plt.show()
""")

# Section 6
cell4 = nbf.v4.new_markdown_cell("""## 6. Preparación y Transformación Textual (Pipeline NLP PySpark)

Dado que este es un dataset clínico tabular estructurado y carece de campos de texto libre como tweets o foros, combinaremos las descripciones de especialidades y los diagnósticos en una sola columna **`texto_clinico`**. 
Insertaremos texto ruidoso simulado (hashtags, @, URLs, signos de puntuación, contracciones) en un registro para demostrar de forma práctica el pipeline de limpieza estipulado en los requerimientos.
""")

cell5 = nbf.v4.new_code_cell("""from pyspark.sql.functions import col, concat_ws, lower, regexp_replace, trim, udf, lit, when
from pyspark.sql.types import StringType
import contractions

# UDF para expandir contracciones
def expand_contractions(text):
    if text:
        return contractions.fix(text)
    return text

expand_contractions_udf = udf(expand_contractions, StringType())

# 1. Crear la columna de texto clínico
df_text = df.withColumn('texto_clinico', concat_ws(' ', col('medical_specialty'), col('diag_1'), col('diag_2'), col('diag_3')))

# 2. Inyectar ruido simulado en el primer registro para ver la limpieza en acción
ruido = "Paciente don't react to insulin @doctorHouse #diabetes http://www.hospital.com <br> check glu!"
df_text = df_text.withColumn(
    'texto_clinico',
    when(col('encounter_id') == 110939484, lit(ruido)).otherwise(col('texto_clinico'))
)

# 3. Pipeline de NLP PySpark
df_clean = df_text \\
    .withColumn('texto_limpio', col('texto_clinico')) \\
    .withColumn('texto_limpio', regexp_replace('texto_limpio', r'@\\w+', '')) \\
    .withColumn('texto_limpio', regexp_replace('texto_limpio', r'#\\w+', '')) \\
    .withColumn('texto_limpio', regexp_replace('texto_limpio', r'<.*?>', '')) \\
    .withColumn('texto_limpio', regexp_replace('texto_limpio', r'http\\S+', '')) \\
    .withColumn('texto_limpio', expand_contractions_udf('texto_limpio')) \\
    .withColumn('texto_limpio', regexp_replace('texto_limpio', r'diabetis', 'diabetes')) \\
    .withColumn('texto_limpio', regexp_replace('texto_limpio', r'[^a-zA-Z\\s]', ' ')) \\
    .withColumn('texto_limpio', regexp_replace('texto_limpio', r'\\s+', ' ')) \\
    .withColumn('texto_limpio', lower(trim('texto_limpio'))) \\
    .filter(col('texto_limpio') != '') \\
    .dropDuplicates(['texto_limpio'])

# Mostrar la comparación entre el original con ruido y el resultado limpio
df_clean.filter(col('encounter_id') == 110939484).select('texto_clinico', 'texto_limpio').show(truncate=False)
""")

cell6 = nbf.v4.new_markdown_cell("""## 7. Verificación de Limpieza (Frecuencia de Palabras y WordCloud)""")

cell7 = nbf.v4.new_code_cell("""from pyspark.sql.functions import explode, split
from wordcloud import WordCloud

# Tokenizar (separar por espacio)
words_df = df_clean.select(explode(split(col('texto_limpio'), '\\s+')).alias('word'))
words_df = words_df.filter((col('word') != '') & (col('word') != '?'))

# Calcular frecuencias
word_counts = words_df.groupBy('word').count().orderBy('count', ascending=False)

print("Top 15 palabras más frecuentes:")
word_counts.show(15)

# Generar WordCloud
word_counts_pd = word_counts.limit(100).toPandas()
# Convertimos el DataFrame de frecuencias a un diccionario para wordcloud
word_freq = dict(zip(word_counts_pd['word'], word_counts_pd['count']))

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud de Diagnósticos y Especialidades Limpias', fontsize=16)
plt.show()
""")

# Evitamos agregar los integrantes dos veces si ya existen al final
last_cell = nb.cells[-1]

# Quitar la ultima celda de integrantes temporalmente
if 'Integrantes:' in last_cell.source:
    integrantes_cell = nb.cells.pop(-1)
else:
    integrantes_cell = nbf.v4.new_markdown_cell("""## Integrantes:\n\n- Sebastian Mendieta\n- Jean Villavicencio\n- Byron Reyes""")

# Agregar celdas
nb.cells.extend([cell1, cell2, cell3, cell4, cell5, cell6, cell7, integrantes_cell])

with open(file_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Celdas de Fase II agregadas con éxito.")
