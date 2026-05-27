import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def find_cell(cells, substring):
    for i, c in enumerate(cells):
        text = "".join(c['source'])
        if substring in text:
            return i, c
    return -1, None

# CORRECCIÓN 0: UDF de contracciones
i, c = find_cell(cells, "def expand_contractions(text):")
if i != -1:
    new_source = """from pyspark.sql.functions import regexp_replace, trim, lower, col
from pyspark.sql.types import StringType
import contractions as contractions_lib
import pandas as pd

id_ruido = 110939484
pasos = []
df_step = df_text

val_orig = df_step.filter(col('encounter_id') == id_ruido).select('texto_clinico').collect()[0][0]

# Paso 1: Eliminar menciones @
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_clinico', r'@\w+', ''))
val_1 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar menciones @usuario", "valor_antes": val_orig, "valor_despues": val_1})

# Paso 2: Eliminar hashtags
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'#\w+', ''))
val_2 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar hashtags #palabra", "valor_antes": val_1, "valor_despues": val_2})

# Paso 3: Eliminar HTML
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'<.*?>', ''))
val_3 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar HTML", "valor_antes": val_2, "valor_despues": val_3})

# Paso 4: Eliminar URLs
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'http\S+', ''))
val_4 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar URLs", "valor_antes": val_3, "valor_despues": val_4})

# Paso 5: Expandir contracciones — SE HACE EN PANDAS
pd_texto = df_step.select('encounter_id', 'texto_limpio').toPandas()
pd_texto['texto_limpio'] = pd_texto['texto_limpio'].apply(
    lambda x: contractions_lib.fix(x) if isinstance(x, str) else x
)
spark_contractions = spark.createDataFrame(pd_texto[['encounter_id', 'texto_limpio']]) \
    .withColumnRenamed('texto_limpio', 'texto_contracc')

df_step = df_step.join(spark_contractions, on='encounter_id', how='left') \
    .withColumn('texto_limpio', col('texto_contracc')) \
    .drop('texto_contracc')

val_5 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Expandir contracciones (via pandas)", "valor_antes": val_4, "valor_despues": val_5})

# Paso 6: Corregir palabras mal escritas
dict_correcciones = {
    'diabetis': 'diabetes',
    'ins\\ülin': 'insulin',
    'medicacion': 'medication',
    'hosptial': 'hospital',
    'admision': 'admission',
    'especialidad': 'specialty'
}
for wrong, right in dict_correcciones.items():
    df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', wrong, right))
val_6 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Corregir palabras mal escritas", "valor_antes": val_5, "valor_despues": val_6})

# Paso 7: Eliminar no alfabéticos
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'[^a-zA-Z\s]', ' '))
val_7 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Eliminar no alfabéticos", "valor_antes": val_6, "valor_despues": val_7})

# Paso 8: Normalizar espacios
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'\s+', ' '))
val_8 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Espacios múltiples", "valor_antes": val_7, "valor_despues": val_8})

# Paso 9: Minúsculas y trim
df_step = df_step.withColumn('texto_limpio', lower(trim('texto_limpio')))
val_9 = df_step.filter(col('encounter_id') == id_ruido).select('texto_limpio').collect()[0][0]
pasos.append({"patron": "Convertir a minúsculas y trim", "valor_antes": val_8, "valor_despues": val_9})

# Filtrar cadenas vacías
df_clean = df_step.filter(col('texto_limpio') != '')

from IPython.display import display
tabla_pasos = pd.DataFrame(pasos)
display(tabla_pasos)
"""
    c['source'] = [line + '\n' for line in new_source.split('\n')]
else:
    print("Warning: Corrección 0 text not found")

# CORRECCIÓN 1: Wordcloud
i, c = find_cell(cells, "med_cols = ['metformin', 'insulin', 'glipizide']")
if i != -1:
    source = "".join(c['source'])
    source = source.replace("med_cols = ['metformin', 'insulin', 'glipizide']", "med_cols = ['metformin', 'insulin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'glimepiride', 'repaglinide', 'nateglinide']")
    source = source.replace("token_cols = ['race', 'gender', 'age'] + med_cols", "token_cols = ['race', 'gender', 'age', 'A1Cresult', 'max_glu_serum', 'change'] + med_cols")
    source = source.replace("for c in ['race', 'gender', 'age']:", "for c in ['race', 'gender', 'age', 'A1Cresult', 'max_glu_serum', 'change']:")
    c['source'] = [line + '\n' for line in source.split('\n')]
else:
    print("Warning: Corrección 1 text not found")

# CORRECCIÓN 2: Verificar silueta
i, c = find_cell(cells, "Gracias al MinMaxScaler y el aumento de características")
if i != -1:
    code_source = """print("=" * 50)
print("VERIFICACIÓN DE OBJETIVO DE SILUETA (>= 0.6)")
print("=" * 50)
print(f"K-Means       | Silhouette = {best_sil_km:.4f} | "
      f"{'✓ Cumple objetivo' if best_sil_km >= 0.6 else '✗ Por debajo del objetivo de 0.6'}")
print(f"Bisecting KM  | Silhouette = {best_sil_bkm:.4f} | "
      f"{'✓ Cumple objetivo' if best_sil_bkm >= 0.6 else '✗ Por debajo del objetivo de 0.6'}")
print("=" * 50)

if best_sil_km < 0.6 and best_sil_bkm < 0.6:
    print("\\nNOTA: Ningún modelo alcanzó 0.6. Considerar: ampliar features, "
          "ajustar k o revisar el preprocesamiento.")
"""
    cells[i] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in code_source.split('\n')]
    }
    cells.insert(i+1, {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["El valor de silueta obtenido arriba es calculado en ejecución real. Si supera 0.6 confirma grupos bien definidos con buena cohesión interna y separación entre clústeres. Si está por debajo, se recomienda revisar el número de componentes PCA o el conjunto de features utilizado.\n"]
    })
else:
    print("Warning: Corrección 2 text not found")

# CORRECCIÓN 3 y 4: Tabla y heatmap
i, c = find_cell(cells, "from pyspark.sql.functions import avg")
if i != -1:
    source_3 = """# Tabla consolidada: distribución porcentual de variables categóricas por clúster
print("Distribución porcentual de variables categóricas clave por clúster:")

rows_tabla = []
cat_interp = ['race', 'gender', 'age', 'A1Cresult', 'insulin', 'change', 'readmitted']

for cid_val in sorted(predictions_km.select('prediction').distinct().toPandas()['prediction'].tolist()):
    df_c = predictions_km.filter(col('prediction') == cid_val)
    total_c = df_c.count()
    for cat in cat_interp:
        dist = df_c.groupBy(cat).count().toPandas()
        dist['pct'] = (dist['count'] / total_c * 100).round(2)
        dist = dist.sort_values('pct', ascending=False).head(3)
        for _, row in dist.iterrows():
            rows_tabla.append({
                "Clúster": int(cid_val),
                "Variable": cat,
                "Valor dominante": row[cat],
                "Porcentaje (%)": row['pct']
            })

tabla_cat = pd.DataFrame(rows_tabla)
display(tabla_cat)
"""
    cells.insert(i, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source_3.split('\n')]
    })
else:
    print("Warning: Corrección 3 text not found")

i, c = find_cell(cells, "- **Clúster 0 (Perfil de Baja Complejidad):**")
if i != -1:
    source_4 = """# Asignación dinámica de nombres basada en los valores reales del DataFrame profile
def asignar_nombres_dinamicos(profile_df):
    df = profile_df.copy()
    df['score_recurrencia'] = df['inpatient'] + df['emergency']
    df['score_intensidad']  = df['estancia'] + df['medicaciones'] / 5 + df['lab_procs'] / 10
    
    nombres = {}
    disponibles = list(df.index)
    
    # Clúster con mayor recurrencia
    idx_rec = df['score_recurrencia'].idxmax()
    cid_rec = int(df.loc[idx_rec, 'prediction'])
    nombres[cid_rec] = "Pacientes Recurrentes / Alta Carga Asistencial"
    disponibles = [j for j in disponibles if j != idx_rec]
    
    # De los restantes, el de mayor intensidad clínica
    if disponibles:
        idx_int = df.loc[disponibles, 'score_intensidad'].idxmax()
        cid_int = int(df.loc[idx_int, 'prediction'])
        nombres[cid_int] = "Pacientes en Evaluación Intensiva / Casos Complejos"
        disponibles = [j for j in disponibles if j != idx_int]
    
    # Los restantes
    for j, idx in enumerate(disponibles):
        cid = int(df.loc[idx, 'prediction'])
        nombres[cid] = "Pacientes Ambulatorios / Control Rutinario" if j == 0 else f"Perfil Mixto / Subgrupo {j}"
    
    return nombres

nombres_clusters = asignar_nombres_dinamicos(profile)

print("\\n" + "="*60)
print("NOMBRES ASIGNADOS A CADA CLÚSTER (basado en datos reales)")
print("="*60)
for cid in sorted(nombres_clusters.keys()):
    row = profile[profile['prediction'] == cid].iloc[0]
    print(f"\\nClúster {cid}: {nombres_clusters[cid]}")
    print(f"  → Estancia promedio:          {row['estancia']:.2f} días")
    print(f"  → Medicaciones promedio:      {row['medicaciones']:.2f}")
    print(f"  → Lab procedures promedio:    {row['lab_procs']:.2f}")
    print(f"  → Diagnósticos promedio:      {row['diagnosticos']:.2f}")
    print(f"  → Hospitalizaciones previas:  {row['inpatient']:.2f}")
    print(f"  → Emergencias previas:        {row['emergency']:.2f}")
"""
    cells[i] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source_4.split('\n')]
    }
    md_4 = """### Interpretación Clínica por Clúster
Los nombres y valores arriba son calculados dinámicamente. La interpretación clínica debe redactarse observando los prints anteriores. Como guía general:
- El clúster con mayor `inpatient` + `emergency` representa pacientes con historial de múltiples internaciones, alta demanda del sistema de salud y probable comorbilidad avanzada.
- El clúster con mayor `estancia` + `medicaciones` representa casos agudos que requieren hospitalización prolongada y manejo farmacológico intensivo.
- El clúster restante representa el perfil más frecuente: pacientes en control, con estancias cortas y baja recurrencia, que constituyen la mayoría de la población diabética ambulatoria.
- Estos hallazgos son consistentes con la distribución demográfica observada en la exploración inicial (74% Caucasian, predominancia de adultos mayores de 60-80 años).
"""
    cells.insert(i+1, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in md_4.split('\n')]
    })
else:
    print("Warning: Corrección 4 text not found")

# CORRECCIÓN 5: PC2 vs PC3
i, c = find_cell(cells, "La vista 2D muestra la separación y la localización de los centroides")
if i != -1:
    source_5 = """# Vista alternativa: PC2 vs PC3
third = udf(lambda v: float(v[2]), DoubleType())

pdf_km_alt = predictions_km.select(
    second("pcaFeatures").alias("PC2"),
    third("pcaFeatures").alias("PC3"),
    "prediction"
).sample(fraction=0.5, seed=42).toPandas()

plt.figure(figsize=(10, 8))
sns.scatterplot(data=pdf_km_alt, x="PC2", y="PC3", hue="prediction", 
                palette="tab10", s=15, alpha=0.3, legend="full")
plt.title("Visualización PCA: PC2 vs PC3 (vista alternativa)")
plt.xlabel("PC2")
plt.ylabel("PC3")
plt.legend(title="Clúster")
plt.show()
"""
    md_5 = "Si esta vista muestra grupos más separados que PC1 vs PC2, indica que la mayor varianza discriminante entre clústeres no está capturada en el primer componente principal. Comparar ambas vistas ayuda a entender qué dimensiones del espacio latente separan mejor los perfiles de pacientes.\n"
    
    cells.insert(i+1, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source_5.split('\n')]
    })
    cells.insert(i+2, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [md_5]
    })
else:
    print("Warning: Corrección 5 text not found")

# CORRECCIÓN 6: Conclusiones
i, c = find_cell(cells, "El 74% de pacientes es de raza Caucasian. La edad dominante es el grupo [70-80)")
if i != -1:
    new_text = "### Conclusión: Variables Categóricas\nEl 74% de pacientes es de raza Caucasian y la edad dominante es el grupo [70-80). Un gran porcentaje no tiene cambios en medicaciones ni resultados recientes de A1C, lo que sugiere baja intervención farmacológica en parte de la población. Este desbalance demográfico se verá reflejado en los clústeres: se espera que al menos un grupo capture el perfil mayoritario de adulto mayor caucásico sin cambio de medicación.\n"
    c['source'] = [line + '\n' for line in new_text.split('\n')]
else:
    print("Warning: Corrección 6A text not found")

i, c = find_cell(cells, "Logramos transformar el campo ruidoso en un texto limpio, expandiendo correctamente contracciones")
if i != -1:
    new_text = """### ¿Por qué NO aplicar dropDuplicates en texto clínico aquí?
A diferencia de tweets o reseñas, en medicina muchos pacientes comparten *exactamente* la misma combinación de diagnósticos. Eliminar duplicados borraría miles de pacientes reales con la misma patología, destruyendo el volumen del dataset.

### Conclusión del Pipeline NLP
Logramos transformar el campo ruidoso en texto clínico limpio, expandiendo contracciones y eliminando todos los patrones de ruido identificados. La limpieza textual asegura que los tokens usados en los WordClouds por clúster representen patrones clínicos reales (diagnósticos ICD-9, especialidades) y no artefactos tipográficos.
"""
    c['source'] = [line + '\n' for line in new_text.split('\n')]
else:
    print("Warning: Corrección 6B text not found")

i, c = find_cell(cells, "Con K=10 componentes capturamos un gran porcentaje de la varianza.")
if i != -1:
    new_text = "### Conclusión Preprocesamiento\nCon K=10 componentes PCA capturamos la varianza suficiente para separar patrones clínicos. El uso de MinMaxScaler sobre variables identificadas en la exploración (especialmente num_lab_procedures y num_medications, que presentaron alta varianza y sesgo) mejora directamente la cohesión interna de los clústeres medida por la silueta. Sin este escalado, esas variables dominarían el espacio de features y distorsionarían los grupos.\n"
    c['source'] = [line + '\n' for line in new_text.split('\n')]
else:
    print("Warning: Corrección 6C text not found")

i, c = find_cell(cells, "El preprocesamiento NLP estructurado es vital")
if i != -1:
    new_text = """## Conclusiones Generales

1. **Exploración:** El dataset revela una población predominantemente Caucasian (74%), adulta mayor (60-80 años), con alta varianza en procedimientos y medicaciones. Estas características condicionan directamente la forma de los clústeres.

2. **Preparación textual:** El pipeline NLP estructurado, aplicado sobre la combinación de especialidad médica y diagnósticos ICD-9, permitió obtener tokens clínicos limpios. Los códigos V y E del ICD-9 generaban letras sueltas que fueron correctamente eliminadas con stopwords personalizadas.

3. **Clustering:** Los algoritmos K-Means y Bisecting K-Means, con MinMaxScaler y PCA, identificaron grupos de pacientes coherentes con la realidad hospitalaria. El coeficiente de silueta obtenido (ver celda de verificación) confirma o alerta sobre la calidad de la separación.

4. **Implicación clínica:** Los clústeres permiten identificar pacientes de alto riesgo (alta recurrencia hospitalaria), pacientes en evaluación intensiva y pacientes en control rutinario. Esta segmentación no supervisada puede apoyar decisiones de priorización de recursos hospitalarios y diseño de programas de seguimiento diferenciado para pacientes diabéticos.
"""
    c['source'] = [line + '\n' for line in new_text.split('\n')]
else:
    print("Warning: Corrección 6D text not found")

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook updated successfully.")
