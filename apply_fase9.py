import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = """import re
import contractions as contractions_lib
from pyspark.sql.functions import regexp_replace, trim, lower, col

# ── Paso 1-4: limpiezas simples con regexp en Spark ──────────────────────────
df_step = df_text
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_clinico', r'@\w+',   ''))
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio',  r'#\w+',   ''))
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio',  r'<.*?>',  ''))
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio',  r'http\S+',''))

# ── Paso 5: expandir contracciones en Pandas (sin UDF, sin pyarrow) ──────────
# Traemos solo la columna que necesitamos a Pandas, procesamos, y reconstruimos
pd_tmp = df_step.select('texto_limpio').toPandas()
pd_tmp['texto_limpio'] = pd_tmp['texto_limpio'].apply(
    lambda x: contractions_lib.fix(x) if isinstance(x, str) else x
)
# Reconstruimos el DataFrame de Spark reemplazando la columna procesada
from pyspark.sql import Row
pd_full = df_step.toPandas()
pd_full['texto_limpio'] = pd_tmp['texto_limpio'].values
df_step = spark.createDataFrame(pd_full)

# ── Paso 6: corregir palabras mal escritas ────────────────────────────────────
dict_correcciones = {
    'diabetis':    'diabetes',
    'hosptial':    'hospital',
    'medicacion':  'medication',
    'admision':    'admission',
    'especialidad':'specialty',
}
dict_correcciones.update(correcciones_dinamicas)
for wrong, right in dict_correcciones.items():
    df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', wrong, right))

# ── Pasos 7-9: eliminar no-alfa, normalizar espacios, minúsculas ──────────────
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'[^a-zA-Z\s]', ' '))
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'\s+', ' '))
df_step = df_step.withColumn('texto_limpio', lower(trim(col('texto_limpio'))))

# ── Filtrar cadenas vacías y cachear ─────────────────────────────────────────
df_clean = df_step.filter(col('texto_limpio') != '')
df_clean = df_clean.cache()
print(f"Registros tras limpieza: {df_clean.count()}")"""

replaced = False
for c in nb['cells']:
    if c.get('cell_type') == 'code':
        source = "".join(c.get('source', []))
        if "df_step = df_text" in source and "df_clean = df_step.filter(col('texto_limpio') != '')" in source:
            c['source'] = [line + '\n' for line in new_source.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
            replaced = True
            break

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

if replaced:
    print("Celda reemplazada exitosamente por la version con toPandas completo.")
else:
    print("No se encontró la celda para reemplazar.")
