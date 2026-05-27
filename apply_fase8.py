import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = """from pyspark.sql.functions import regexp_replace, trim, lower, col, udf
from pyspark.sql.types import StringType
import contractions as contractions_lib

df_step = df_text

# Paso 1: Eliminar menciones @
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_clinico', r'@\w+', ''))
# Paso 2: Eliminar hashtags
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'#\w+', ''))
# Paso 3: Eliminar HTML
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'<.*?>', ''))
# Paso 4: Eliminar URLs
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'http\S+', ''))
# Paso 5: Expandir contracciones con UDF clásico (sin pyarrow)
def expand_contractions_fn(text):
    try:
        return contractions_lib.fix(text) if isinstance(text, str) else text
    except Exception:
        return text

expand_udf = udf(expand_contractions_fn, StringType())
df_step = df_step.withColumn('texto_limpio', expand_udf(col('texto_limpio')))

# Paso 6: Corregir palabras mal escritas
dict_correcciones = {
    'diabetis': 'diabetes',
    'hosptial': 'hospital',
    'medicacion': 'medication',
    'admision': 'admission',
    'especialidad': 'specialty',
}
dict_correcciones.update(correcciones_dinamicas)
for wrong, right in dict_correcciones.items():
    df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', wrong, right))

# Paso 7: Eliminar caracteres no alfabéticos
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'[^a-zA-Z\s]', ' '))
# Paso 8: Normalizar espacios múltiples
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'\s+', ' '))
# Paso 9: Minúsculas y trim
df_step = df_step.withColumn('texto_limpio', lower(trim(col('texto_limpio'))))

# Filtrar cadenas vacías
df_clean = df_step.filter(col('texto_limpio') != '')
df_clean = df_clean.cache()
print(f"Registros tras limpieza: {df_clean.count()}")"""

replaced = False
for c in nb['cells']:
    if c.get('cell_type') == 'code':
        source = "".join(c.get('source', []))
        # The previous cell starts with:
        if "from pyspark.sql.functions import regexp_replace, trim, lower, col" in source and "df_clean = df_step.filter" in source:
            c['source'] = [line + '\n' for line in new_source.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
            replaced = True
            break

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

if replaced:
    print("Celda reemplazada exitosamente por la version con UDF clasico.")
else:
    print("No se encontró la celda para reemplazar.")
