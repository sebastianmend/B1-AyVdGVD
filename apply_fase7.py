import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = """from pyspark.sql.functions import regexp_replace, trim, lower, col
import contractions as contractions_lib
import pandas as pd

df_step = df_text

# Paso 1: Eliminar menciones @
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_clinico', r'@\w+', ''))
# Paso 2: Eliminar hashtags
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'#\w+', ''))
# Paso 3: Eliminar HTML
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'<.*?>', ''))
# Paso 4: Eliminar URLs
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'http\S+', ''))
# Paso 5: Expandir contracciones en Pandas y regresar a Spark SIN join
pd_texto = df_step.select('texto_limpio').toPandas()
pd_texto['texto_limpio'] = pd_texto['texto_limpio'].apply(
    lambda x: contractions_lib.fix(x) if isinstance(x, str) else x
)
from pyspark.sql.types import StringType
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(StringType())
def expand_contractions_udf(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: contractions_lib.fix(x) if isinstance(x, str) else x)

df_step = df_step.withColumn('texto_limpio', expand_contractions_udf(col('texto_limpio')))

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
        if "# Paso 1: Eliminar menciones @" in source and "df_clean = df_step.filter" in source:
            c['source'] = [line + '\n' for line in new_source.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
            replaced = True
            break

# In case the user also wanted `df_step = df_text` cell replaced which was split,
# the new source has `df_step = df_text`.
# Wait, if I split it earlier into part1 and part2, part1 was `df_step = df_text`.
# Since the new code INCLUDES `df_step = df_text`, I should also remove the standalone `df_step = df_text` cell to avoid duplicate.
if replaced:
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') == 'code':
            src = "".join(c.get('source', [])).strip()
            if src == "df_step = df_text":
                nb['cells'].pop(i)
                break

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

if replaced:
    print("Celda reemplazada exitosamente.")
else:
    print("No se encontró la celda para reemplazar.")
