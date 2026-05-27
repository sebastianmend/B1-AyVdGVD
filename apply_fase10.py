import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = """import re
import contractions as contractions_lib
from pyspark.sql.functions import regexp_replace, trim, lower, col

# Todo el pipeline NLP se hace en Spark con regexp_replace puro.
# Las contracciones se expanden con regexp_replace sobre los patrones
# ms comunes del ingls, sin necesitar UDF ni pyarrow.

df_step = df_text

# Paso 1: Eliminar menciones @
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_clinico', r'@\w+', ''))
# Paso 2: Eliminar hashtags
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'#\w+', ''))
# Paso 3: Eliminar HTML
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'<.*?>', ''))
# Paso 4: Eliminar URLs
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'http\S+', ''))

# Paso 5: Expandir contracciones con regexp_replace (sin UDF, sin pyarrow)
contracciones_map = [
    (r"won't", "will not"), (r"can't", "cannot"), (r"n't", " not"),
    (r"'re", " are"), (r"'s", " is"), (r"'d", " would"),
    (r"'ll", " will"), (r"'ve", " have"), (r"'m", " am"),
]
for patron, reemplazo in contracciones_map:
    df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', patron, reemplazo))

# Paso 6: Corregir palabras mal escritas
dict_correcciones = {
    'diabetis':     'diabetes',
    'hosptial':     'hospital',
    'medicacion':   'medication',
    'admision':     'admission',
    'especialidad': 'specialty',
}
dict_correcciones.update(correcciones_dinamicas)
for wrong, right in dict_correcciones.items():
    df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', wrong, right))

# Paso 7: Eliminar caracteres no alfabticos
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'[^a-zA-Z\s]', ' '))
# Paso 8: Normalizar espacios mltiples
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'\s+', ' '))
# Paso 9: Minsculas y trim
df_step = df_step.withColumn('texto_limpio', lower(trim(col('texto_limpio'))))

# Filtrar cadenas vacas y cachear
df_clean = df_step.filter(col('texto_limpio') != '')
df_clean = df_clean.cache()
print(f"Registros tras limpieza: {df_clean.count()}")"""

replaced = False
for c in nb['cells']:
    if c.get('cell_type') == 'code':
        source = "".join(c.get('source', []))
        if "import re" in source and "import contractions as contractions_lib" in source and "df_step = df_text" in source:
            # We fix encoding explicitly using utf-8 literals in string
            c['source'] = [line + '\n' for line in new_source.replace('ms', 'más').replace('ingls', 'inglés').replace('alfabticos', 'alfabéticos').replace('mltiples', 'múltiples').replace('Minsculas', 'Minúsculas').replace('vacas', 'vacías').split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n')
            replaced = True
            break

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

if replaced:
    print("Celda reemplazada exitosamente por la version con regexp_replace puro.")
else:
    print("No se encontró la celda para reemplazar.")
