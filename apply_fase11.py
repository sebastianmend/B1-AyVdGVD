import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Operación 1: Reemplazar celda 19
source_19 = """from pyspark.sql.functions import explode, split, length, lower as lower_fn
from spellchecker import SpellChecker

print("Análisis de errores ortográficos en el dataset completo...")
words_df_pre = df_text.select(explode(split(lower_fn(col('texto_clinico')), r'\\s+')).alias('word'))
words_df_pre = words_df_pre.filter(length(col('word')) > 2)
word_counts = words_df_pre.groupBy('word').count().orderBy('count', ascending=False).limit(200).toPandas()

spell = SpellChecker()
unknown = spell.unknown(word_counts['word'].tolist())

print("Top palabras posiblemente erróneas y sugerencias:")
correcciones_dinamicas = {}
encontradas = 0
for _, row in word_counts.iterrows():
    w = row['word']
    if w in unknown:
        sugerencia = spell.correction(w)
        if sugerencia and sugerencia != w:
            print(f"Original: {w:<15} | Sugerencia: {sugerencia:<15} | Frecuencia: {row['count']}")
            correcciones_dinamicas[w] = sugerencia
            encontradas += 1
            if encontradas >= 20:
                break"""
nb['cells'][19]['source'] = [line + '\n' for line in source_19.split('\n')]
nb['cells'][19]['source'][-1] = nb['cells'][19]['source'][-1].rstrip('\n')

# Operación 2: Reemplazar celda 20
source_20 = """from pyspark.sql.functions import regexp_replace, trim, lower, col

df_step = df_text

# Paso 1: Eliminar menciones @
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_clinico', r'@\\w+', ''))
# Paso 2: Eliminar hashtags
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'#\\w+', ''))
# Paso 3: Eliminar HTML
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'<.*?>', ''))
# Paso 4: Eliminar URLs
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'http\\S+', ''))
# Paso 5: Expandir contracciones con regexp_replace
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
# Paso 7: Eliminar caracteres no alfabéticos
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'[^a-zA-Z\\s]', ' '))
# Paso 8: Normalizar espacios múltiples
df_step = df_step.withColumn('texto_limpio', regexp_replace('texto_limpio', r'\\s+', ' '))
# Paso 9: Minúsculas y trim
df_step = df_step.withColumn('texto_limpio', lower(trim(col('texto_limpio'))))

# Filtrar cadenas vacías y cachear
df_clean = df_step.filter(col('texto_limpio') != '')
df_clean = df_clean.cache()
print(f"Registros tras limpieza: {df_clean.count()}")"""
nb['cells'][20]['source'] = [line + '\n' for line in source_20.split('\n')]
nb['cells'][20]['source'][-1] = nb['cells'][20]['source'][-1].rstrip('\n')

# Operación 3: Eliminar celda 21
del nb['cells'][21]

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Celdas 19 y 20 reemplazadas. Celda 21 eliminada exitosamente.")
