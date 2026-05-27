import json
import re

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def find_cell(cells, substring):
    for i, c in enumerate(cells):
        if c is None: continue
        text = "".join(c.get('source', []))
        if substring in text:
            return i, c
    return -1, None

# 1. PROBLEMA 1: El notebook est desordenado
# Reordena las celdas para que sigan ese orden lgico de arriba hacia abajo:
# Ttulo e integrantes -> Imports y SparkSession -> Carga del dataset -> Exploracin inicial -> 
# Descripcin textual -> Variables numricas -> Variables categricas -> 
# Seccin NLP (primero df_text, luego verificacin de patrones, luego SpellChecker, luego pipeline de pasos 1-9) -> El resto igual.

# 2. PROBLEMA 2: Hay ruido artificial inyectado que no debe estar.
# En la celda que crea df_text, elimina completamente el bloque que inyecta ruido.
idx, c_dftext = find_cell(cells, "df_text = df.withColumn('texto_clinico', concat_ws(")
if idx != -1:
    source = "".join(c_dftext['source'])
    source = re.sub(r'# 2\. Inyectar ruido simulado.*?\)\n', '', source, flags=re.DOTALL)
    c_dftext['source'] = [line + '\n' for line in source.split('\n') if line]

# Tambin elimina id_ruido y captura de val_orig, etc.
idx, c_pipe = find_cell(cells, "df_step = df_text")
if idx != -1:
    source = "".join(c_pipe['source'])
    source = re.sub(r'id_ruido = 110939484\n', '', source)
    source = re.sub(r"val_orig = df_step\.filter\(col\('encounter_id'\) == id_ruido\)\.select\('texto_clinico'\)\.collect\(\)\[0\]\[0\]\n", '', source)
    
    for p in range(1, 10):
        source = re.sub(r"val_" + str(p) + r" = df_step\.filter\(col\('encounter_id'\) == id_ruido\)\.select\('texto_limpio'\)\.collect\(\)\[0\]\[0\]\n", '', source)
        source = re.sub(r"pasos\.append\(\{.*?\}\)\n", f'# Ejecutado Paso {p}\n', source)
    
    source = re.sub(r'pasos = \[\]\n', '', source)
    source = re.sub(r'from IPython\.display import display\n', '', source)
    source = re.sub(r'tabla_pasos = pd\.DataFrame\(pasos\)\n', '', source)
    source = re.sub(r'display\(tabla_pasos\)\n', '', source)
    
    c_pipe['source'] = [line + '\n' for line in source.split('\n') if line.strip() != '']

# 3. PROBLEMA 3: Bug en la celda de WordClouds
idx_wc, c_wc = find_cell(cells, "axes[cid].imshow(wc")
if idx_wc != -1:
    source = "".join(c_wc['source'])
    # Replace the loop
    source = re.sub(r'for .*?:', "for idx, cid_val in enumerate(sorted(predictions_km.select('prediction').distinct().toPandas()['prediction'].tolist())):", source)
    source = source.replace("sample[sample['prediction'] == cid]", "sample[sample['prediction'] == cid_val]")
    source = source.replace("axes[cid].imshow", "axes[idx].imshow")
    source = source.replace("axes[cid].axis", "axes[idx].axis")
    source = source.replace("axes[cid].set_title(f\"Clúster {cid}", "axes[idx].set_title(f\"Clúster {cid_val}")
    c_wc['source'] = [line + '\n' for line in source.split('\n') if line]

# 4. PROBLEMA 4: La celda de SpellChecker usa df_step antes de existir
idx_spell, c_spell = find_cell(cells, "errores ortogr")
if idx_spell != -1:
    source = "".join(c_spell['source'])
    source = source.replace("col('texto_limpio')", "col('texto_clinico')")
    c_spell['source'] = [line + '\n' for line in source.split('\n') if line]

# Ahora, el REORDENAMIENTO de todas estas celdas (PROBLEMA 1 y PROBLEMA 4 posicin).
# Necesitamos identificar y extraer las celdas desordenadas:
# - Title/Imports (podran estar en index 0 o ms abajo)
# - df_text (creacin)
# - pipeline (df_step = df_text)
# - spell checker
# - patrones

# First, extract the specific NLP cells from the notebook
nlp_cells_to_extract = []
def extract_and_remove(substring):
    global cells
    idx, c = find_cell(cells, substring)
    if idx != -1:
        cells.pop(idx)
        return c
    return None

c_dftext = extract_and_remove("df_text = df.withColumn('texto_clinico', concat_ws(")
c_patrones = extract_and_remove("--- Verificación de Patrones en texto_clinico ---")
c_top30 = extract_and_remove("Top 30 palabras antes de limpieza ortogr")
c_spell = extract_and_remove("errores ortogr")
c_pipe = extract_and_remove("df_step = df_text")

# We also need to split `c_pipe` into two parts: "df_step = df_text" and "# Paso 1..."
if c_pipe:
    source_pipe = "".join(c_pipe['source'])
    parts = source_pipe.split("# Paso 1: Eliminar menciones @")
    if len(parts) == 2:
        c_pipe_part1 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in parts[0].split('\n') if line]
        }
        c_pipe_part2 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in ("# Paso 1: Eliminar menciones @" + parts[1]).split('\n') if line]
        }
    else:
        c_pipe_part1 = c_pipe
        c_pipe_part2 = None
else:
    c_pipe_part1 = None
    c_pipe_part2 = None


# We also need to ensure the Notebook starts with:
# Title/Integrantes -> Imports -> Carga dataset
# Right now, since we removed the NLP cells that were at the top, the top should naturally be Title.
# Let's verify the first cell is the title.
# If not, let's find the title and move it to the top.
c_title = extract_and_remove("Integrantes:")
c_imports = extract_and_remove("import os\nimport sys\n")
c_carga = extract_and_remove("file_path = \"diabetes_v1_.csv\"")

# Let's insert them at the very beginning in order
insert_idx = 0
if c_title: cells.insert(insert_idx, c_title); insert_idx += 1
if c_imports: cells.insert(insert_idx, c_imports); insert_idx += 1
# Also the Markdown "## Carga del Dataset"
c_carga_md = extract_and_remove("## Carga del Dataset")
if c_carga_md: cells.insert(insert_idx, c_carga_md); insert_idx += 1
if c_carga: cells.insert(insert_idx, c_carga); insert_idx += 1

# Now find the "## Preparación y Limpieza del Texto (Pipeline NLP)" header and insert the NLP cells after it.
idx_nlp_header, _ = find_cell(cells, "## Preparación y Limpieza del Texto (Pipeline NLP)")

if idx_nlp_header != -1:
    insert_idx = idx_nlp_header + 1
    # Orden correcto: df_text -> patrones -> top30 -> pipe_part1 (df_step=df_text) -> spell -> pipe_part2 (Pasos 1-9)
    if c_dftext: cells.insert(insert_idx, c_dftext); insert_idx += 1
    if c_patrones: cells.insert(insert_idx, c_patrones); insert_idx += 1
    if c_top30: cells.insert(insert_idx, c_top30); insert_idx += 1
    if c_pipe_part1: cells.insert(insert_idx, c_pipe_part1); insert_idx += 1
    if c_spell: cells.insert(insert_idx, c_spell); insert_idx += 1
    if c_pipe_part2: cells.insert(insert_idx, c_pipe_part2); insert_idx += 1

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Modificaciones de Fase 5 aplicadas exitosamente.")
