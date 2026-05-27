import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def find_cell(cells, substring):
    for i, c in enumerate(cells):
        text = "".join(c.get('source', []))
        if substring in text:
            return i, c
    return -1, None

# 1 & 2: Reordenar y arreglar la seccin de NLP
# Primero, busquemos todas las celdas involucradas para extraerlas
idx_header = find_cell(cells, "## Preparacin y Limpieza del Texto (Pipeline NLP)")[0]
idx_patrones, c_patrones = find_cell(cells, "--- Verificacin de Patrones en texto_clinico ---")
idx_top30, c_top30 = find_cell(cells, "Top 30 palabras antes de limpieza ortogrfica:")
idx_spell, c_spell = find_cell(cells, "Anlisis de errores ortogrficos en el dataset completo...")
idx_paso1, c_paso1 = find_cell(cells, "# Paso 1: Eliminar menciones @")

# Extraer las celdas (haciendo copias profundas)
import copy
if idx_patrones != -1: cell_patrones = copy.deepcopy(c_patrones)
else: cell_patrones = None

if idx_top30 != -1: cell_top30 = copy.deepcopy(c_top30)
else: cell_top30 = None

if idx_spell != -1: 
    cell_spell = copy.deepcopy(c_spell)
    # 1. (continuacin) Agregar correcciones_dinamicas = {} al inicio seguro
    source_spell = "".join(cell_spell['source'])
    if "correcciones_dinamicas = {}" not in source_spell:
        source_spell = "correcciones_dinamicas = {}\n" + source_spell
        cell_spell['source'] = [line + '\n' for line in source_spell.split('\n')]
else: cell_spell = None

if idx_paso1 != -1: cell_paso1 = copy.deepcopy(c_paso1)
else: cell_paso1 = None

# Eliminar estas celdas de su ubicacin actual
indices_to_remove = sorted([i for i in [idx_patrones, idx_top30, idx_spell, idx_paso1] if i != -1], reverse=True)
for i in indices_to_remove:
    cells.pop(i)

# Crear la celda de df_text que se perdi
cell_df_text = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from pyspark.sql.functions import col, concat_ws, lit, when\n\n",
        "# 1. Crear la columna de texto clnico\n",
        "df_text = df.withColumn('texto_clinico', concat_ws(' ', col('medical_specialty'), col('diag_1'), col('diag_2'), col('diag_3')))\n\n",
        "# 2. Inyectar ruido simulado\n",
        "ruido = \"Paciente don't react to insulin @doctorHouse #diabetes http://www.hospital.com <br> check glu!\"\n",
        "df_text = df_text.withColumn(\n",
        "    'texto_clinico',\n",
        "    when(col('encounter_id') == 110939484, lit(ruido)).otherwise(col('texto_clinico'))\n",
        ")\n"
    ]
}

# Insertar en el orden correcto despus del header
# Encontramos de nuevo el header porque los ndices cambiaron
idx_header = find_cell(cells, "## Preparacin y Limpieza del Texto (Pipeline NLP)")[0]

insert_idx = idx_header + 1
cells.insert(insert_idx, cell_df_text); insert_idx += 1
if cell_patrones: cells.insert(insert_idx, cell_patrones); insert_idx += 1
if cell_top30: cells.insert(insert_idx, cell_top30); insert_idx += 1
if cell_spell: cells.insert(insert_idx, cell_spell); insert_idx += 1
if cell_paso1: cells.insert(insert_idx, cell_paso1)


# 3. Importar 'when' en Preprocesamiento para ML
i, c = find_cell(cells, "from pyspark.ml import Pipeline")
if i != -1:
    source = "".join(c['source'])
    if "from pyspark.sql.functions import when" not in source:
        source = "from pyspark.sql.functions import when\n" + source
        c['source'] = [line + '\n' for line in source.split('\n')]

# 4. En Visualizacin PCA 2D, aadir features_col_bkm
i, c = find_cell(cells, "features_col_actual = \"pcaFeatures_15\"")
if i != -1:
    source = "".join(c['source'])
    # Buscar get_pca_df(predictions_bkm...)
    if "features_col_bkm" not in source:
        source = source.replace(
            "features_col_actual = \"pcaFeatures_15\" if \"pcaFeatures_15\" in predictions_km.columns else \"pcaFeatures\"",
            "features_col_actual = \"pcaFeatures_15\" if \"pcaFeatures_15\" in predictions_km.columns else \"pcaFeatures\"\n" + 
            "features_col_bkm = \"pcaFeatures_15\" if \"pcaFeatures_15\" in predictions_bkm.columns else \"pcaFeatures\""
        )
        source = source.replace(
            "pdf_bkm = get_pca_df(predictions_bkm, features_col_actual)",
            "pdf_bkm = get_pca_df(predictions_bkm, features_col_bkm)"
        )
        c['source'] = [line + '\n' for line in source.split('\n')]

# 5. En Interpretacin de Clsteres, cambiar el loop for cid in range(best_k_km):
i, c = find_cell(cells, "for cid in range(best_k_km):")
if i != -1:
    source = "".join(c['source'])
    source = source.replace("for cid in range(best_k_km):", "for cid_val in sorted(predictions_km.select('prediction').distinct().toPandas()['prediction'].tolist()):")
    # Y reemplazar uso de cid a cid_val solo dentro de este contexto.
    # El bucle usa "print(f\"\n--- Clster {cid} ---\")" y "col(\"prediction\") == cid"
    source = source.replace("Clster {cid}", "Clster {cid_val}")
    source = source.replace("col(\"prediction\") == cid", "col(\"prediction\") == cid_val")
    c['source'] = [line + '\n' for line in source.split('\n')]

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Modificaciones de Fase 4 aplicadas exitosamente.")
