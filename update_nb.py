import json

file_path = "analisis_diabetes.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_carga = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 💡 Conclusión de la Carga de Datos\n",
    "\n",
    "El dataset se ha cargado correctamente infiriendo un esquema inicial de **50 columnas**. Observamos una rica mezcla de variables de identificación (`encounter_id`, `patient_nbr`), demográficas (`race`, `gender`, `age`), y una gran cantidad de atributos clínicos. Esto confirma que estamos ante un conjunto de datos robusto, ideal para descubrir patrones clínicos relevantes y realizar modelos analíticos de alto valor."
   ]
}

cell_num = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📊 Conclusión del Resumen Numérico\n",
    "\n",
    "Al observar las métricas centrales, confirmamos que contamos con una base sólida de más de **50,000 registros**. Identificamos un total de **13 variables numéricas**. Las estadísticas descriptivas básicas, como la media y la desviación estándar, nos indican que existe una variabilidad importante en los datos (por ejemplo, en el tipo de admisión o tiempo en hospital), lo que sugiere que nuestra población de pacientes es diversa en cuanto a la severidad y manejo de su condición médica."
   ]
}

cell_cat = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📌 Conclusión del Resumen Categórico\n",
    "\n",
    "Este primer vistazo a las **36 variables categóricas** revela insights demográficos inmediatos:\n",
    "\n",
    "- **Raza y Género:** Predomina significativamente la población `Caucasian` (~74%), y existe un balance general entre géneros con una ligera mayoría `Female`.\n",
    "- **Edad:** Existe una concentración natural en grupos de la tercera edad (especialmente `[70-80)` y `[60-70)`). \n",
    "\n",
    "Estos hallazgos son totalmente congruentes con la naturaleza del padecimiento de la diabetes, afectando en mayor proporción a adultos mayores, y nos dan un excelente punto de partida para segmentar y profundizar el análisis."
   ]
}

cells = nb['cells']
new_cells = []

for cell in cells:
    new_cells.append(cell)
    
    if cell['cell_type'] == 'code' and any('df.show(5)' in line for line in cell.get('source', [])):
        new_cells.append(cell_carga)
        
    elif cell['cell_type'] == 'code' and any('No se encontraron variables numéricas' in line for line in cell.get('source', [])):
        new_cells.append(cell_num)
        
    elif cell['cell_type'] == 'code' and any('No se encontraron variables categóricas' in line for line in cell.get('source', [])):
        new_cells.append(cell_cat)

nb['cells'] = new_cells

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
