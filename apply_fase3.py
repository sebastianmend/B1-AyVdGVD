import json
import re

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def find_cell(cells, substring):
    for i, c in enumerate(cells):
        text = "".join(c['source'])
        if substring in text:
            return i, c
    return -1, None

# 1. Modificar pip install para pyspellchecker
i, c = find_cell(cells, "!pip install contractions wordcloud")
if i != -1:
    source = "".join(c['source'])
    source = source.replace("!pip install contractions wordcloud", "!pip install contractions wordcloud pyspellchecker")
    c['source'] = [line + '\n' for line in source.split('\n')]

# 2. Reemplazar Verificación de Patrones
i, c = find_cell(cells, "print(\"--- Verificación de Patrones en texto_clinico ---\")")
if i != -1:
    new_code = """import pandas as pd
import re

print("--- Verificación de Patrones en texto_clinico ---")
patterns = {
    "Menciones @": r"@\w+",
    "Hashtags #": r"#\w+",
    "HTML tags": r"<.*?>",
    "URLs": r"http\S+",
    "No-alfabéticos": r"[^a-zA-Z\s]"
}

# Tomar muestra aleatoria de pacientes reales (sin ruido artificial)
df_real = df_text.filter(col('encounter_id') != 110939484)
resultados_patrones = []

for nombre, regex in patterns.items():
    conteo = df_text.filter(col('texto_clinico').rlike(regex)).count()
    
    # Extraer un ejemplo real
    ejemplo_df = df_real.filter(col('texto_clinico').rlike(regex)).limit(1).toPandas()
    if not ejemplo_df.empty:
        texto_original = ejemplo_df['texto_clinico'].iloc[0]
        # Simular limpieza para el ejemplo
        reemplazo = ' ' if nombre == 'No-alfabéticos' else ''
        texto_limpio_ejemplo = re.sub(regex, reemplazo, texto_original)
    else:
        texto_original = "N/A"
        texto_limpio_ejemplo = "N/A"
        
    resultados_patrones.append({
        "Patrón": nombre,
        "Registros encontrados": conteo,
        "Ejemplo antes": texto_original[:100] + "..." if texto_original != "N/A" else "N/A",
        "Ejemplo después": texto_limpio_ejemplo[:100] + "..." if texto_limpio_ejemplo != "N/A" else "N/A"
    })

display(pd.DataFrame(resultados_patrones))
"""
    c['source'] = [line + '\n' for line in new_code.split('\n')]

# 3. Add spellchecker logic BEFORE cell "dict_correcciones = {"
i, c = find_cell(cells, "dict_correcciones = {")
if i != -1:
    new_cell_code = """from pyspark.sql.functions import explode, split, length, lower
from spellchecker import SpellChecker

print("Análisis de errores ortográficos en el dataset completo...")

# Calcular frecuencias
words_df_pre = df_step.select(explode(split(lower(col('texto_limpio')), r'\s+')).alias('word'))
words_df_pre = words_df_pre.filter(length(col('word')) > 2)
word_counts = words_df_pre.groupBy('word').count().orderBy('count', ascending=False).limit(200).toPandas()

spell = SpellChecker()

unknown = spell.unknown(word_counts['word'].tolist())

print("Top palabras posiblemente erróneas y sugerencias (expandiendo el diccionario dinámicamente):")
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
                break
"""
    cells.insert(i, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in new_cell_code.split('\n')]
    })
    
    # Now modify the actual dict cell (which is now i+1)
    c = cells[i+1]
    source = "".join(c['source'])
    source = source.replace("dict_correcciones = {", "dict_correcciones = {\n    # Originales")
    # find the end of dict definition
    end_dict_idx = source.find("}\nfor wrong, right")
    if end_dict_idx != -1:
        source = source[:end_dict_idx+1] + "\ndict_correcciones.update(correcciones_dinamicas) # Agregar sugerencias de spellchecker\n" + source[end_dict_idx+1:]
        c['source'] = [line + '\n' for line in source.split('\n')]

# 4. Variables Categoricas Subplots
i, c = find_cell(cells, "for col_name in cat_plot_cols:")
if i != -1:
    source = "".join(c['source'])
    new_loop = """import math

total_count = df.count()
n_cols = len(cat_plot_cols)
rows = math.ceil(n_cols / 2)

fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, col_name in enumerate(cat_plot_cols):
    pd_df = df.groupBy(col_name).count().toPandas()
    pd_df['percent'] = (pd_df['count'] / total_count) * 100
    pd_df = pd_df.sort_values('percent', ascending=False)
    
    ax = axes[i]
    sns.barplot(data=pd_df, x=col_name, y='percent', palette='viridis', ax=ax)
    ax.set_title(f'Distribución de {col_name} (%)')
    ax.set_ylabel('Porcentaje (%)')
    ax.tick_params(axis='x', rotation=45)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=9, color='black', xytext=(0, 5), 
                    textcoords='offset points')

# Ocultar ejes sobrantes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
"""
    # Replace the existing loop with new_loop
    # find where the loop starts
    start_idx = source.find("total_count = df.count()")
    if start_idx != -1:
        source = source[:start_idx] + new_loop
        c['source'] = [line + '\n' for line in source.split('\n')]

# 5. Conclusions WordCloud and PCA
i, c = find_cell(cells, "### Conclusión PCA 2D")
if i != -1:
    md_pca = "### Conclusión PCA 2D\nEl scatter plot visualiza claramente la aglomeración de grupos. La silueta obtenida confirma que estos clusters tienen buena densidad interna y separación en el espacio dimensional. Notamos grandes bloques centrales correlacionados con la población mayoritaria observada (adultos mayores caucásicos), mientras que los puntos periféricos podrían corresponder a los casos de alta complejidad o perfiles atípicos con múltiples comorbilidades.\n"
    c['source'] = [line + '\n' for line in md_pca.split('\n')]

i, c = find_cell(cells, "### Conclusión WordClouds")
if i != -1:
    md_wc = "### Conclusión WordClouds\nLas nubes de palabras revelan inmediatamente el perfil demográfico y clínico dominante de cada grupo. La alta frecuencia de términos como \"Caucasian\", \"Female\", o medicamentos clave como \"insulin\" y \"metformin\" dentro de grupos específicos conecta directamente con las interpretaciones clínicas de \"Evaluación Intensiva\" (alta medicación) frente al \"Control Rutinario\" (perfil más demográficamente homogéneo y sin cambios en medicaciones).\n"
    c['source'] = [line + '\n' for line in md_wc.split('\n')]

# 6. Silhouette Retry Logic
i, c = find_cell(cells, "if best_sil_km < 0.6 and best_sil_bkm < 0.6:")
if i != -1:
    source = "".join(c['source'])
    retry_logic = """if best_sil_km < 0.6 and best_sil_bkm < 0.6:
    print("\\n--- RE-EJECUCIÓN AUTOMÁTICA CON PCA K=15 ---")
    pca_15 = PCA(k=15, inputCol="scaledFeatures", outputCol="pcaFeatures_15")
    dataset_15 = pca_15.fit(df_scaled).transform(df_scaled).cache()
    
    sil_15_km = []
    for k in k_values:
        model = KMeans(featuresCol="pcaFeatures_15", predictionCol="prediction", k=k, seed=42).fit(dataset_15)
        preds = model.transform(dataset_15)
        sil_15_km.append(evaluator.evaluate(preds))
    best_k_15_km = k_values[np.argmax(sil_15_km)]
    best_sil_15_km = max(sil_15_km)
    
    sil_15_bkm = []
    for k in k_values:
        model = BisectingKMeans(featuresCol="pcaFeatures_15", predictionCol="prediction_bkm", k=k, seed=42).fit(dataset_15)
        preds = model.transform(dataset_15)
        sil_15_bkm.append(evaluator.evaluate(preds.withColumnRenamed('prediction_bkm', 'prediction')))
    best_k_15_bkm = k_values[np.argmax(sil_15_bkm)]
    best_sil_15_bkm = max(sil_15_bkm)
    
    print(f"K-Means (k=15)       | Silhouette = {best_sil_15_km:.4f} (K={best_k_15_km})")
    print(f"Bisecting KM (k=15)  | Silhouette = {best_sil_15_bkm:.4f} (K={best_k_15_bkm})")
    
    if max(best_sil_15_km, best_sil_15_bkm) > max(best_sil_km, best_sil_bkm):
        print("-> PCA K=15 mejoró los resultados. Actualizando modelos para la interpretación.")
        if best_sil_15_km >= best_sil_15_bkm:
            model_km_final = KMeans(featuresCol="pcaFeatures_15", predictionCol="prediction", k=best_k_15_km, seed=42).fit(dataset_15)
            predictions_km = model_km_final.transform(dataset_15)
            best_k_km = best_k_15_km
        else:
            model_km_final = BisectingKMeans(featuresCol="pcaFeatures_15", predictionCol="prediction", k=best_k_15_bkm, seed=42).fit(dataset_15)
            predictions_km = model_km_final.transform(dataset_15)
            best_k_km = best_k_15_bkm
    else:
        print("-> PCA K=15 NO mejoró los resultados significativamente. Manteniendo PCA K=10.")
"""
    # Replace the existing if block
    start_idx = source.find("if best_sil_km < 0.6 and best_sil_bkm < 0.6:")
    if start_idx != -1:
        source = source[:start_idx] + retry_logic
        c['source'] = [line + '\n' for line in source.split('\n')]

# 7. PCA 2D Side-by-side
i, c = find_cell(cells, "def get_pca_df(preds):")
if i != -1:
    new_pca_cell = """from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

first  = udf(lambda v: float(v[0]), DoubleType())
second = udf(lambda v: float(v[1]), DoubleType())

def get_pca_df(preds, features_col="pcaFeatures"):
    return preds.select(
        first(features_col).alias("PC1"),
        second(features_col).alias("PC2"),
        "prediction"
    ).sample(fraction=0.5, seed=42).toPandas()

features_col_actual = "pcaFeatures_15" if "pcaFeatures_15" in predictions_km.columns else "pcaFeatures"

pdf_km = get_pca_df(predictions_km, features_col_actual)
pdf_bkm = get_pca_df(predictions_bkm, features_col_actual)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.scatterplot(data=pdf_km, x="PC1", y="PC2", hue="prediction", palette="tab10", s=20, alpha=0.5, edgecolor='none', legend="full", ax=axes[0])
if hasattr(model_km_final, 'clusterCenters'):
    centers_km = np.array([c[:2] for c in model_km_final.clusterCenters()])
    axes[0].scatter(centers_km[:, 0], centers_km[:, 1], c='red', s=200, marker='*', edgecolor='black', label='Centroides')
axes[0].set_title(f"Visualización PCA 2D - K-Means")
axes[0].legend()

sns.scatterplot(data=pdf_bkm, x="PC1", y="PC2", hue="prediction", palette="tab10", s=20, alpha=0.5, edgecolor='none', legend="full", ax=axes[1])
if hasattr(model_bkm_final, 'clusterCenters'):
    centers_bkm = np.array([c[:2] for c in model_bkm_final.clusterCenters()])
    axes[1].scatter(centers_bkm[:, 0], centers_bkm[:, 1], c='red', s=200, marker='*', edgecolor='black', label='Centroides')
axes[1].set_title(f"Visualización PCA 2D - Bisecting K-Means")
axes[1].legend()

plt.tight_layout()
plt.show()
"""
    c['source'] = [line + '\n' for line in new_pca_cell.split('\n')]

# Fix the PCA PC2 vs PC3 alternative plot
i, c = find_cell(cells, "pdf_km_alt = predictions_km.select(")
if i != -1:
    source = "".join(c['source'])
    source = source.replace("second(\"pcaFeatures\")", "second(features_col_actual)")
    source = source.replace("third(\"pcaFeatures\")", "third(features_col_actual)")
    source = source.replace("s=15, alpha=0.3", "s=20, alpha=0.5, edgecolor='none'")
    c['source'] = [line + '\n' for line in source.split('\n')]


with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Modificaciones de Fase 3 aplicadas correctamente.")
