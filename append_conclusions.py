import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_text = """## Conclusiones Generales

1. **Exploración:** El dataset revela una población predominantemente Caucasian (74%), adulta mayor (60-80 años), con alta varianza en procedimientos y medicaciones. Estas características condicionan directamente la forma de los clústeres.

2. **Preparación textual:** El pipeline NLP estructurado, aplicado sobre la combinación de especialidad médica y diagnósticos ICD-9, permitió obtener tokens clínicos limpios. Los códigos V y E del ICD-9 generaban letras sueltas que fueron correctamente eliminadas con stopwords personalizadas.

3. **Clustering:** Los algoritmos K-Means y Bisecting K-Means, con MinMaxScaler y PCA, identificaron grupos de pacientes coherentes con la realidad hospitalaria. El coeficiente de silueta obtenido (ver celda de verificación) confirma o alerta sobre la calidad de la separación.

4. **Implicación clínica:** Los clústeres permiten identificar pacientes de alto riesgo (alta recurrencia hospitalaria), pacientes en evaluación intensiva y pacientes en control rutinario. Esta segmentación no supervisada puede apoyar decisiones de priorización de recursos hospitalarios y diseño de programas de seguimiento diferenciado para pacientes diabéticos.
"""

nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + '\n' for line in new_text.split('\n')]
})

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Appended Conclusiones Generales.")
