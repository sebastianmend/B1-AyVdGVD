import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = """from collections import Counter

med_cols = ['metformin', 'insulin', 'glipizide', 'glyburide', 'pioglitazone',
            'rosiglitazone', 'glimepiride', 'repaglinide', 'nateglinide']
token_cols = ['race', 'gender', 'age', 'A1Cresult', 'max_glu_serum', 'change'] + med_cols

sample = predictions_km.select(['prediction'] + token_cols).sample(fraction=0.3, seed=42).toPandas()

cluster_ids = sorted(predictions_km.select('prediction').distinct().toPandas()['prediction'].tolist())

fig, axes = plt.subplots(1, len(cluster_ids), figsize=(6 * len(cluster_ids), 5))
if len(cluster_ids) == 1:
    axes = [axes]

for idx, cid_val in enumerate(cluster_ids):
    sub = sample[sample['prediction'] == cid_val]
    counter = Counter()
    for _, row in sub.iterrows():
        for c in ['race', 'gender', 'age', 'A1Cresult', 'max_glu_serum', 'change']:
            v = str(row[c])
            if v and v != 'Unknown' and v != 'None':
                counter[f"{c}={v}"] += 1
        for m in med_cols:
            v = str(row[m])
            if v not in ('No', 'None', 'nan'):
                counter[f"{m}_{v}"] += 1
    wc = WordCloud(width=600, height=400, background_color='white',
                   colormap='viridis', max_words=40).generate_from_frequencies(counter)
    axes[idx].imshow(wc, interpolation='bilinear')
    axes[idx].axis('off')
    axes[idx].set_title(f"Clúster {cid_val} (n={len(sub)})")

plt.suptitle("Wordclouds por clúster — K-Means")
plt.tight_layout()
plt.show()"""

for c in nb['cells']:
    if c.get('cell_type') == 'code':
        source = "".join(c.get('source', []))
        if "from collections import Counter" in source:
            c['source'] = [line + '\n' for line in new_source.split('\n')]
            c['source'][-1] = c['source'][-1].rstrip('\n') # remove trailing newline on last line just in case, though Jupyter handles it
            break

with open('diabetes_analisis_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Celda reemplazada exitosamente.")
