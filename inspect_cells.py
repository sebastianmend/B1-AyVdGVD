import json

with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    source = "".join(c.get('source', []))
    print(f"--- CELL {i} ({c['cell_type']}) ---")
    print(source[:100].replace('\n', ' '))
