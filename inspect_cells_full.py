import json
with open('diabetes_analisis_completo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i in range(16, 20):
    print(f"--- CELL {i} ---")
    print("".join(nb['cells'][i].get('source', [])))
