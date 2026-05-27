import json

def dump(nb_path, out_path):
    with open(nb_path, encoding='utf-8') as fin:
        nb = json.load(fin)
    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, cell in enumerate(nb['cells']):
            fout.write(f'--- CELL {i} ({cell["cell_type"]}) ---\n')
            fout.write(''.join(cell['source']) + '\n\n')

dump('analisis_diabetes.ipynb', 'nb1_dump.txt')
dump('samue/diabetes_clustering_fase2.ipynb', 'nb2_dump.txt')
