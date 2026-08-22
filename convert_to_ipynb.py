import os
import json

python_files = {
    "src/models/e1_0_baseline.py": "kaggle_notebooks/x-locvul-e1-0.ipynb",
    "src/models/e1_1_codebert.py": "kaggle_notebooks/x-locvul-e1-1.ipynb",
    "src/models/e1_2_unixcoder.py": "kaggle_notebooks/x-locvul-e1-2.ipynb",
    "src/models/e1_3_unix_multi.py": "kaggle_notebooks/x-locvul-e1-3.ipynb",
    "src/models/e1_4_hard_mining.py": "kaggle_notebooks/x-locvul-e1-4.ipynb",
    "src/models/e1_5_shuffled_cwe.py": "kaggle_notebooks/x-locvul-e1-5.ipynb",
    "src/models/e1_6_hierarchical_cwe.py": "kaggle_notebooks/x-locvul-e1-6.ipynb",
    "src/models/evaluate_e1.py": "kaggle_notebooks/x-locvul-evaluate.ipynb"
}

os.makedirs("kaggle_notebooks", exist_ok=True)

# Path to inject for Kaggle
KAGGLE_FROZEN_DIR = '"/kaggle/input/notebooks/linhlhp/x-locvul-e0/frozen_data"'
KAGGLE_OUTPUT_DIR_PREFIX = '"/kaggle/working/saved_models_e1_'

def create_notebook(py_path, ipynb_path, output_num):
    with open(py_path, 'r', encoding='utf-8') as f:
        code_lines = f.readlines()
        
    # Modify paths for Kaggle
    for i, line in enumerate(code_lines):
        if line.startswith('DATA_DIR = "./data/manifests"') or line.startswith('FROZEN_DIR = "./data/manifests"'):
            code_lines[i] = f'FROZEN_DIR = {KAGGLE_FROZEN_DIR}\n'
            # Also replace DATA_DIR with FROZEN_DIR everywhere
        elif 'DATA_DIR' in line:
            code_lines[i] = line.replace('DATA_DIR', 'FROZEN_DIR')
        elif line.startswith('OUTPUT_DIR = "./saved_models_'):
            code_lines[i] = f'OUTPUT_DIR = {KAGGLE_OUTPUT_DIR_PREFIX}{output_num}"\n'
        # Fix paths inside evaluate_e1.py
        elif line.strip().startswith('"checkpoint": "./saved_models_e1_'):
            code_lines[i] = line.replace('"checkpoint": "./saved_models_e1_', f'"checkpoint": "/kaggle/working/saved_models_e1_')
    
    # Prepend pip installs for NN models as the first cell
    install_cell = []
    if output_num in ["1", "2", "3", "4", "5", "6"]:
        install_cell = ["!pip install transformers datasets scikit-learn -q\n"]
    elif output_num == "eval":
        install_cell = ["!pip install transformers datasets scikit-learn statsmodels -q\n"]
    else:
        install_cell = ["!pip install scikit-learn -q\n"]

    cells = []
    if install_cell:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": install_cell
        })

    # Split code_lines into multiple cells based on print("X. ") statements
    import re
    current_cell = []
    for line in code_lines:
        if re.match(r'^print\("\d+\.\s', line) or line.startswith('def run_stage_1_and_2'):
            if current_cell:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": current_cell
                })
                current_cell = []
        current_cell.append(line)
        
    if current_cell:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": current_cell
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

for py_path, ipynb_path in python_files.items():
    # Extract the number from the filename to pass as output_num
    # e.g., src/models/e1_0_baseline.py -> output_num = "0"
    base = os.path.basename(py_path)
    if "evaluate" in base:
        num = "eval"
    else:
        num = base.split('_')[1]
    create_notebook(py_path, ipynb_path, num)

print("Notebooks created successfully!")
