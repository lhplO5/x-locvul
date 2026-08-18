import zipfile
import xml.etree.ElementTree as ET
import json
import os

def read_docx(path):
    try:
        with zipfile.ZipFile(path) as z:
            tree = ET.XML(z.read('word/document.xml'))
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p in tree.findall('.//w:p', ns):
                text = ''.join(n.text for n in p.findall('.//w:t', ns) if n.text)
                if text:
                    paragraphs.append(text)
            return '\n'.join(paragraphs)
    except Exception as e:
        return str(e)

def read_ipynb(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            cells_text = []
            for c in nb.get('cells', []):
                ctype = c.get('cell_type', '')
                source = ''.join(c.get('source', []))
                cells_text.append(f"[{ctype.upper()}]\n{source}")
            return '\n\n'.join(cells_text)
    except Exception as e:
        return str(e)

docs_dir = 'd:/x-locvul/docs'
notebooks_dir = 'd:/x-locvul/kaggle_notebooks'

with open('d:/x-locvul/output.txt', 'w', encoding='utf-8') as out_f:
    out_f.write("="*50 + "\n")
    out_f.write("DOCS\n")
    out_f.write("="*50 + "\n")
    for f in os.listdir(docs_dir):
        if f.endswith('.docx'):
            out_f.write(f"\n--- {f} ---\n")
            content = read_docx(os.path.join(docs_dir, f))
            out_f.write(content + "\n")

    out_f.write("\n" + "="*50 + "\n")
    out_f.write("NOTEBOOKS\n")
    out_f.write("="*50 + "\n")
    for f in os.listdir(notebooks_dir):
        if f.endswith('.ipynb'):
            out_f.write(f"\n--- {f} ---\n")
            content = read_ipynb(os.path.join(notebooks_dir, f))
            out_f.write(content + "\n")
