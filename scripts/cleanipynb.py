# cleanipynb.py
import json
import glob
import sys
import os


def clean_notebook(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cellid = 0
    for cell in notebook.get("cells", []):
        if "execution_count" in cell:
            cell["execution_count"] = None
        if "id" in cell:
            cell["id"] = str(cellid)
            cellid += 1
        if "metadata" in cell:
            cell["metadata"] = {}
        if "outputs" in cell:
            del cell["outputs"]

    if "metadata" in notebook:
        del notebook["metadata"]

    if "nbformat" in notebook:
        del notebook["nbformat"]

    if "nbformat_minor" in notebook:
        del notebook["nbformat_minor"]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ipynb_files = glob.glob(
        os.path.join(project_root, "tests/python/**/*.ipynb"), recursive=True
    )

    for ipynb in ipynb_files:
        print(f"Cleaning: {ipynb}")
        clean_notebook(ipynb)
