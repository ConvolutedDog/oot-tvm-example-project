# cleanipynb.py
import json
import glob
import sys


def clean_notebook(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    for cell in notebook.get("cells", []):
        if "execution_count" in cell:
            del cell["execution_count"]
        if "id" in cell:
            del cell["id"]
        if "metadata" in cell:
            del cell["metadata"]
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
    for ipynb in glob.glob("../tests/python/tvm/**/*-test.ipynb", recursive=True):
        print(f"Cleaning: {ipynb}")
        clean_notebook(ipynb)
