# A simple script to clean up Jupyter notebooks in place.
import json
import glob
import os


def clean_notebook(file_path: str) -> None:
    """
    Clean up a Jupyter notebook file in place.

    Args:
        file_path (str): The path to the Jupyter notebook file.

    Normal format of a Jupyter notebook is as follows:

    {
      "cells": [
        {
          "cell_type": "code",
          "execution_count": null,
          "id": "0",
          "metadata": {},
          "source": [
            "import tvm\n",
          ]
        },
        ...
      ],
      "metadata": {
        "kernelspec": {
          "display_name": "tvm-build-venv",
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
          "version": "3.11.13"
        }
      },
      "nbformat": 4,
      "nbformat_minor": 5
    }

    execution_count, id, metadata, outputs, metadata, nbformat, and nbformat_minor
    are always modified to ensure the notebook is valid to run and it will cause the
    git repo to be modified.

    This script will modify the notebook file in place to remove these modifications.
    """
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
        # if "outputs" in cell:
        #     del cell["outputs"]

    # if "metadata" in notebook:
    #     del notebook["metadata"]

    # if "nbformat" in notebook:
    #     del notebook["nbformat"]

    # if "nbformat_minor" in notebook:
    #     del notebook["nbformat_minor"]
    
    if "metadata" in notebook:
      if "kernelspec" in notebook["metadata"]:
        if "display_name" in notebook["metadata"]["kernelspec"]:
          notebook["metadata"]["kernelspec"]["display_name"] = ""
      if "language_info" in notebook["metadata"]:
        if "version" in notebook["metadata"]["language_info"]:
          notebook["metadata"]["language_info"]["version"] = ""

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
