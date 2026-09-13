"""Executa as Partes 0, 1 e 2 do notebook e grava o checkpoint final.

Uso: python train.py
Requer as 670 amostras extraídas em data/BBBC038/ e um kernel Python 3.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


SOURCE = Path("task1.ipynb")
OUTPUT = Path("task1_treinado.ipynb")
LAST_CELL_ID = "part2-correction-test"

notebook = nbformat.read(SOURCE, as_version=4)
last_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.get("id") == LAST_CELL_ID
)
notebook.cells = notebook.cells[:last_index + 1]

print("Treinando até a Parte 2; saída:", OUTPUT.resolve())
client = NotebookClient(notebook, timeout=None, kernel_name="python3")
client.execute()
nbformat.write(notebook, OUTPUT)
print("Treino concluído; checkpoint em checkpoints/small_unet_boundary_watershed_v2.pt")
