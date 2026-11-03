"""Analiz betiklerinin ürettiği notebook'u diske yazıp çalıştıran ortak yardımcı."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

from src.config import PROJECT_ROOT


def write_and_execute_notebook(notebook: nbf.NotebookNode, path: Path) -> None:
    """Notebook'u yazar ve proje kökünden çalıştırarak çıktılarını yerinde doldurur."""
    with path.open("w", encoding="utf-8") as file:
        nbf.write(notebook, file)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(path)],
        check=True,
        cwd=PROJECT_ROOT,
    )


__all__ = ["write_and_execute_notebook"]
