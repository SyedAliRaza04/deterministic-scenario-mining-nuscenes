"""Build a Kaggle notebook from plain .py/.md sources. Utility, not a step.

WHY THIS EXISTS. A notebook is JSON with the source split into one-line strings, so editing
one by hand means editing escaped JSON — and an editing mistake there is invisible until the
GPU session opens the file. Three notebooks now share the same metadata block (accelerator,
internet, kernel), and R4 says the thing written three times should be written once.

Usage:  ./venv/bin/python scripts/make_notebook.py <spec.py>

A spec module defines CELLS: a list of (kind, text) where kind is 'md' or 'code'.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

META = {
    "accelerator": "GPU",
    "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
               "isInternetEnabled": True, "language": "python",
               "sourceType": "notebook"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


def build(cells: list[tuple[str, str]], out_path: Path | str) -> str:
    """Write the notebook. Source is stored line-by-line, as Jupyter writes it."""
    nb = {"cells": [], "metadata": META, "nbformat": 4, "nbformat_minor": 0}
    for kind, text in cells:
        lines = text.strip("\n").splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            pass
        else:
            lines = [ln for ln in lines]
        cell = {"cell_type": "markdown" if kind == "md" else "code",
                "metadata": {}, "source": lines}
        if kind != "md":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
    out = Path(out_path)
    out.write_text(json.dumps(nb, indent=1) + "\n")
    # Round-trips through json and through nbformat's own shape: a notebook Kaggle cannot
    # open is discovered here rather than in the session.
    reread = json.loads(out.read_text())
    assert len(reread["cells"]) == len(cells), "cell count changed on the round trip"
    return str(out)


if __name__ == "__main__":
    spec_path = Path(sys.argv[1])
    sys.path.insert(0, str(spec_path.parent))
    spec = __import__(spec_path.stem)
    print(build(spec.CELLS, ROOT / spec.OUT))
