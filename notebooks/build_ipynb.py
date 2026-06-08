"""Convert source.py (`# %%` / `# %% [markdown]` cell-marked) into a Jupyter notebook.

Workflow:
    1. Edit notebooks/source.py - it's a runnable script with `# %%` cell markers.
    2. Run `python notebooks/build_ipynb.py` to regenerate the .ipynb.

Cell rules:
    `# %% [markdown]` starts a markdown cell. Body lines lose the leading `# `.
    `# %%`            starts a code cell. `plt.savefig(...) / plt.close()` pairs
                      are rewritten to `plt.show()` so the notebook renders inline
                      while the source script still runs headless.
    Lines before the first marker (e.g. the module docstring) are skipped.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "source.py"
OUT = HERE / "basis_refinement_tutorial.ipynb"


def parse_cells(text: str) -> list[dict]:
    # Drop a leading module docstring, if present.
    m = re.match(r'^"""[\s\S]*?"""\s*\n', text)
    if m:
        text = text[m.end():]

    cells: list[dict] = []
    kind: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            md_lines = []
            for line in body.split("\n"):
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": md_lines,
            })
        else:
            body_nb = re.sub(
                r"plt\.savefig\([^\)]*\)\s*\n\s*plt\.close\(\)",
                "plt.show()",
                body,
            )
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": body_nb.split("\n"),
            })

    for line in text.split("\n"):
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            if kind is not None:
                buf.append(line)
    flush()
    return cells


def finalize(cells: list[dict]) -> None:
    # nbformat wants source as list of strings, each ending with "\n" except the last.
    for c in cells:
        lines = c["source"]
        c["source"] = [l + "\n" if i < len(lines) - 1 else l
                       for i, l in enumerate(lines)]


def main() -> None:
    text = SRC.read_text()
    cells = parse_cells(text)
    finalize(cells)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (proxyasset)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"Wrote {OUT.relative_to(HERE.parent)} - {len(cells)} cells ({n_md} md, {n_code} code).")


if __name__ == "__main__":
    main()
