"""Spec: C1 bootstrap-only notebook on main (mount Drive, clone, install deps, record env)."""

from nbkit import Notebook

NOTEBOOK = Notebook(
    chunk="C1",
    run=1,
    slug="bootstrap",
    title="Colab bootstrap",
    branch="main",
    cells=[],
)
