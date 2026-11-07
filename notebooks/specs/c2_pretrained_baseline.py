"""Spec: C2 run 1, pretrained Bangla VITS voices: fetch, inspect and baseline audio."""

from nbkit import Notebook, code

NOTEBOOK = Notebook(
    chunk="C2",
    run=1,
    slug="pretrained-baseline",
    title="C2 — pretrained model and baseline",
    branch="chunk/C2-pretrained-baseline",
    gpu=True,
    cells=[
        code("C2.1", "Fetch pretrained voices", "!boinga fetch-pretrained --voice both",
             expect=[r"PASS: female cached", r"PASS: male cached"]),
        code("C2.2", "Inspect pretrained voices", "!boinga inspect-pretrained --voice both",
             expect=[r"PASS: inspected female", r"PASS: inspected male",
                     r"=== BEGIN configs/pretrained_facts.json ==="]),
    ],
)
