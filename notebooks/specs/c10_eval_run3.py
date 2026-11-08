"""Spec: C10 run 3, blind A/B evaluation: unblind the filled rating sheets into reports/eval.md."""

# Runs 1 (prompts) and 2 (audio + blind sheet) are verified; this run only reads Drive
# eval/final/key.csv and eval/final/ratings/*.csv, so it needs no GPU and no corpus.

from nbkit import Notebook, code, md

NOTEBOOK = Notebook(
    chunk="C10",
    run=3,
    slug="eval",
    title="scoring",
    branch="chunk/C10-eval",
    gpu=False,
    cells=[
        md("""
            **Score the ratings.** Before running: put each rater's filled sheet in Drive
            `eval/final/ratings/` as `<rater>.csv` (from Google Sheets: File > Download >
            Comma-separated values). This cell checks every sheet has all 40 rows of the key
            and that every score is a whole number 1-5. Any bad cell is listed with its file,
            spreadsheet row and column, and nothing is scored: fix the sheet and run again
            (no value is ever guessed). Then it unblinds with `eval/final/key.csv` and writes
            `reports/eval.md`: mean ± sd per criterion for A (baseline) vs B (fine-tuned),
            per-prompt B win rate, every comment by voice, and the limitations. Raters appear
            as R1..Rn. Expect `PASS: scored <n> raters` and the report between BEGIN/END
            markers.
        """),
        code("C10.4", "Score ratings", "!boinga eval-score",
             expect=[r"PASS: scored \d+ raters", r"=== BEGIN reports/eval\.md ==="]),
    ],
)
