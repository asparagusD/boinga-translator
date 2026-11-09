"""Spec: C10 run 4, blind A/B evaluation: optional MCD of A and B vs the real recordings."""

# Run 3 scored the listening test into reports/eval.md (committed). This run reads the C10.2
# audio in Drive eval/final/ (A_baseline, B_boinga, R_real, index.csv) and appends an
# "Objective check (relative only)" section to that eval.md. No GPU, no corpus, no key.csv.

from nbkit import Notebook, code, md

NOTEBOOK = Notebook(
    chunk="C10",
    run=4,
    slug="eval",
    title="objective check",
    branch="chunk/C10-eval",
    gpu=False,
    cells=[
        md("""
            **MCD vs the real recordings.** For each of the 20 prompts, mel-cepstral
            distortion (MCD, in dB) of the baseline voice A and of the fine-tuned voice B
            against the corpus's real recording R of the same sentence: librosa MFCC 1-13,
            leading/trailing silence trimmed, the two clips aligned in time with DTW. Lower is
            closer to the real recording. The real speaker is not the voice of A or B, so MCD
            mixes speaker and dialect: it is only a relative A vs B cross-check of the
            listening test, which outranks it. It reads Drive `eval/final/` (no blind key) and
            replaces or appends the section "Objective check (relative only)" in
            `reports/eval.md`. A tqdm bar counts the prompts (about a minute on CPU). Expect
            `PASS: mcd table (20 prompts)` and eval.md between BEGIN/END markers.
        """),
        code("C10.5", "MCD vs real recordings", "!boinga eval-mcd",
             expect=[r"PASS: mcd table", r"=== BEGIN reports/eval\.md ==="]),
    ],
)
