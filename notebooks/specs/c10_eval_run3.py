"""Spec: C10 run 3, blind A/B evaluation: rate the blind clips here, then score into eval.md."""

# Runs 1 (prompts) and 2 (audio + blind sheet) are verified. The user rates alone (no rater
# panel), so this run has an interactive rating screen; it reads Drive eval/final/blind/,
# writes eval/final/ratings/me.csv, then unblinds with eval/final/key.csv. No GPU, no corpus.

from nbkit import Notebook, code, md

NOTEBOOK = Notebook(
    chunk="C10",
    run=3,
    slug="eval",
    title="rating and scoring",
    branch="chunk/C10-eval",
    gpu=False,
    cells=[
        md("""
            **Rate the blind clips.** A rating screen for the 20 prompts, one at a time: the
            sentence, then `clip1` and `clip2` (the two voices in a random order; nothing here
            says which is which), each with four 1-5 scores and an optional comment.
            `intelligibility`: how easily you understand the words. `naturalness`: how human
            and fluent it sounds. `dialect_authenticity`: how much it sounds like a native
            Chittagonian speaker. `dialect_words`: are the Chittagonian-specific words said
            correctly. (Full scales: `eval/final/blind/INSTRUCTIONS.md`.) **Save & next**
            saves to Drive `eval/final/ratings/me.csv` after every prompt; a prompt with a
            missing score is refused. You can stop at any time: running this cell again
            resumes at the first unrated prompt, and **Previous** lets you change a saved
            prompt. Nothing is checked automatically in this cell.
        """),
        code("C10.R", "Rate the blind clips", """
            from boinga_translator import rating_app

            rating_app.rate("me")
        """, expect=[]),
        md("""
            **Check the ratings are complete.** "Run all" reaches this cell before you have
            rated anything, so the first time it stops here with `FAIL: rated 0 of 40 clips`:
            that is expected. Rate all 20 prompts above, then click this cell and use
            Runtime > Run cell and below. Expect `PASS: rated 40 of 40 clips`. If you typed
            your scores straight into Drive `eval/final/blind/rating_sheet.csv` instead of
            using the screen, this cell copies them to `eval/final/ratings/me.csv` first
            (`copied 40 rated clips from blind/rating_sheet.csv ...`); a saved `me.csv` with
            fewer rated clips is kept as `me.csv.<time>.bak`.
        """),
        code("C10.R.check", "Check the ratings are complete", """
            from boinga_translator import rating_app

            rating_app.check("me")
        """, expect=[r"PASS: rated 40 of 40 clips"]),
        md("""
            **Score the ratings.** Reads every sheet in Drive `eval/final/ratings/` (yours
            from the screen above, plus any other rater's `<rater>.csv` you put there). It
            checks every sheet has all 40 rows of the key and that every score is a whole
            number 1-5; any bad cell is listed with its file, spreadsheet row and column, and
            nothing is scored (no value is ever guessed). Then it unblinds with
            `eval/final/key.csv` and writes `reports/eval.md`: mean ± sd per criterion for A
            (baseline) vs B (fine-tuned), per-prompt B win rate, every comment by voice, and
            the limitations. Raters appear as R1..Rn. Expect `PASS: scored <n> raters` and the
            report between BEGIN/END markers.
        """),
        code("C10.4", "Score ratings", "!boinga eval-score",
             expect=[r"PASS: scored \d+ raters", r"=== BEGIN reports/eval\.md ==="]),
    ],
)
