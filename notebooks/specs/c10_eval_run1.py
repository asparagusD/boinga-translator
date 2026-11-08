"""Spec: C10 run 1, blind A/B evaluation: pick 20 held-out prompts (audio and sheet follow)."""

from nbkit import Notebook, code, form, md

NOTEBOOK = Notebook(
    chunk="C10",
    run=1,
    slug="eval",
    title="C10 run 1 — prompts, audio, rating sheet",
    branch="chunk/C10-eval",
    gpu=True,
    cells=[
        md("""
            **Restore data on this VM.** The prompt pick reads the corpus workbook (Chittagonian
            text and its Bangla translation) from this VM's fast disk and `splits.csv` from
            Drive. `fetch-data` makes sure the corpus is here; `prepare` restores the prepared
            dataset from its Drive tar (progress bars) instead of recomputing it, and rewrites
            nothing about the fixed test split. Expect `PASS: dataset ready` and
            `PASS: prepared <n> clips`.
        """),
        code("C10.pre", "Restore data", "!boinga fetch-data\n!boinga prepare",
             expect=[r"PASS: dataset ready", r"PASS: prepared \d+ clips"]),
        md("""
            **Pick the 20 evaluation prompts.** P01-P15 are clean clips of our training
            speaker's test split (never trained on; their real recording is the reference R).
            P16-P20 are clean official-test clips of other speakers (one per voice cluster),
            never in our splits. Lines of 5-15 words come first, then the ones least like their
            Standard Bangla translation (the most dialectal). The text is kept verbatim. If
            `configs/eval_prompts.json` already exists in this checkout, it is kept as it is.
            Expect one line per prompt, `PASS: 20 prompts` and the file printed between
            BEGIN/END markers for the repo.
        """),
        code("C10.1", "Pick evaluation prompts", "!boinga pick-eval-prompts",
             expect=[r"PASS: 20 prompts", r"=== BEGIN configs/eval_prompts\.json ==="]),
        md("""
            **Show the prompts for review.** Prints each prompt id with its Chittagonian line
            (`ctg`) and the corpus's Bangla translation (`bn`), so you can read them with a
            Chittagonian speaker before filling in the form below. Nothing is checked
            automatically.
        """),
        code("C10.1.show", "Show prompts", """
            from boinga_translator import evaluate

            evaluate.show_prompts()
        """, expect=[]),
        md("""
            **Your observations — prompt review.** Read the 20 prompts above with a
            Chittagonian speaker. `REVIEWED_BY`: who reviewed them (a first name or initials).
            `IDS_TO_REPLACE`: comma-separated ids of prompts that are not natural Chittagonian,
            are offensive, or are hard to read aloud (e.g. `P03,P11`), or `none`. `NOTES`:
            anything else, or `none`. All three fields are required. Ids listed here are
            replaced in a later run with `boinga pick-eval-prompts --exclude ...`; the other ids
            keep their text.
        """),
        form("C10.1-notes", "Prompt review (with a Chittagonian speaker)", fields={
            "REVIEWED_BY": "",
            "IDS_TO_REPLACE": "",
            "NOTES": "",
        }),
    ],
)
