"""Spec: C3 run 2, Chittagonian corpus: index, audit, summary and dialect baseline (T4)."""

from nbkit import Notebook, code, form, md

# Cells call the CLI from Python and stop Run all on a non-zero exit: a failing `!` shell
# line does not stop the run (C3 run 1, DEBUG_LOG).

NOTEBOOK = Notebook(
    chunk="C3",
    run=2,
    slug="data-audit",
    title="Index, audit, summary",
    branch="chunk/C3-data-audit",
    gpu=True,  # T4 for the C3.6 baseline synthesis; the audit cells only use the CPU
    cells=[
        md("""
            **Dataset on this VM.** Makes sure the corpus is on the fast disk: reuses the
            verified Drive copy or downloads it; download, unzip and copy show progress bars.
            Expect `PASS: dataset ready`.
        """),
        code("C3.2.fetch", "Ensure dataset on this VM", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready"]),
        md("""
            **Index the corpus.** One row per spreadsheet line with its wav, dialect text,
            translation and split. Expect `PASS: index has 8009 rows (8004 with audio)`.
        """),
        code("C3.3", "Index corpus", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["index"]) != 0:
                colab_runtime.fail("index failed (see the FAIL line above)")
        """, expect=[r"PASS: index has 8009 rows \(8004 with audio\)"]),
        md("""
            **Audit every clip (10–30 min, resumable).** Per clip: format, duration, levels,
            silence, and characters the voice does not know; saved to Drive `reports/audit.csv`
            in parts, so a re-run continues. A progress bar counts clips. Expect `PASS: audited
            <n> clips`.
        """),
        code("C3.4c", "Audit all clips (10–30 min)", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["audit"]) != 0:
                colab_runtime.fail("audit failed (see the FAIL line above)")
        """, expect=[r"PASS: audited \d+ clips"], timeout_min=45),
        md("""
            **Audit summary.** Applies the exclusion rules and writes `reports/audit_summary.md`
            (clean hours, reasons, charset problems). Expect `PASS: summary written` and the
            report block.
        """),
        code("C3.5", "Audit summary", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["audit-summary"]) != 0:
                colab_runtime.fail("audit-summary failed (see the FAIL line above)")
        """, expect=[r"PASS: summary written", r"=== BEGIN reports/audit_summary.md ==="]),
        md("""
            **Listen.** 20 random clean clips with their transcripts, to judge whether the text
            matches the audio. Your judgement goes in the form under "Your observations".
        """),
        code("C3.5.play", "Listen: 20 random clean clips", """
            from boinga_translator import audit_summary, colab_runtime

            colab_runtime.play(*audit_summary.sample_clips(20))
        """, expect=[r"AUDIO \d+ \["]),
        md("""
            **Your observations — corpus.** Do the transcripts match, does it sound
            Chittagonian, how noisy is it, how many speakers did you hear? Pick from the lists,
            add notes, then run this cell again.
        """),
        form("C3.5-notes", "Your observations — corpus", fields={
            "TRANSCRIPTS_MATCH_AUDIO": ["", "yes", "mostly", "no"],
            "SOUNDS_CHITTAGONIAN": ["", "yes", "mixed", "no, standard Bangla"],
            "NOISE_LEVEL": ["", "clean", "some noise", "noisy"],
            # No speaker field in the corpus; the single-speaker plan depends on this answer.
            "SPEAKERS_HEARD": ["", "one", "a few (2-5)", "many"],
            "NOTES": "",
        }),
        md("""
            **Dialect prompts.** Picks 3 held-out test-split lines (5-12 words, clean) least
            like their own Bangla translation into `configs/baseline_prompts.json`, then both
            pretrained voices read them (a progress bar per voice). Expect `PASS: picked 3
            dialect prompts`, `PASS: baseline female`, `PASS: baseline male` and the prompts
            block.
        """),
        # Gate G1 recorded no preferred voice (quality "same"), so both voices read the lines.
        code("C3.6", "Pick dialect prompts and synthesise them", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["pick-baseline-ctg"]) != 0:
                colab_runtime.fail("pick-baseline-ctg failed (see the FAIL line above)")
            if cli.main(["baseline", "--voice", "both"]) != 0:
                colab_runtime.fail("baseline failed (see the FAIL line above)")
        """, expect=[r"PASS: picked 3 dialect prompts", r"PASS: baseline female",
                     r"PASS: baseline male", r"=== BEGIN configs/baseline_prompts.json ==="],
             timeout_min=30),
        md("""
            **Listen.** The pretrained voices reading the Chittagonian lines (CTG_1-3). Your
            judgement goes in the form under "Your observations".
        """),
        code("C3.6.play", "Listen: pretrained voices on dialect text", """
            from boinga_translator import baseline

            baseline.play_all(prompts=baseline.load_prompts(prefix="CTG_"))
        """, expect=[r"AUDIO female CTG_1", r"AUDIO male CTG_1"]),
        md("""
            **Your observations — baseline on dialect text.** How well do the pretrained voices
            read Chittagonian? Pick from the list, add notes, then run this cell again.
        """),
        form("C3.6-notes", "Your observations — baseline on dialect text", fields={
            "HANDLES_DIALECT": ["", "well", "partly", "badly"],
            "NOTES": "",
        }),
    ],
)
