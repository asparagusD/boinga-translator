"""Spec: C8.2c, keep model + config copies of run1's listened checkpoints on Drive (no training)."""

from nbkit import Notebook, code, form, md

NOTEBOOK = Notebook(
    chunk="C8",
    run=1,
    slug="keep",
    title="C8 — keep run1 milestones",
    branch="chunk/C8-finetune",
    gpu=False,
    cells=[
        md("""
            **No training here.** This notebook only copies files that run 1 already wrote to
            Drive, so a CPU runtime is enough (no GPU needed). Coqui keeps only the last 2
            `checkpoint_<N>.pth` of a run; any later training of run1 would delete step 36500.
        """),
        md("""
            **Rank run1's checkpoints.** Re-reads run1's training log and checkpoint list on
            Drive (seconds) and writes `reports/candidates_run1.md` again; it should match the
            committed one. The keep step below copies the steps this ranking puts first.
            Expect the report block and `PASS: candidates listed (17 checkpoints)`.
        """),
        code("C8K.1", "Rank run1's checkpoints", "!boinga candidates --run run1",
             expect=[r"PASS: candidates listed \(17 checkpoints\)"]),
        md("""
            **Keep the listened steps.** For each step the listening form played (the 4 best
            by eval loss_mel and the last: 20451, 36500, 36000, 14436), loads the ~1 GB
            checkpoint on the CPU, saves it without the optimizer and scaler state (much
            smaller) and the run's config.json to Drive
            `milestones/run1/step_<N>/`, reloads the copy and compares every tensor with the
            original. A tqdm bar counts the checkpoints (a few minutes, mostly Drive I/O).
            Expect four `step N: kept X MB` lines and `PASS: kept 4 milestones (0 already
            kept)`; a re-run prints `PASS: kept 0 milestones (4 already kept)`.
        """),
        code("C8K.2", "Keep the listened steps", "!boinga keep --run run1",
             expect=[r"PASS: kept \d+ milestones \(\d+ already kept\)"],
             forbid=[r"FAIL: keep"], timeout_min=60),
        md("""
            **What is on Drive now.** Lists `milestones/run1/` with each step's files and
            sizes, so you can see the copies next to the free space printed above. Expect
            `step_14436`, `step_20451`, `step_36000` and `step_36500`, each with
            `config.json`, `keep.json` and `model.pth`.
        """),
        code("C8K.3", "What is on Drive now", """
            from boinga_translator import paths

            for d in sorted((paths.milestones() / "run1").iterdir()):
                files = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(d.iterdir())}
                print(d.name, files)
        """, expect=[r"step_36500 \{'config\.json'"]),
        md("""
            **The copy loads for synthesis.** Loads the step-36500 copy (not the run folder's
            checkpoint) with coqui's Synthesizer on the CPU, the way C9 will use it. Takes under
            a minute. Expect the `synth:` line and `PASS: milestone 36500 loads`.
        """),
        code("C8K.4", "The copy loads for synthesis", """
            from boinga_translator import milestones, pretrained, synth_basic

            d = milestones.milestone_dir("run1", 36500)
            synth_basic.load_synth(d / pretrained.MODEL_FILE, d / pretrained.CONFIG_FILE,
                                   use_cuda=False)
            print("PASS: milestone 36500 loads")
        """, expect=[r"PASS: milestone 36500 loads"]),
        md("""
            **Your check.** Nothing to listen to: just confirm the four folders appeared and
            how much Drive space is left (from the PASS line above). Write `none` in NOTES if
            there is nothing to say.
        """),
        form("C8K-notes", "Your check — milestones", fields={
            "FOUR_FOLDERS_PRESENT": ["", "yes", "no"],
            "NOTES": "",
        }),
    ],
)
