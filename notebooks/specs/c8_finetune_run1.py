"""Spec: C8 run 1, fine-tune run1 on the VM disk, sample, rank, keep on Drive and listen."""

from nbkit import Notebook, code, form, md

NOTEBOOK = Notebook(
    chunk="C8",
    run=1,
    slug="finetune",
    title="C8 run 1 — fine-tune",
    branch="chunk/C8-finetune",
    gpu=True,
    cells=[
        md("""
            **Prepared data on this VM.** Training reads wavs and manifests from the fast disk.
            `fetch-data` makes sure the corpus is here; `prepare` restores the prepared dataset
            from its Drive tar (progress bars) instead of recomputing it. Expect `PASS: dataset
            ready` and `PASS: prepared <n> clips`. Checkpoints are written to this VM's disk,
            not Drive; only the steps you listen to are copied to Drive at the end (under 1 GB
            each, a few GB in all), so Drive needs that much free space and an empty trash.
        """),
        code("C8.pre", "Restore prepared data", "!boinga fetch-data\n!boinga prepare",
             expect=[r"PASS: dataset ready", r"PASS: prepared \d+ clips"]),
        md("""
            **What healthy training looks like.** In coqui's step logs (every 25 steps) and in
            TensorBoard, `loss_mel` and `loss_1` (generator total) fall quickly over the first
            few thousand steps, then flatten; the discriminator loss `loss_0` stays roughly
            level. Bad signs: any `nan`/`inf` loss, or `loss_mel` rising steadily over several
            epochs. A validation pass runs at the end of every epoch (~401 steps) and prints
            `EVAL PERFORMANCE`.

            **Keep this VM alive until the end.** The run lives on this VM's disk (Drive's trash
            filled up with deleted checkpoints in the first run 1 and lost them). A disconnect
            or runtime restart before the "Keep" cells loses the run: "Run all" then starts
            again from the pretrained voice. Keep the tab open.
        """),
        md("""
            **TensorBoard (optional).** Opens coqui's loss curves and the per-epoch test-sentence
            audio for run1's folder on the VM disk. Nothing is checked; skip it if it is slow to
            load.
        """),
        code("C8.tb", "TensorBoard (optional)", """
            from boinga_translator import paths

            %load_ext tensorboard
            %tensorboard --logdir {paths.run_dir('run1')}
        """, expect=[]),
        md("""
            **Train run1 to its budget (hours).** Starts run1 from the pretrained male
            checkpoint and stops at the step budget in `train_config.RUN_BUDGETS` (36,500
            steps, ~91 epochs at batch 16). The first run 1 took ~2.4 h on the RTX PRO 6000
            (1.5 h of step time, the rest eval and saves). It needs 20 GB free on the VM disk
            (`space: ... GB free`). coqui's step logs are the progress output. Expect
            `budget: 36500 steps (RUN_BUDGETS)` and `STOP at step 36500`; a re-run in the same
            VM prints `already at step 36500` and trains nothing.
        """),
        code("C8.train", "Train run1 to its budget (hours)",
             "!boinga train --run run1",
             expect=[r"STOP at step \d+|PASS: run run1 already at step \d+"],
             forbid=[r"Partial model initialization"], gpu=True, timeout_min=360),
        md("""
            **Throughput of run1.** Reads run1's own coqui log and run record: seconds per step,
            steps per hour, peak GPU memory and any NaN loss into `reports/throughput.json`.
            Expect `PASS: throughput recorded`; a NaN loss prints `FAIL: NaN loss`.
        """),
        code("C8.thr", "Throughput of run1", "!boinga throughput --run run1",
             expect=[r"PASS: throughput recorded"]),
        md("""
            **Sample every checkpoint.** For each checkpoint on the VM disk (the last 2 and
            every best model) and once for the pretrained voice, synthesises the 5 test
            sentences (val lines) with the fixed seed and noise scales into Drive
            `eval/run1/step_<N>/`. Audio from the first run 1 is replaced, never reused. Expect
            one `step N: mean X s` line per step and `PASS: sampled K checkpoints`.
        """),
        code("C8.sample", "Sample every checkpoint", "!boinga sample --run run1",
             expect=[r"PASS: sampled \d+ checkpoints"], gpu=True),
        md("""
            **Rank checkpoints.** Matches each checkpoint to the nearest validation pass in the
            training log and sorts by validation `loss_mel` (`loss_total` = the sum coqui picks
            its best model by). Writes `reports/candidates_run1.md`. Expect the report block and
            `PASS: candidates listed`.
        """),
        code("C8.cand", "Rank checkpoints", "!boinga candidates --run run1",
             expect=[r"PASS: candidates listed", r"=== BEGIN reports/candidates_run1.md ==="]),
        md("""
            **Keep the listened steps on Drive.** Copies the steps the Listen cell plays (the 4
            best by eval loss_mel and the last) from the VM disk to Drive
            `milestones/run1/step_<N>/`: model weights without the optimizer, plus config.json,
            each copy compared with its checkpoint tensor by tensor. It refuses to start if
            Drive lacks the space. A tqdm bar counts the checkpoints (a few minutes). Expect
            `step N: kept X MB` lines and `PASS: kept K milestones`.
        """),
        code("C8.keep", "Keep the listened steps on Drive", "!boinga keep --run run1",
             expect=[r"PASS: kept \d+ milestones \(\d+ already kept\)"],
             forbid=[r"FAIL: keep"], timeout_min=60),
        md("""
            **Check they really reached Drive.** The Drive mount uploads in the background and
            hides failed uploads, so this waits for every pending upload (up to 20 minutes, no
            progress bar: the flush gives none), unmounts and mounts Drive again, then checks
            each kept `model.pth` has its recorded size. Stay at the computer for this cell:
            the remount opens a Google sign-in popup that "Run all" waits on. Expect `PASS: K
            milestones on Drive`. If it fails, do not close the VM: free Drive space, empty its
            trash, then run "Keep" and this cell again.
        """),
        code("C8.flush", "Check they really reached Drive", """
            from google.colab import drive

            from boinga_translator import cli, colab_runtime, paths

            try:
                drive.flush_and_unmount(timeout_ms=20 * 60 * 1000)
            except Exception as exc:  # stuck uploads: never wait the default 24 h
                colab_runtime.fail(f"Drive flush did not finish: {exc!r}; free Drive space")
            drive.mount(str(paths.COLAB_DRIVE.parent))
            if cli.main(["check-milestones", "--run", "run1"]) != 0:
                colab_runtime.fail("milestones are not on Drive (see the FAIL line above)")
        """, expect=[r"PASS: \d+ milestones on Drive"], timeout_min=30),
        md("""
            **Listen: pretrained vs best candidates vs last.** The 5 sentences by the pretrained
            male voice, then by the 4 best checkpoints by `loss_mel` and the last one; each
            label starts with its step. These are exactly the steps kept on Drive. Listen for
            clarity, Chittagonian pronunciation and artefacts (buzz, metallic noise, skipped or
            repeated words). Nothing is checked automatically: write what you hear below.
        """),
        code("C8.play", "Listen: pretrained vs best candidates vs last", """
            from boinga_translator import ckpt_sampler, colab_runtime

            colab_runtime.play(*zip(*ckpt_sampler.players("run1"), strict=True))
        """, expect=[]),
        md("""
            **Your observations — run 1.** `BEST_STEP`: the step you would keep (a number from
            the labels above). `VERDICT`: promote that step, train longer, or retry with a lower
            learning rate. Compare against the pretrained voice for clarity and Chittagonian
            pronunciation; list any artefacts with their step. Every field is required: write
            `none` in ARTEFACTS or NOTES if there is nothing to say.
        """),
        form("C8-notes", "Your observations — run 1", fields={
            "BEST_STEP": "",
            "VERDICT": ["", "good: promote BEST_STEP", "undertrained: train longer",
                        "degraded: retry with lower LR"],
            "CLEARER_THAN_PRETRAINED": ["", "yes", "same", "no"],
            "MORE_CHITTAGONIAN_THAN_PRETRAINED": ["", "yes", "same", "no"],
            "ARTEFACTS": "",
            "NOTES": "",
        }),
    ],
)
