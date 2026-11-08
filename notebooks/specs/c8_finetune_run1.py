"""Spec: C8 run 1, fine-tune run1 with checkpoints on Drive, check the uploads, sample, listen."""

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
            ready` and `PASS: prepared <n> clips`.
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

            **Checkpoints on Drive.** A ~1 GB checkpoint is written to Drive every 10,000
            steps and at the end (10k, 20k, 30k, 36.5k), and none is ever deleted or replaced:
            the first run 1 lost its checkpoints when coqui's deletes filled Drive's trash.
            No best-model files are saved. **If Colab disconnects:** reconnect and "Run all";
            training resumes from the newest checkpoint (`RESUME from step N`), repeating at
            most 10,000 steps (~25 min).
        """),
        md("""
            **TensorBoard (optional).** Opens coqui's loss curves and the per-epoch test-sentence
            audio for run1's Drive folder. Nothing is checked; skip it if it is slow to load.
        """),
        code("C8.tb", "TensorBoard (optional)", """
            from boinga_translator import paths

            %load_ext tensorboard
            %tensorboard --logdir {paths.run_dir('run1')}
        """, expect=[]),
        md("""
            **Train run1 to its budget (hours; resumable).** Starts run1 from the pretrained
            male checkpoint (or resumes it) and stops at the step budget in
            `train_config.RUN_BUDGETS` (36,500 steps, ~91 epochs at batch 16). The first run 1
            took ~2.4 h on the RTX PRO 6000. Before training it checks Drive has room for this
            session's checkpoints (`space: ... GB free (need ... GB)`; ~6 GB for a fresh run)
            and turns best-model saves off. coqui's step logs are the progress output. Expect
            `budget: 36500 steps (RUN_BUDGETS)` and `STOP at step 36500`; a re-run after the
            end prints `already at step 36500` and trains nothing.
        """),
        code("C8.train", "Train run1 to its budget (hours; resumable)",
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
            **Record the checkpoints.** Reads each checkpoint and the config in run1's folder
            and records its size and md5 on this VM's disk (not Drive), as written by
            training. A tqdm bar counts the files (~1 GB each, a minute or two). Expect one
            line per file and `PASS: recorded 5 files of run run1` (4 checkpoints + config).
        """),
        code("C8.rec", "Record the checkpoints", "!boinga record-run --run run1",
             expect=[r"PASS: recorded \d+ files of run run1"]),
        md("""
            **Check they really reached Drive.** The Drive mount uploads in the background and
            hides failed uploads, so this waits for every pending upload (up to 20 minutes, no
            progress bar: the flush gives none), unmounts and mounts Drive again, then lists
            run1's folder through the Drive API (what Drive really holds, not the mount's
            local copies) and compares every recorded file's size and md5. Stay at the computer
            for this cell: the remount and the API each open a Google sign-in popup that
            "Run all" waits on; choose the account whose Drive is mounted (the cell prints the
            API account). Expect `PASS: 5 files of run run1 on Drive (Drive API)`. This only
            detects a loss: if a checkpoint is missing or differs, it most likely did not
            upload and cannot be recovered from this VM. Free Drive space, empty its trash,
            and tell Claude which files failed before training again.
        """),
        code("C8.flush", "Check they really reached Drive", """
            from google.colab import drive

            from boinga_translator import cli, colab_runtime, paths

            try:
                drive.flush_and_unmount(timeout_ms=20 * 60 * 1000)
            except Exception as exc:  # stuck uploads: never wait the default 24 h
                colab_runtime.fail(f"Drive flush did not finish: {exc!r}; free Drive space")
            drive.mount(str(paths.COLAB_DRIVE.parent))
            if cli.main(["check-run", "--run", "run1", "--drive-api"]) != 0:
                colab_runtime.fail("run1's checkpoints are not all on Drive (see above)")
        """, expect=[r"PASS: \d+ files of run run1 on Drive \(Drive API\)"], timeout_min=30),
        md("""
            **Sample every checkpoint.** For each checkpoint (10k, 20k, 30k, 36.5k) and once
            for the pretrained voice, synthesises the 5 test sentences (val lines) with the
            fixed seed and noise scales into Drive `eval/run1/step_<N>/`. Audio from the first
            run 1 is replaced, never reused. Expect one `step N: mean X s` line per step and
            `PASS: sampled K checkpoints`.
        """),
        code("C8.sample", "Sample every checkpoint", "!boinga sample --run run1",
             expect=[r"PASS: sampled \d+ checkpoints"], gpu=True),
        md("""
            **Rank checkpoints.** Matches each checkpoint to the nearest validation pass in the
            training log and sorts by validation `loss_mel`. Writes
            `reports/candidates_run1.md`. Expect the report block and `PASS: candidates listed`.
        """),
        code("C8.cand", "Rank checkpoints", "!boinga candidates --run run1",
             expect=[r"PASS: candidates listed", r"=== BEGIN reports/candidates_run1.md ==="]),
        md("""
            **Listen: pretrained vs the checkpoints.** The 5 sentences by the pretrained male
            voice, then by each checkpoint (best eval loss_mel first, then the last); each
            label starts with its step. Listen for clarity, Chittagonian pronunciation and
            artefacts (buzz, metallic noise, skipped or repeated words). Nothing is checked
            automatically: write what you hear below.
        """),
        code("C8.play", "Listen: pretrained vs the checkpoints", """
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
