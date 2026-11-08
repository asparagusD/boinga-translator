"""Spec: C8 run 1, full fine-tune of run1 to its step budget, then sample, rank and listen."""

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
            ready` and `PASS: prepared <n> clips`. Before the first run, empty the Drive trash:
            run1 keeps 2 checkpoints + the best (~1 GB each) on Drive.
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

            **If Colab disconnects:** reconnect and "Run all" again. Data is restored, and
            training resumes from the newest checkpoint on Drive (`RESUME from step N`); at most
            the steps since the last save (every 500 steps) are repeated.
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
            `train_config.RUN_BUDGETS` (36,500 steps, ~91 epochs at batch 16; ~3.5 h of step
            time measured, plus eval and checkpoint saves, not measured). coqui's step logs are
            the progress output. Expect `budget: 36500 steps (RUN_BUDGETS)` and `STOP at step
            36500`; a re-run after the end prints `already at step 36500` and trains nothing.
        """),
        code("C8.train", "Train run1 to its budget (hours; resumable)",
             "!boinga train --run run1",
             expect=[r"STOP at step \d+|PASS: run run1 already at step \d+"],
             forbid=[r"Partial model initialization"], gpu=True, timeout_min=360),
        md("""
            **Throughput of run1.** Reads run1's own coqui log and run record: seconds per step,
            steps per hour, peak GPU memory and any NaN loss into `reports/throughput.json`
            (replaces the smoke timing). Expect `PASS: throughput recorded`; a NaN loss prints
            `FAIL: NaN loss`.
        """),
        code("C8.thr", "Throughput of run1", "!boinga throughput --run run1",
             expect=[r"PASS: throughput recorded"]),
        md("""
            **Sample every checkpoint.** For each checkpoint still on Drive (the last 2 + the
            best model), and once for the pretrained voice, synthesises the 5 test sentences
            (val lines) with the fixed seed and noise scales into `eval/run1/step_<N>/`. Steps
            already sampled are skipped. Expect one `step N: mean X s` line per new step and
            `PASS: sampled K checkpoints`.
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
            **Listen: pretrained vs best candidates vs last.** The 5 sentences by the pretrained
            male voice, then by the 4 best checkpoints by `loss_mel` and the last one; each
            label starts with its step. Listen for clarity, Chittagonian pronunciation and
            artefacts (buzz, metallic noise, skipped or repeated words). Nothing is checked
            automatically: write what you hear in the form below.
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
