"""Spec: C12 end-to-end notebook: the whole pipeline in 17 sections, training behind RUN_TRAINING."""

# Every cell calls an existing `boinga` command or a tested function already run in C1-C11
# (tests/test_final_spec.py checks each `!boinga <cmd>` against cli.build_parser()). Cells with
# skip_unless="RUN_TRAINING" print "SKIPPED (RUN_TRAINING=False)" when the flag is off, which
# the reader counts as PASS. Promote stays manual (`boinga promote --run run1 --step <N>`).

from nbkit import Notebook, code, md

RT = "RUN_TRAINING"
REPO_URL = "https://github.com/asparagusD/boinga-translator"

NOTEBOOK = Notebook(
    chunk="C12",
    run=1,
    slug="final",
    title="boinga-translator — end-to-end",
    branch="chunk/C12-notebook",
    output="notebooks/boinga_translator.ipynb",
    gpu=True,
    cells=[
        md(f"""
            # boinga-translator — Chittagonian text-to-speech, end to end

            This notebook runs the whole proof of concept in 17 sections: environment, data,
            checks, the pretrained baseline, fine-tuning, the best checkpoint, a comparison
            with the baseline and the Gradio demo. Every step is a tested `boinga` command
            from the repo; the cells only call them. See the [README]({REPO_URL}#readme).

            **Licences.** Base model: Bangla VITS `tts_models/bn/custom/vits-male` (coqui-tts
            0.28.0), Apache-2.0. Data: "A Synchronized Chittagonian Speech Corpus" (Mendeley
            Data, DOI 10.17632/n9xttk45df.1), CC BY 4.0. Fine-tuned weights stay on your
            Google Drive, never in the repo.

            **Drive.** Everything is kept under `MyDrive/boinga-translator/` and reused, so
            "Run all" can be repeated: finished work is skipped, training resumes.
        """),
        md("""
            **Settings: `RUN_TRAINING`.** Off (default): the smoke test, the fine-tune, the
            checkpoint samples and the ranking print `SKIPPED (RUN_TRAINING=False)`, and the
            rest uses the fine-tuned model already promoted to Drive `best/`. On: those cells
            run (the fine-tune resumes run1 from its newest checkpoint; a fresh run1 takes
            about 2.4 h on an RTX PRO 6000 and ~6 GB of Drive). Tick the box before "Run
            all". Expect `RUN_TRAINING = False` (or `True`).
        """),
        code("S0", "Settings: RUN_TRAINING", """
            RUN_TRAINING = False  #@param {type:"boolean"}
            print(f"RUN_TRAINING = {RUN_TRAINING}")
        """, expect=[r"^RUN_TRAINING = (True|False)$"]),

        md("## 1. Environment & GPU"),
        md("""
            **Environment.** Records Python, torch/CUDA, the GPU, coqui-tts, numpy, git and
            disk space into `reports/env.json` and prints them as a table (the run header
            recorded the same; this shows it in its own section). Expect a `gpu` row
            naming the GPU and `PASS: env recorded`.
        """),
        code("S1", "Environment and GPU", "!boinga env",
             expect=[r"PASS: env recorded"]),

        md("## 2. Install"),
        md("""
            **Pinned install.** The bootstrap cell BOOT.3 above already ran
            `pip install -e .[train]` (coqui-tts 0.28.0, gradio and the pinned extras), quietly
            in a subprocess. This cell only prints the installed versions. Expect
            `coqui-tts 0.28.0` and one line per package.
        """),
        code("S2", "Installed versions", """
            from importlib.metadata import version

            for pkg in ("coqui-tts", "torch", "numpy", "gradio", "boinga-translator"):
                print(f"{pkg} {version(pkg)}")
        """, expect=[r"^coqui-tts 0\.28\.0$", r"^boinga-translator "]),

        md("## 3. Dataset download + licence"),
        md("""
            **Download the corpus.** Streams the pinned Mendeley zip to Drive once (resumes if
            interrupted), checks its size and sha256, unzips it once and copies it to the fast
            VM disk, with progress bars; later runs reuse the Drive copy. Then prints the
            dataset's DOI and licence. Expect `PASS: dataset ready` and
            `licence CC-BY-4.0`.
        """),
        code("S3", "Download the corpus", """
            !boinga fetch-data
            from boinga_translator import settings

            print(f"dataset {settings.DATASET_NAME}")
            print(f"doi {settings.DATASET_DOI}, licence {settings.DATASET_LICENSE}")
        """, expect=[r"PASS: dataset ready", r"doi 10\.17632/n9xttk45df\.1, licence CC-BY-4\.0"]),

        md("## 4. Inspect + playback"),
        md("""
            **Index the corpus.** One row per spreadsheet line with its wav, Chittagonian text,
            Bangla translation and split. Expect `PASS: index has 8009 rows (8004 with
            audio)`.
        """),
        code("S4", "Index the corpus", "!boinga index",
             expect=[r"PASS: index has 8009 rows \(8004 with audio\)"]),
        md("""
            **Audit every clip (resumable).** Per clip: format, duration, levels, silence, and
            characters the voice does not know, judged on the char-mapped text; saved in parts
            to Drive `reports/audit.csv`, so a re-run only checks what changed. A progress bar
            counts clips. Expect `PASS: audited 8009 clips`.
        """),
        code("S4.audit", "Audit every clip", "!boinga audit",
             expect=[r"PASS: audited 8009 clips"], timeout_min=30),
        md("""
            **Listen.** 10 random clean clips (fixed seed) with their Chittagonian text and
            Bangla translation. Nothing is checked automatically. Expect 10 `AUDIO` lines.
        """),
        code("S4.play", "Listen: 10 corpus clips", """
            from boinga_translator import audit_summary, colab_runtime

            colab_runtime.play(*audit_summary.sample_clips(10))
        """, expect=[r"AUDIO \d+ \["]),

        md("## 5. Validation"),
        md("""
            **Audit summary.** Applies the exclusion rules (bad audio, outliers, characters
            excluded by `configs/char_map.json`) and writes `reports/audit_summary.md`. Expect
            `PASS: summary written` and the report between BEGIN/END markers.
        """),
        code("S5", "Audit summary", "!boinga audit-summary",
             expect=[r"PASS: summary written", r"=== BEGIN reports/audit_summary.md ==="]),

        md("## 6. Preprocess + manifests"),
        md("""
            **Prepare the training data (resumable).** Splits, 22,050 Hz wavs and coqui
            manifests on the fast disk; a fresh VM restores the Drive tar backup instead of
            recomputing (progress bars). It trains on the speaker selection chosen by ear in
            C4 (cluster 1, base voice male), which must already be on Drive; this notebook
            does not redo that choice. Expect `PASS: prepared <n> clips` and the
            `configs/splits_summary.json` block.
        """),
        code("S6", "Prepare the training data", "!boinga prepare",
             expect=[r"PASS: prepared \d+ clips",
                     r"=== BEGIN configs/splits_summary.json ==="]),
        md("""
            **Coqui parses our manifests.** Loads the train/val/test manifests with coqui's own
            ljspeech loader and checks counts, audio files and text. Expect `PASS: coqui parsed
            <n> train / <n> val samples`.
        """),
        code("S6.parse", "Coqui parses the manifests", "!python tools/check_manifest_parse.py",
             expect=[r"PASS: coqui parsed \d+ train / \d+ val samples"]),

        md("## 7. Pretrained checkpoint + config"),
        md("""
            **Pretrained voices.** Downloads the female and male Bangla VITS models once
            (~1 GB each, cached on Drive), then reads each config and checkpoint: sample rate,
            character set, weights. Writes `configs/pretrained_facts.json`. Expect `PASS:
            female cached`, `PASS: male cached`, `PASS: inspected female`, `PASS: inspected
            male`.
        """),
        code("S7", "Fetch and inspect the pretrained voices",
             "!boinga fetch-pretrained --voice both\n!boinga inspect-pretrained --voice both",
             expect=[r"PASS: female cached", r"PASS: male cached", r"PASS: inspected female",
                     r"PASS: inspected male"]),

        md("## 8. Baseline"),
        md("""
            **Baseline audio.** Both pretrained voices read `configs/baseline_prompts.json`
            (standard Bangla and Chittagonian lines) with the fixed seed; a progress bar per
            voice. Each wav must be longer than 0.3 s and louder than -50 dBFS. Expect `PASS:
            baseline female` and `PASS: baseline male`.
        """),
        code("S8", "Baseline synthesis", "!boinga baseline --voice both",
             expect=[r"PASS: baseline female", r"PASS: baseline male"], gpu=True),

        md("## 9. Loader check"),
        md("""
            **One-batch loader check.** Builds the VITS model and its train data loader from
            `configs/vits_ctg_finetune.json` exactly as training does and pulls one batch:
            shapes, token ids against the vocabulary, clip lengths, NaN/inf. Expect `PASS:
            loader ok`.
        """),
        code("S9", "One-batch loader check", "!boinga check-loader",
             expect=[r"PASS: loader ok"]),

        md("## 10. Smoke test (RUN_TRAINING)"),
        md("""
            **Fresh smoke run.** Deletes the Drive folder of run `smoke` (only runs named
            `smoke...` under step 1000; ~2 GB goes to the Drive trash) so the smoke test starts
            from the pretrained checkpoint. Expect `PASS: cleaned run smoke` or `PASS: nothing
            to clean`, or `SKIPPED (RUN_TRAINING=False)`.
        """),
        code("S10", "Fresh smoke run", "!boinga clean-run --run smoke",
             expect=[r"PASS: (cleaned run smoke|nothing to clean for run smoke)"],
             skip_unless=RT),
        md("""
            **Smoke test: 50 steps, then resume to 60.** Starts run `smoke` from the pretrained
            male checkpoint and stops at step 50, then runs again with a higher limit: it must
            resume from the step-50 checkpoint, as after a Colab disconnect. coqui's step logs
            are the progress output. Expect `STOP at step 50`, `RESUME from step 50` and `STOP
            at step 60`, or `SKIPPED (RUN_TRAINING=False)`.
        """),
        code("S10.train", "Smoke test and resume",
             "!boinga train --run smoke --max-steps 50\n!boinga train --run smoke --max-steps 60",
             expect=[r"STOP at step 50", r"RESUME from step 50", r"STOP at step 60"],
             forbid=[r"Partial model initialization"], gpu=True, timeout_min=60,
             skip_unless=RT),

        md("## 11. Fine-tune, resumable (RUN_TRAINING)"),
        md("""
            **Fine-tune run1 (hours; resumable).** Trains run1 from the pretrained male
            checkpoint, or resumes its newest checkpoint, up to its step budget (36,500 steps,
            `train_config.RUN_BUDGETS`); a checkpoint goes to Drive every 10,000 steps. If
            Colab disconnects, reconnect and "Run all" again. Expect `STOP at step 36500` or
            `already at step 36500`, then `PASS: throughput recorded`; or `SKIPPED
            (RUN_TRAINING=False)`.
        """),
        code("S11", "Fine-tune run1", "!boinga train --run run1\n!boinga throughput --run run1",
             expect=[r"STOP at step \d+|PASS: run run1 already at step \d+",
                     r"PASS: throughput recorded"],
             forbid=[r"Partial model initialization"], gpu=True, timeout_min=360,
             skip_unless=RT),

        md("## 12. Validation samples (RUN_TRAINING)"),
        md("""
            **Sample every checkpoint.** For each run1 checkpoint and the pretrained voice,
            synthesises 5 validation sentences with the fixed seed into Drive
            `eval/run1/step_<N>/`. Expect `PASS: sampled <n> checkpoints`, or `SKIPPED
            (RUN_TRAINING=False)`.
        """),
        code("S12", "Sample every checkpoint", "!boinga sample --run run1",
             expect=[r"PASS: sampled \d+ checkpoints"], gpu=True, skip_unless=RT),
        md("""
            **Listen: pretrained vs the checkpoints.** The 5 sentences by the pretrained voice,
            then by each checkpoint (best validation loss first), each labelled with its step.
            Nothing is checked automatically.
        """),
        code("S12.play", "Listen: checkpoint samples", """
            from boinga_translator import ckpt_sampler, colab_runtime

            colab_runtime.play(*zip(*ckpt_sampler.players("run1"), strict=True))
        """, expect=[], skip_unless=RT),

        md("## 13. Best checkpoint"),
        md("""
            **Rank the checkpoints.** Matches each run1 checkpoint to the nearest validation
            pass and sorts by validation `loss_mel` into `reports/candidates_run1.md`. Expect
            `PASS: candidates listed` and the report block, or `SKIPPED (RUN_TRAINING=False)`.

            **Promote stays manual.** Choose a step by ear (12.) and the ranking, then run
            `!boinga promote --run run1 --step <N>` in a new cell: it copies that checkpoint
            to Drive `best/` with its provenance. The current `best/` is run1 step 36500.
        """),
        code("S13", "Rank the checkpoints", "!boinga candidates --run run1",
             expect=[r"PASS: candidates listed", r"=== BEGIN reports/candidates_run1.md ==="],
             skip_unless=RT),

        md("## 14. Unseen text"),
        md("""
            **The best model on held-out text.** A fresh Python process loads Drive `best/`
            (nothing from earlier cells in memory) and speaks the first 3 rows of the test
            split, which training never saw, into Drive `eval/reload_check/`. Each wav must be
            longer than 0.5 s and louder than -50 dBFS. Expect `PASS: reload check (3 wavs)`.
        """),
        code("S14", "Speak held-out text with best/", "!python tools/reload_check.py",
             expect=[r"PASS: reload check"], gpu=True),
        md("""
            **Listen: held-out text.** The 3 wavs, each labelled with its clip id and text.
            Nothing is checked automatically.
        """),
        code("S14.play", "Listen: held-out text", """
            from boinga_translator import colab_runtime, synth

            colab_runtime.play(*synth.reload_players())
        """, expect=[]),

        md("## 15. Baseline vs fine-tuned"),
        md("""
            **Evaluation audio.** For the 20 held-out prompts in `configs/eval_prompts.json`:
            the pretrained voice (A), the fine-tuned voice (B) and the real recording (R) in
            Drive `eval/final/`. Wavs already there with the same text are kept, so a re-run
            only fills gaps; after promoting a new `best/`, delete Drive `eval/final/B_boinga/`
            first or B stays the old model. Expect `PASS: generated A=20 B=20 R=20`.
        """),
        code("S15", "Evaluation audio", "!boinga eval-generate",
             expect=[r"PASS: generated A=20 B=20 R=20"], gpu=True),
        md("""
            **Listen: A/B/R for 5 prompts.** P01-P05, each as A (baseline), B (fine-tuned) and
            R (real recording), labelled with the text. The blind listening test and its
            scores are in `reports/eval.md`. Nothing is checked automatically.
        """),
        code("S15.play", "Listen: baseline vs fine-tuned", """
            from boinga_translator import colab_runtime, evaluate

            colab_runtime.play(*evaluate.players(("P01", "P02", "P03", "P04", "P05")))
        """, expect=[r"AUDIO "]),

        md("## 16. Gradio demo"),
        md("""
            **Automated demo checks.** Calls the demo page's own button handler without the
            page: a held-out prompt in both voices, empty text, 400 characters, Latin letters
            and Bengali digits. Expect one `CASE` line per case and `PASS: demo selfcheck (6
            cases)`.
        """),
        code("S16", "Automated demo checks", "!python tools/demo_selfcheck.py",
             expect=[r"PASS: demo selfcheck"], gpu=True),
        md("""
            **Live demo.** Keeps running while the demo is live: open the public link
            (`Running on public URL: https://....gradio.live`) and try both voices. Anyone with
            the link can use it while the cell runs. Press Stop on the cell when done; if
            section 17 did not run after that, run it (Shift+Enter).
        """),
        code("S16.live", "Live demo (stop the cell when done)", """
            from boinga_translator.app_gradio import build_app
            from boinga_translator.synth import VoiceBank

            build_app(VoiceBank()).launch(share=True, debug=True)
        """, expect=[], allow_interrupt=True),

        md("## 17. Reproducibility report"),
        md("""
            **Environment of this run.** Prints `reports/env.json` (written by the run header:
            GPU, versions, git commit) between BEGIN/END markers. C13.1 extends this report.
            Expect `=== BEGIN reports/env.json ===`.
        """),
        code("S17", "Reproducibility report", """
            from boinga_translator import paths, report_io

            report_io.emit(paths.reports_dir() / "env.json", "reports/env.json")
        """, expect=[r"=== BEGIN reports/env.json ==="]),
    ],
)
