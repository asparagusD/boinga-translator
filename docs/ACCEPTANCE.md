# Acceptance checklist — boinga-translator v0.1.0

Evidence for the v0.1.0 release of the Chittagonian VITS proof of concept (C13.4, 2026-11-09).
Every hash is a commit on `main` (or on `chunk/C13-release`, merged with this release); every
digest is in `reports/colab_runs/`. Verdicts: PASS / PARTIAL / FAIL.

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | One target dialect explicitly defined (Chittagonian) | PASS | CLAUDE.md project rules (`444d614`); DECISIONS "Stack: fine-tune Bangla VITS on the Chittagonian corpus" (`28ea218`); dialect-data-rules skill (`e369d76`) |
| 2 | Dataset licence and access conditions checked | PASS | Pinned DOI 10.17632/n9xttk45df.1, CC-BY-4.0, authors, zip URL, size and sha256 in `src/boinga_translator/settings.py` (`93f6feb`); verified download (size + sha256) recorded in `03ae94b`, digest `C3_data-audit_run1.md`; DECISIONS "Licences and attribution (C13.2b)" adds the Kothon source (`ddf301e`) |
| 3 | Audio and transcripts paired and validated (audit summary) | PASS | `reports/audit_summary.md` (`639f189`, char-mapped re-audit `89bd7ce`); digests `C3_data-audit_run2.md`, `C3_data-audit_run3.md`; Gate G2 conditional PASS (`af6d26d`): 5 of 8,009 rows undecodable, row <-> wav mapping r = 0.87 |
| 4 | Pretrained checkpoint provenance and licence documented | PASS | `reports/pretrained_inspect.md` and `configs/pretrained_facts.json` (`633a1f5`); download sizes and sources (`6b4aa11`, digest `C2_pretrained-baseline_run1.md`); Apache-2.0 and author @mobassir94 in `settings.py`; `reports/best_provenance.json` names the base model and licence (`42dd33e`) |
| 5 | Baseline inference works (G1) | PASS | Gate G1 PASS (`9eecb2a`, corrected `910a13b`); baseline listening notes `reports/notes/C2.4-notes.json` (`abce7f8`); digest `C2_pretrained-baseline_run1.md` (10/10 PASS) |
| 6 | Fine-tuning data loader works (C7.2) | PASS | `train_runner.check_loader` + test (`4c24b7a`); passing loader check on Colab (`b2d1884`), digest `C7_train-runner_run1.md` |
| 7 | Smoke test completed incl. resume (C7.4) | PASS | 50-step smoke test + resume to 60, throughput recorded (`40a1a25`); Gate G5 PASS (`889cbfc`); digest `C7_train-runner_run1.md`; re-measured at the RTX PRO 6000 settings (`0df8c10`, digest `C8_timing_run1.md`) |
| 8 | Reloadable fine-tuned checkpoint (C9.3) | PASS | best/ promoted with provenance (`42dd33e`); fresh-process reload check "PASS: reload check (3 wavs)" (`a95535f`), digest `C9_synth-best_run1.md` (10/10 PASS); Gate G6 PASS (`09e4cbc`) |
| 9 | Adapted model speaks unseen text (C9.3 / C10.2) | PASS | Reload check reads test-split clips 6, 29, 58, never trained on (`a95535f`); B_boinga speaks the 20 held-out test-split prompts (`5388ccd`, `reports/eval_generation.md`, digest `C10_eval_run2.md`) |
| 10 | Baseline and adapted outputs use the same prompts (C10.2) | PASS | `configs/eval_prompts.json` (20 reviewed prompts, `97e9cee`); `evaluate.generate` speaks each prompt with A_baseline and B_boinga (`5388ccd`): 20 wavs per set in `reports/eval_generation.md` |
| 11 | Audio samples saved and indexed | PASS | `evaluate.generate` writes Drive `eval/final/{A_baseline,B_boinga,R_real}/` and `eval/final/index.csv`; run 2 generated 20/20/20 and the C10.2.play cell read them back through index.csv (digest `C10_eval_run2.md`, 9/9 PASS, `5388ccd`); re-run on a fresh runtime in S15 of `boinga_translator.md` |
| 12 | Limitations and evaluation method stated honestly | PASS | `reports/eval.md` method, limitations and relative-only MCD (`b3e5ffa`, `5c68b34`); DECISIONS "C10 evaluation result"; README Results and Limitations, numbers tested against their sources (`ddf301e`) |
| 13 | Notebook reruns from a fresh Colab runtime (G7) | PARTIAL | Gate G7 PASS: 31/31 cells on a fresh T4 runtime with `RUN_TRAINING = False` (`a7f6174`, digest `boinga_translator.md`). Since then C13.1 (`c946dde`) changed S17 from printing env.json to `!boinga repro` (new expect patterns); the released notebook has not been re-run on Colab (C13.H/V skipped) |
| 14 | No paid resource used | PARTIAL | Free: Colab T4 (G7, `a7f6174`), Google Drive, GitHub; Hugging Face not used (C13.3 skipped). Paid: training and timing ran on an NVIDIA RTX PRO 6000 through the Colab pay-as-you-go plan (`reports/throughput.json`, DECISIONS "RTX PRO 6000 training settings", "Run 1 budget": "paid GPU time"), the one paid resource CLAUDE.md allows (written for an A100). No paid API or service |
| 15 | Redistributed artefacts comply with their licences | PASS | The repo holds code, configs and text reports only: no audio, transcripts dump, checkpoints or dataset (pre-commit guard `8f56f81`, `.gitignore`); fine-tuned weights stay on the owner's Drive and were not uploaded (C13.3 skipped); README "Licences and attribution" credits the base model (Apache-2.0) and the dataset authors + DOI (CC BY 4.0) (`ddf301e`); our code is Apache-2.0 (`LICENSE`, `ddf301e`) |
| 16 | The name boinga-translator is used throughout | PASS | `git grep -i -E "dialectadapt\|dialect-adapt\|dialect_adapt"` -> no match in tracked files other than this checklist; `git log --all -i --grep=dialectadapt` -> only the C13.4 commits that quote this search; DECISIONS "Naming: boinga-translator / boinga_translator" (`28ea218`) |
| 17 | Every Colab result commit has a matching digest | PARTIAL | Every executed notebook has its digest: 19 for `notebooks/runs/` (C1 pilot to C11) plus `boinga_translator.md` for the C12 notebook. `notebooks/runs/C1_bootstrap_run1.ipynb` is the spec-built bootstrap that replaced the pasted C1.5 cells; it is a setup notebook with no results of its own. One result commit has none: `9f760cf` (C1.5, first `reports/env.json`), saved from pasted cell output before the notebook reader existed; the C1 pilot re-verified that environment with a digest (`fa46243`, `C1_pilot_run1.md`). `notebooks/runs/C13_release_run1.ipynb` is built but was not run (C13.H/V skipped), so it has no result commit yet |

## Summary

14 PASS, 3 PARTIAL, 0 FAIL. The PARTIAL items: paid Colab GPU time for training (allowed by
CLAUDE.md), the released notebook's changed S17 not yet re-run on Colab, and one pre-tooling C1
report without a digest. Skipped in C13 (user, chat): the model card, Hugging Face upload and the C13 Colab
run (C13.3a/b, C13.H/V).
