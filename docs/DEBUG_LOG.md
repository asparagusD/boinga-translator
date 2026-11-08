# Debug log

Every bug handled with the debug-protocol skill gets an entry here (step 7),
newest last, committed together with its `fix(<scope>): ...` commit.

Template:

```
## YYYY-MM-DD C<n>.<m> <one-line symptom>
**Symptom:** <exact command, error / traceback excerpt, where it ran (local / Colab)>
**Root cause:** <the first wrong value and why it was wrong>
**Fix:** <smallest change that fixed the root cause>
**Test added:** <tests/test_x.py::test_name, or the Colab cell re-run and its output>
**Commit:** <short hash>
```

## Entries

## 2026-11-07 C1.5 Colab cell 4: ImportError: cannot import name 'env_report'
**Symptom:** First Colab bootstrap run. Cell 4 raised `ImportError: cannot import
name 'env_report' from 'boinga_translator'`; cell 2 had printed
`e5c1a7b merge(C0): ...` and `PASS: /content/boinga-translator on main`.
**Root cause:** `BRANCH` in cell 2 was left at its default `"main"`, so Colab
checked out main (C0 only). `env_report.py` exists only on `chunk/C1-paths-env`.
Operator step, not a code bug.
**Fix:** Set `BRANCH = "chunk/C1-paths-env"` and re-ran cells 2 and 4 (cell 3's
`[train]` deps are identical on both branches). Cell 2 then printed `cc31783`, and
cell 4 printed `PASS: env recorded`.
**Test added:** none (no code change). Follow-up idea: make cell 2 warn loudly when
`BRANCH == "main"` and newer `chunk/*` branches exist.
**Commit:** chore(colab): record first Colab environment report (C1.5 commit B)

## 2026-11-07 C1.5 Colab cell 2 pull aborts on an untracked reports/env.json
**Symptom:** Found in the whole-chunk C1 review: on a reused Colab VM where cell 4
ran before reports/env.json was committed, re-running cell 2 to pull a commit that
tracks it fails with `error: The following untracked working tree files would be
overwritten by merge: reports/env.json` -> `FAIL: git pull failed`.
**Root cause:** cell 2 (and the skill's standard cell header) only ran
`git checkout -- reports/`, which restores tracked files; reports created on Colab
stay untracked until they are committed locally and pushed.
**Fix:** also run `git clean -fdq -- reports/` before fetch/switch/pull in cell 2 and
in the colab-handoff standard header (header pull aligned to `--ff-only`). The pasted
block and the Drive copy are the record.
**Test added:** tests/test_colab_bootstrap.py::test_guards_and_branch asserts
checkout -> clean -> switch order; scratch-repo repro: old sequence errors, fixed
sequence pulls `track env`.
**Commit:** fix(colab): drop untracked Colab reports before pulling

## 2026-11-07 C3.1 build_notebook: slug must be lowercase letters, digits and -
**Symptom:** Local `python tools/build_notebook.py notebooks/specs/c3_data_audit_run1.py`
raised `ValueError: slug must be lowercase letters, digits and -, got 'data_audit'`.
**Root cause:** The C3.1 prompt gave `slug="data_audit"`; nbkit.Notebook only allows
`[a-z0-9-]` in slugs (they become notebook file names).
**Fix:** `slug="data-audit"` (matches the branch name and the C2 `pretrained-baseline`
style); the notebook is notebooks/runs/C3_data-audit_run1.ipynb.
**Test added:** none (nbkit already validates it; tests/test_nbkit.py covers the rule).
**Commit:** data(download): fetch corpus zip with resume, sha256 check and one-time unzip

## 2026-11-07 C3.1 Colab: fetch-data fails with HTTP Error 403: Forbidden
**Symptom:** C3 run 1 (commit c692819, Colab CPU, python 3.13.18), cell C3.1
`!boinga fetch-data` printed `FAIL: fetch-data: HTTP Error 403: Forbidden`.
**Root cause:** the request carried urllib's default `User-Agent: Python-urllib/3.x`,
which Mendeley's Cloudflare front rejects. Local probe of the pinned URL with a 100-byte
range: `Python-urllib/3.13` -> 403 (no redirect); `curl/8.5.0` and a
`boinga-translator/...` agent -> 302 to S3, then `206 Partial Content`,
`Content-Range: bytes 0-99/843601272` (Range works; size matches the pin).
**Fix:** send `User-Agent: boinga-translator/<version> (research dataset download)`
(data_download.USER_AGENT) on every request.
**Test added:** tests/test_data_download.py::test_requests_send_project_user_agent
**Commit:** fix(download): send a project User-Agent; Mendeley 403s Python-urllib

## 2026-11-07 C3.2 Colab: layout inspector PASSes on an empty corpus folder
**Symptom:** C3 run 1 (c692819): after C3.1 failed, cell C3.2 inspected
`/content/boinga-fast/data` with `./ (no files)`, `wav files: 0` and still printed
`PASS: layout inspected` (the reader marked it PASS).
**Root cause:** tools/inspect_layout.py main() only checked that ROOT is a folder; an
empty or audio-less tree is not a corpus, but nothing treated it as an error.
**Fix:** main() prints `FAIL: no .wav files under <root>` and exits 1 (writing no report)
when the tree holds no .wav files.
**Test added:** tests/test_inspect_layout.py::test_main_fails_on_empty_or_wavless_tree
**Commit:** fix(data): fail the layout inspector on a tree without wav files

## 2026-11-07 C3.1 Colab: Run all continued past the failed download cell
**Symptom:** C3 run 1 (c692819): cell C3.1 `!boinga fetch-data` printed
`FAIL: fetch-data: HTTP Error 403: Forbidden` then `CELL-OK C3.1`, and Run all went on
to C3.2.
**Root cause:** a `!` shell command exiting 1 does not raise in Colab (known C2
follow-up), so nothing stopped the run.
**Fix:** cell C3.1 calls `cli.main(["fetch-data"])` from Python and calls
`colab_runtime.fail()` (SystemExit 1, stops Run all) on a non-zero exit; notebook rebuilt.
**Test added:** none new: tests/test_notebooks.py keeps the notebook in sync with the
spec; tests/test_colab_runtime.py covers fail(); the re-run shows the cell's behaviour.
**Commit:** fix(notebook): stop Run all when the C3.1 download fails

## 2026-11-07 C3.6 pre-commit: test_repo_prompts_file_shape fails on the C3 run 2 prompts
**Symptom:** Committing the C3 run 2 results, the pre-commit pytest failed:
`tests/test_baseline.py::test_repo_prompts_file_shape` - `Left contains one more item:
'_sources'` (local, working tree holding configs/baseline_prompts.json from run 2).
**Root cause:** The C2 test pinned the file to exactly STD_1-3 + CTG_1-3 with CTG null;
C3.6 (05e51ce) added `_sources` and fills CTG_1-3, but did not update this test (the
committed file still had nulls, so it passed until the run-2 file arrived).
**Fix:** The test accepts both states: CTG null without `_sources`, or CTG filled with
`_sources` {CTG_k: digit clip id}; STD checks unchanged.
**Test added:** the updated tests/test_baseline.py::test_repo_prompts_file_shape passes on
the committed file and on the run-2 file.
**Commit:** fix(baseline): accept filled dialect prompts in the prompts-file shape test

## 2026-11-07 C6.4 Colab: `boinga prepare` fails with KeyError 'clip_id' before any transform
**Symptom:** C6 run 1 (1a9b636, Colab CPU, Python 3.13): cell C6.4 printed "selection:
7506 clips, 0 excluded by char_map" then `FAIL: prepare: KeyError: 'clip_id'`; C6.3 and
C6.4.play failed on the missing metadata_train.csv. Locally all tests passed.
**Root cause:** `prepare.make_splits` did `df.loc[wanted].reset_index()` with `wanted` an
unnamed pd.Index. pandas 3 keeps the frame's index name ("clip_id"); pandas 2.x names
the result's index after the indexer (None), so reset_index made a column "index" and
the next `df["clip_id"]` raised. The local venv has pandas 3.1.1, Colab an older 2.x
(not recorded: env.json has no pandas field). Confirmed by running the prepare tests
under pandas 2.3.3 with uv: 18 failed with the same KeyError.
**Fix:** `df.loc[wanted].rename_axis("clip_id").reset_index()`.
**Test added:** none new; the existing tests/test_prepare.py and test_prepare_build.py
reproduce it under pandas 2.3.3 (uv run --with pandas==2.3.3) and pass after the fix
(54 passed; full suite 455 passed under pandas 2.3.3 and 3.1.1).
**Commit:** fix(prepare): keep the clip_id column name on pandas 2 in make_splits

## 2026-11-08 C7.1 Colab: coqui loads the fine-tune config but drops the seed override
**Symptom:** C7 run 1 (87c0470, Colab GPU): cell C7.1 printed "PASS: config written" and
"sample_rate 22050, batch_size 16, lr_gen 0.0001, ..." then `FAIL: coqui loaded the config
but not these overrides: ['seed']`. The smoke and resume cells still ran; their log shows
"Torch seed: 54321" (the pretrained config's value), not settings.SEED 1234.
**Root cause:** coqui's training seed field is `training_seed` (TrainerConfig); the
pretrained config.json has `training_seed: 54321` and no `seed` key (the overrides table
shows `seed` as "(missing)"). coqpit's load_json only reads known fields, so our `seed`
override was dropped silently. tools/validate_config.py (compares every override after the
load) caught it, as designed.
**Fix:** override `training_seed` instead of `seed` in train_config.OVERRIDES.
**Test added:** tests/test_train_config.py checks `training_seed` == settings.SEED and that
no `seed` key is written; the real coqui load is re-checked by cell C7.1 in the re-run.
**Commit:** fix(train): set coqui's training_seed instead of an ignored seed key

## 2026-11-08 C8 run 1: the trained checkpoints never reached Drive
**Symptom:** C8 run 1 (8005e5d) passed 13/13 cells and the user picked step 36500 by ear, but
the C8.2c keep notebook (55ba369, CPU) failed: `no checkpoint under .../runs/run1`. On Drive
(read through the Drive connector) the run folder holds only best_model_1203.pth,
best_model_1604.pth and best_model.pth (997.8 MB each); every file in it last changed at
22:13 UTC, 8 minutes into a run that ended ~00:27, including trainer_0_log.txt and the
TensorBoard events file. Small files elsewhere kept arriving (runs/run1/boinga_run.json at
00:26, eval/run1/step_*/ at 00:39-00:43). Drive free: 6.4 GB at the start, 0.8 GB after.
No checkpoint_*.pth exists on Drive or in its trash.
**Root cause (most likely; Drive's own logs are not visible):** the Colab Drive mount writes
locally and uploads in the background. coqui writes a ~1 GB checkpoint every 500 steps and
deletes older ones; a file deleted through the mount goes to Drive's trash and keeps using
space. With 6.4 GB free, Drive dropped below the size of one checkpoint after a few saves,
every later upload in the run folder failed silently and its queue stalled; small files in
other folders still fit. The VM saw its own local copies, so sampling and ranking of all
17 checkpoints worked; they vanished with the VM. The C7 follow-up "check Drive free space
before C8" was still open when run 1 started.
**Fix:** runs train on the VM's fast disk (paths.runs() -> fast()/runs), where deletes free
space at once; `boinga train` refuses to start with < 20 GB free there; sampled audio is tied
to the run folder (a retrained run never reuses old clips). C8.5b copies the listened
checkpoints to Drive milestones/ and checks after a flush and remount that they arrived.
**Test added:** tests/test_paths.py (runs on the fast disk), tests/test_train_run.py
(test_new_run_needs_disk_space), tests/test_ckpt_sampler.py (test_retrained_run_resamples).
**Commit:** fix(train): train runs on the VM disk so Drive never holds deleted checkpoints
