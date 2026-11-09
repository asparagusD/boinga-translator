# Progress

## End-of-day 2 handover (2026-11-08)
- Done: C8 (run1 retrained, 4 checkpoints on Drive) and C9 merged to main (`ad4b9ba`); Gate G6 PASS:
  run1 step 36500 promoted to Drive best/ (reports/best_provenance.json), reloads in a fresh process,
  user: intelligible.
- On Drive (boinga-translator/): runs/run1/ (10k/20k/30k/36.5k, ~4 GB), best/ (model.pth ~1 GB +
  config + provenance), eval/run1/, eval/reload_check/; 4.5 GB free before the best/ copy.
- First task tomorrow: C10 plan (line up the listening raters first; /clear before C10).
- Risks: Drive nearly full (empty the trash); G6 rests on one listener and 3 clips; best/ not checked
  through the Drive API.
- Open bugs: none blocking; follow-ups: coqui "newline not in vocabulary" (C8), Drive API check of best/ (C9).

## End-of-day 1 handover (2026-11-08)
- Done: C0-C7 merged to main (`d49ee06`); Gate G5 PASS: the runner restores the pretrained male
  VITS, smoke 50 steps + resume to 60 passed on Colab (0.34 s/step, 3.39 GB, no NaN).
- On Drive (boinga-translator/): corpus zip + unzip, prepared-fd93f4827c6d2b41.tar, pretrained/
  {female,male}, runs/smoke (checkpoints 50 and 60, ~2 GB), speaker_emb/, reports/.
- First task tomorrow: C8 plan: make `boinga train` use RUN_BUDGETS (it ignores it today), pick
  the C8 batch size for the RTX PRO 6000, then build-config + train the full run.
- Risks: Drive ~6.4 GB free (~1 GB per checkpoint, the old smoke run sits in the Drive trash);
  T4-based override reasons; a run's settings are fixed once it starts (new name to change).
- Open bugs: none blocking; C7 follow-ups below (C8 section); C7.4.d listening note missing.

Tick a subchunk in the same commit that completes it (boinga-workflow step 6),
with its short commit hash. Chunks C1–C13 get their subchunk list when the
promptbook for that chunk is handed over.

## C0 — Working system (no ML code)
- [x] C0.1 Repo skeleton, .gitignore, .gitattributes, pyproject.toml (`31e9666`)
- [x] C0.2 CLAUDE.md project rules (`444d614`)
- [x] C0.3a Skills: boinga-workflow, atomic-commit (`97acf15`)
- [x] C0.3b Skills: repo-graph, debug-protocol (`104814d`)
- [x] C0.3c Skills: colab-handoff, dialect-data-rules, vits-finetune (`e369d76`)
- [x] C0.4 Subagents: code-reviewer, debugger (`12a8f5c`)
- [x] C0.5a Repo graph tool: tools/repo_graph.py + generated map (`57b91a7`)
- [x] C0.5b Repo graph tests (`73bed9f`)
- [x] C0.5c Claude Code PostToolUse hook + permissions allow list (`c0fb112`)
- [x] C0.6 Git hooks: commit-msg format, binary guard, fresh repo graph (`8f56f81`)
- [x] C0.7 PROGRESS, DECISIONS and DEBUG_LOG docs (`28ea218`)

## C1 — Paths, settings, report I/O, env report, Colab bootstrap
- [x] C1.1 paths.py: single source of truth for paths (Drive on Colab, ./local elsewhere) (`8834a02`)
- [x] C1.2 settings.py: pinned model ids, dataset URL and sha256, seed (`93f6feb`)
- [x] C1.3 report_io.py: print files between BEGIN/END markers and import pasted output (`63f4c1b`)
- [x] C1.4 env_report.py + `boinga env` CLI (`4d3399f`)
- [x] C1.5 Colab bootstrap cells (notebooks/colab_bootstrap.py) + first Colab env report;
  two commits: A code + local tests (`cc31783`), B pasted reports/env.json (`9f760cf`);
  fix: drop untracked Colab reports before pulling (`29c7184`)
- [x] C1.6 Colab notebook notebooks/c1_colab_bootstrap.ipynb generated from the bootstrap
  cells by tools/build_notebooks.py (sync enforced by tests/test_notebooks.py);
  superseded by C1.6c (notebooks/runs/C1_bootstrap_run1.ipynb, tools/build_notebook.py)

### C1 follow-ups (from the whole-chunk review; not blocking)
- [ ] Bootstrap cell 2: warn loudly when BRANCH == "main" and newer chunk/* branches exist
- [ ] Drive budget: 13.0 GiB free after cleanup (was 5.9); keep unzipped data on /content,
  last 2 + best checkpoints, empty Drive trash after deletes (decide in the data/train chunks)
- [ ] report_io: Drive copy lands at ROOT/reports/reports/...; mirror repo paths as root()/rel
- [ ] paths.require_drive(): reusable Drive-mount guard instead of ensure_dirs() in write_report
- [ ] paths.COLAB_REPO constant + test that bootstrap literals match paths constants
- [ ] paths.REPO assumes an editable/source install: assert REPO/pyproject.toml exists
- [ ] DECISIONS: bootstrap cells may hold logic (they run before the package is importable)
- [ ] Optional: enforce ruff E501 (100 chars) and wrap the existing long lines

## C1N — notebook run pipeline
Branch `chunk/C1N-notebook-pipeline`. Per chunk: Claude Code writes code + a notebook spec;
the generated notebook is run once on Colab; the executed .ipynb goes to `colab_runs/`
(git-ignored); `tools/read_colab_run.py` verifies it and saves its reports per subchunk.
- [x] C1.6a tools/nbkit.py: spec API (md / code / form cells, cell ids, expectations)
- [x] C1.6b tools/build_notebook.py: spec -> .ipynb with the bootstrap cells and branch
  baked in, spec hash in metadata (the commit is recorded at run time by C1.7), `--check` for stale notebooks
- [x] C1.6c retire tools/build_notebooks.py: C1 bootstrap notebook is now
  notebooks/runs/C1_bootstrap_run1.ipynb from notebooks/specs/c1_bootstrap.py
- [x] C1.7 src/boinga_translator/colab_runtime.py: run_header (RUN-HEADER line + env
  report), cell_ok (CELL-OK line), record_notes, expect_files, play
  Merge gate: C1N must not merge to main without C1.7 (every built notebook's HDR cell
  calls run_header; BOOT.2 checks out the spec's branch, main for the C1 bootstrap)
- [x] C1.8a tools/read_colab_run.py core: load executed .ipynb, drop binary outputs, trim text,
  identify cells, per-cell status vs the spec (PASS / FAIL / ERROR / NOT RUN), form values,
  token redaction
- [x] C1.8b read_colab_run CLI: spec auto-found from RUN-HEADER, staleness, digest
  reports/colab_runs/<chunk>_<slug>_run<k>.md (slug added in ecdc585), stdout table, --save-reports, exit code
- [x] C1.9 colab-handoff skill update, new colab-verify skill, CLAUDE.md, .gitignore
  (`colab_runs/`), DECISIONS entry; colab-verify must say a STALE run exits 1 (re-run it)
- [x] C1.10 pilot: env-check spec + notebook, run on Colab, verified with the reader
  (A = spec + notebook, B = reports from the verified run)
- [x] C1.11a notebook rules in CLAUDE.md and the colab-handoff skill: an md() explanation
  before every cell; tqdm for cells that may run over a minute (BOOT.3 pip is the exception)
- [x] C1.11b every built notebook explains every cell: the builder writes the BOOT.1-4/HDR
  explanations, keeps a form's md() with it and refuses an unexplained cell; md() added to
  all C1-C6 specs (text only), notebooks rebuilt. The C1-C6 run notebooks changed, so an
  old executed copy in colab_runs/ now reads STALE: expected, not a regression
- [x] C1.11c tqdm progress for the slow steps without one (no logic change): corpus unzip and
  fast-disk copy, speaker embedding, baseline synthesis, prepared tar restore and backup

### C1N follow-ups (before the C1N merge)
- [x] reports/env.json is emitted twice per run (BOOT.4 and HDR's run_header);
  report_io.parse_blocks rejects duplicates, so save_pasted_reports fails on a full paste.
  Make HDR the only emitter in built notebooks; the C1.8 reader parses per cell anyway
  Done: BOOT.4 no longer records env. Known exception: the C1 pilot spec's own C1.10.a
  cell (user-specified) still emits env.json next to HDR
- [x] Stop-on-FAIL: cell_ok only prints CELL-OK and expect_files only returns False, so a
  FAIL line does not stop Run all; decide raise-in-cell vs reader-only (C1.8 / C1.9)

## C2 — Pretrained Bangla VITS inspection and baseline audio
Branch `chunk/C2-pretrained-baseline`. One notebook for the chunk:
spec `notebooks/specs/c2_pretrained_baseline.py` ->
`notebooks/runs/C2_pretrained-baseline_run1.ipynb`
(T4). Every subchunk that adds cells rebuilds the notebook (pre-commit `--check`); only
C2.H hands it over. Local tests never import torch/TTS (fakes only).
- [x] C2.1 pretrained.fetch(voice) + `boinga fetch-pretrained --voice female|male|both`:
  coqui ModelManager, fallback to the pinned release zip (safe unzip); model.pth + config.json
  cached in paths.pretrained()/<voice>/ (`7cc2d92`); results: both voices 997.77 MB, via
  ModelManager (`6b4aa11`)
- [x] C2.2 pretrained.inspect(voice) + `boinga inspect-pretrained --voice both` (`668da92`);
  results: configs/pretrained_facts.json, reports/pretrained_inspect.md, charsets (`633a1f5`)
- [x] C2.3 synth_basic.py: set_seed(), load_synth(), say() (`cf2728e`); Colab run confirmed
  all 3 inference scales are set
- [x] C2.4 baseline.py + `boinga baseline`, configs/baseline_prompts.json, cells C2.4,
  C2.4.play, form C2.4-notes (`68ab521`); results: both voices intelligible (`abce7f8`).
  Not done from the plan: reports/baseline.json and RTF
- [x] C2.H hand-off: notebook already built and pushed in `68ab521` (no new commit)
- [x] C2.V run 1 verified: 10/10 PASS, not stale; results commits `6b4aa11`, `633a1f5`,
  `abce7f8`. Nukta (U+09BC) is in both charsets, but NFC decomposes য়/ড়/ঢ়, so the token
  form the model learned is unverified (C3 follow-up)
- [x] C2.5 Gate G1: PASS (DECISIONS "Gate G1 — pretrained model"; this commit)

### C2 follow-ups
- [x] coqui temp-path rewrite: none seen; the 4 absolute config paths are the model
  author's (/home/ansary/Shabab/...); the fine-tune config must override them (C4)
- [ ] Record and pin sha256 of the cached model.pth/config.json (moved from C2.5)
- [x] C3 text front-end: how coqui's tokenizer treats "।": it is in the vocab as the eos
  token (characters.py _create_vocab); mapped to "." anyway (DECISIONS "Character map",
  `ae735d9`)
- [ ] C4: the "Initializing with English" language log of the coqui synthesizer (harmless?
  check that no English text processing is applied to Bangla input)
- [x] C3: precomposed vs NFC-decomposed য়/ড়/ঢ়: precomposed after NFC via char_map
  (DECISIONS "Character map", `ae735d9`)
- [ ] C4 (was C3): one pretrained.model_symbols(config) helper; the charset is read two
  ways today (baseline._model_charset from the cached config.json,
  data_audit.load_model_charset from pretrained_facts.json); use paths.configs_dir()
- [ ] C4: copy the full characters block from the pretrained config.json (is_unique /
  is_sorted fix embedding row order); measure RTF/throughput in the smoke test
- [ ] `inspect-pretrained --voice <one>` rebuilds pretrained_inspect.md with that voice
  only; use `--voice both` to refresh it (or merge the .md like the facts JSON)
- [ ] C4: choose the voice to fine-tune (female or male); baseline quality was "same", the
  corpus has many speakers (DECISIONS "Speakers": mixed voices); the audit defaults to
  female, re-run it with `--voice male` if male is chosen
- [ ] `!boinga fetch-pretrained` exiting 1 does not stop Run all; once later cells depend on
  C2.1, call pretrained from Python and stop with colab_runtime.fail()
- [x] After C3: add held-out Chittagonian corpus lines (verbatim transcripts) to the
  baseline prompts and re-run the pretrained baseline on them (C3.6 `05e51ce`, `8893a67`)

## C3 — Chittagonian corpus download and audit
Branch `chunk/C3-data-audit`. Three notebooks, one per run: run 1 records the real
corpus layout, run 2 is written against it (index, audit, summary, dialect baseline),
run 3 re-audits on the char-mapped text. Local tests use tiny synthetic data only.

Run 1 (spec `notebooks/specs/c3_data_audit_run1.py`, CPU):
- [x] C3.1 code: data_download.fetch() + `boinga fetch-data` + cell C3.1: zip to Drive with
  Range resume, size + sha256 check, one-time unzip on Drive, per-VM copy to the fast disk
  (`93964ba`); fixes `5fbf2bd` (User-Agent, Mendeley 403), `f06163c` (stop Run all);
  results: 8,005 files (8,004 wav + 1 xlsx), sha256 ok, 844 MB in 23 s (this commit)
  Review follow-up: time unzip-to-Drive + copy vs extracting the Drive zip straight to /content
- [x] C3.2 code: tools/inspect_layout.py + cell C3.2: tree to depth 3 with counts per
  extension, table heads (csv/tsv/txt/json/xlsx/xls < 5 MB), 3 wav headers, speaker/gender/
  region/age/split values -> reports/corpus_layout.txt (`c692819`); fix `1b9275f` (no-wav
  FAIL); results: layout recorded in DECISIONS "Corpus layout" (this commit)
- [x] C3.H1 hand-off run 1: notebook built and pushed with C3.2 (`c692819`), re-run at
  `f06163c` after the fixes (no separate commit)
- [x] C3.V1 verify run 1: 7/7 PASS at `f06163c` (first attempt at `c692819` failed: 403);
  results `03ae94b` (download) and this commit (reports/corpus_layout.txt + DECISIONS)

Run 2 (spec `notebooks/specs/c3_data_audit_run2.py`):
- [x] C3.3 code: corpus.load_index() + `boinga index` + run-2 spec (cells C3.2.fetch, C3.3);
  wav stem = Excel row (DECISIONS "Corpus row <-> wav mapping"); 8,009 rows, 8,004 with
  audio (`14d90cb`); results: Colab run 2 index train 6403 / val 801 / test 800 / none 5,
  8,004 with audio, no speaker field (this commit)
- [x] C3.4a data_audit.audio_stats() (local): format, duration, peak, clip_ratio, rms_db,
  lead/trail silence (librosa trim, top_db 40) (this commit)
- [x] C3.4b data_audit.text_stats() + load_model_charset() (local): lengths, chars not in
  the voice's characters|punctuations|space, oov_ratio, Latin, digits (this commit)
- [x] C3.4c code: data_audit.audit() + `boinga audit --voice` (default female: G1 "same")
  + cell C3.4c: every index row -> Drive reports/audit.csv, Pool(2), resumable
  (`c9874aa`); results: Colab run 2 audited 8,009 rows in 47 s (168.7 rows/s, Pool(2),
  T4 VM CPU), charset female-a4543ed63b; Drive reports/audit.csv + audit.parts (this commit)
- [x] C3.5 code: audit_summary.clean_mask/summarise/sample_clips + `boinga audit-summary`
  + cells C3.5, C3.5.play, form C3.5-notes -> reports/audit_summary.md (`8631cdb`);
  results: 9.04 h decodable, 2,881 clean clips / 4.10 h, listening "mostly" matched,
  "standard Bangla", clean, many speakers (this commit)
- [x] C3.6 code: `boinga pick-baseline-ctg` (clean test-split clips, 5-12 words, least like
  the Bangla) -> CTG_1-3 verbatim + _sources in configs/baseline_prompts.json; cells C3.6,
  C3.6.play, form C3.6-notes (`05e51ce`); fix `b87125e` (prompts-file test); results:
  CTG_1-3 = clips 386, 29, 625; 12/12 baseline wavs PASS; HANDLES_DIALECT "partly"
  (this commit)
- [x] C3.H2 hand-off run 2: notebook built for a T4 runtime (this commit)
- [x] C3.V2 verify run 2: 14/14 PASS at `f4c7ab2`; results `db85fcf` (index), `c894178`
  (audit), `639f189` (summary + listening), `8893a67` (dialect baseline)
- [x] C3.7 Gate G2: first FAIL (`ea56f4a`), re-decided PASS with conditions after the user
  clarified "dialect words, standard accent" (DECISIONS "Gate G2 — data (re-decided)";
  this commit)

### C3 follow-ups (conditions of Gate G2, before C4)
- [x] C3.8a Danda "।" and other out-of-charset characters: configs/char_map.json +
  DECISIONS "Character map" (also settles the nukta form: precomposed) (this commit)
- [x] C3.8b text_frontend.prepare/excluded (NFC + char_map); audit stats on the mapped
  text (ctg_text_model, excluded_char, charset_id includes the map digest); clean_mask:
  "excluded character" + "unknown after char_map" replace oov_ratio > 0.02; baseline
  synthesises the mapped text (this commit)
- [x] C3.8c re-run audit + summary on Colab with the map: clean 7,513 clips / 8.18 h
  (was 2,881 / 4.10 h); G2 condition 1 met
  - [x] C3.H3 run-3 notebook (fetch, audit, summary; CPU) built and pushed (`eb3d3e0`)
  - [x] C3.V3 verify run 3: 8/8 PASS at `eb3d3e0`, audit 8,009 rows in 62 s (this commit)
- [x] C3.9 Speakers: mixed voices (user decision, measuring declined); fallback = largest
  SpeechBrain ECAPA cluster if C4 hears an unstable voice (DECISIONS "Speakers"; this commit)
- [ ] Keep the official test split (incl. clips 386, 29, 625) out of training in C4

### C3 follow-ups (from the whole-chunk review; for C4)
- [ ] Before C4 builds splits: compute duplicate_text on ctg_text_model (the mapped text)
  instead of ctg_text_nfc, so identical model inputs cannot land in two splits
- [ ] audit_summary: reject an audit.csv whose charset_id was made with another char map
  (compare with text_frontend.map_digest()), not only one missing columns
- [ ] baseline._model_charset docstring: the check now runs on text_frontend.prepare()
  output, not "raw text before coqui's cleaner"
- [ ] Tests: move ROWS/_corpus (test_corpus) and _row (test_audit_summary) into
  tests/conftest.py or tests/fakes.py; pin test_repo_prompts_file_shape to the filled state
- [ ] oov_ratio and has_digits are informational now (no rule reads them); drop or label

## C4 — Choose one consistent training speaker
Branch `chunk/C4-speaker`. One notebook for the chunk: spec `notebooks/specs/c4_speaker.py`
-> `notebooks/runs/C4_speaker_run1.ipynb` (T4). Every subchunk that adds cells rebuilds
the notebook (pre-commit `--check`); only C4.H hands it over. Local tests use fake
encoders and synthetic embeddings, never speechbrain or the corpus.
Path: no speaker ids (reports/audit_summary.md "no speaker field"; DECISIONS "Corpus
layout") -> C4.2a, C4.2b, C4.3. This chunk revisits DECISIONS "Speakers" (A, mixed voices):
the form keeps "mixed" selectable, and C4.4 records the outcome as a new entry.
Line counts are non-test estimates, excluding the generated map/notebook and PROGRESS.
- [ ] C4.1 skipped: the corpus has no speaker ids, so there is no speaker to pick from
  metadata (DECISIONS "Corpus layout"; audit_summary.md "Speakers")
- [x] C4.2a code: `speaker_select.embed(audit_df, cache_dir=paths.speaker_emb())`: one
  embedding per clean clip (clean_mask, all three splits) cached on Drive as
  `speaker_emb/<clip_id>.npy`, cached clips skipped, `encoder.txt` marker refuses mixing two
  encoders, >1% failed clips -> FAIL; resemblyzer 0.1.4 (VoiceEncoder + preprocess_wav,
  pinned in [train] with setuptools<81), SpeechBrain ECAPA fallback if it cannot import
  (extra `speaker-fallback`); `boinga embed-speakers`; `paths.speaker_emb()`; cells `C4.0`
  (fetch-data) and `C4.2a` in `notebooks/specs/c4_speaker.py`; DECISIONS "Speaker encoder";
  tests `tests/test_speaker_select.py` (fake encoder) (this commit)
  - [x] C4.2a results: C4 run 1 (ad690fd, T4): 7,513 of 7,513 clean clips embedded with
    resemblyzer-0.1.4, 0 failed; webrtcvad imported pkg_resources with a deprecation
    warning only; duration not recorded by the cell (this commit)
- [x] C4.2b code: `speaker_select.cluster(emb_dir, audit_df)`: sklearn
  AgglomerativeClustering (cosine, average linkage) at thresholds 0.20-0.50 step 0.05 (tree
  built once, cached), per threshold clusters / largest-cluster clean minutes / silhouette;
  chosen = best silhouette among thresholds whose largest cluster has >= 30 clean min, else
  most minutes; clusters ranked 1..K by clean minutes; labels -> Drive
  `speaker_clusters.csv` (`paths.speaker_clusters()`); `reports/speaker_ranking.md`
  (threshold table, top 20 clusters with train/val/test minutes and 5 seeded clip ids,
  emitted); `candidates(n=3)`; `boinga cluster-speakers`; cell `C4.2b`; DECISIONS "Speaker
  clustering"; tests `tests/test_speaker_cluster.py` (synthetic 2- and 3-voice embeddings)
  (this commit). Listening to the top clusters is not in this subchunk (C4.3 play cell)
  - [x] C4.2b results: C4 run 1: the rule chose threshold 0.50 (silhouette 0.196), 3
    clusters: #1 7,506 clips / 490.73 min (all speakers merged), #2 5 clips / 0.18 min, #3 2
    clips / 0.11 min; threshold 0.30 had 187 clusters, largest 72.08 min (silhouette 0.125)
    (this commit). The rule is degenerate on this corpus: see C4.4
- [x] C4.3 code: `speaker_select.listen_clips(id, n=10, seed)` (seeded clips of one
  ranked cluster, labelled with the verbatim transcript), `play_candidates(3)` (heading with
  clean minutes per candidate), `save_selection(id, base_voice)` -> Drive
  `speaker_selection.csv` (`paths.speaker_selection()`) + `configs/speaker_selection.json`
  {kind "cluster", id, n_clips, clean_minutes, minutes_by_official_split, csv_sha256,
  base_voice} (emitted), `save_from_form()` (saves only with CHOSEN_ID, GENDER and
  ONE_CONSISTENT_VOICE = yes, else `NOTES-MISSING choice incomplete`); `boinga save-speaker
  --id N --base-voice female|male`; cell `C4.3.play`, form `C4.3-notes` (CHOSEN_ID,
  ONE_CONSISTENT_VOICE, GENDER, AUDIO_QUALITY, NOTES); DECISIONS "Speaker choice"; tests
  `tests/test_speaker_choice.py` (this commit)
  - [x] C4.3 results: mixed voices = cluster 1 (7,506 clips, 490.73 clean min; train
    393.72 / val 48.25 / test 48.76), base voice male; configs/speaker_selection.json from
    run 2, form notes from run 1; DECISIONS "Training speaker" (this commit)
  - Moved out of C4 (C4.3 promptbook): per-split training manifests, cross-split duplicate
    text on `ctg_text_model`, text_frontend.prepare, resampling (training-data prep, C5)
- [x] C4.H hand-off: `notebooks/runs/C4_speaker_run1.ipynb` (T4) rebuilt, unchanged since
  `d856b3b`, pushed; link, form instructions and save name `colab_runs/C4_speaker_run1.ipynb`
  given; duration not measured (embedding estimate 20-40 min) (this commit)
- [x] C4.V verify run 1 (colab-verify): 10/10 PASS at `ad690fd`, not stale; results
  `abc441e` (C4.2a), `1e15f75` (C4.2b). Form: CHOSEN_ID 2 (5 clips, 0.18 min), one
  consistent voice, male, some noise. The 0.50 threshold merged 7,506 of 7,513 clips into
  cluster 1, so the user chose (chat) mixed voices = cluster 1, base voice male; the run-1
  selection (cluster 2) is superseded and was not committed
- [x] C4.3b run 2 (CPU, `notebooks/specs/c4_speaker_run2.py`): `boinga save-speaker --id 1
  --base-voice male` from run 1's Drive speaker_clusters.csv -> configs/speaker_selection.json
  (built `fcb64e5`)
- [x] C4.V2 verify run 2: 6/6 PASS at `fcb64e5`, not stale; results in the C4.3 commit
- [ ] Follow-up: the threshold rule (best silhouette among largest >= 30 min) picks a
  near-single cluster on this corpus; cap the largest cluster's share before reusing it
- [x] C4.4 Gate G3: PASS (490.73 clean min >= 30; DECISIONS "Gate G3 — speaker data";
  this commit). Rule from the C4.V promptbook: >= 30 min PASS, 10-30 PASS WITH RISK,
  < 10 FAIL -> C5, C6, then Chunk F

### C4 follow-ups (from the whole-chunk review; for C5)
- [ ] C5 checks `csv_sha256` of configs/speaker_selection.json against the Drive
  speaker_selection.csv before building training lists (with a test); the CSV holds only
  clip_id, split_official, duration_s, so join audit.csv for wav paths and transcripts
  - [x] sha check: `speaker_select.load_selection()` + tests (C5.2); training lists must
    still use it (C6)
- [ ] Take the speaker only from configs/speaker_selection.json: reports/notes/C4.3-notes.json
  keeps the superseded run-1 pick (CHOSEN_ID "2")
- [ ] save_selection writes the Drive CSV before the repo JSON: write the JSON first or rely
  on the sha check above
- [ ] speaker_select.py tidy-ups: `lines =[` spacing in _render; use SPLITS in the ranking
  table instead of the literal tuple
- [ ] reports/env.json now holds run 2's CPU environment; run 1's T4 environment is in
  DECISIONS "Speaker encoder" and the run-1 digest

## C5 — Text frontend for Chittagonian text
Branch `chunk/C5-text-frontend`. Builds on C3.8 (text_frontend.prepare/excluded,
configs/char_map.json, already approved). Two CPU notebooks: run 1 = OOV report on the C4
selection against the base voice (male), run 2 = coqui tokenizer check. Local tests use
synthetic text, fake charsets and a fake tokenizer (coqui is not installed locally).
- [x] C5.1 `text_frontend.normalise()` (local): NFC + collapse + strip moved from
  `corpus.normalise_text` into text_frontend (single home; corpus and baseline call it),
  behaviour unchanged, so audit charset_id/map digest stay valid; tests pin what Python's
  NFC does (U+09DC -> U+09A1 U+09BC; U+09A1 U+09BC unchanged) (`9faf56c`)
- [x] C5.2 code: `text_oov.oov_report(texts, charset)` (char, U+XXXX, name, count,
  clips_affected, 3 example words) + `render()` + `boinga oov`: the selected clips
  (`speaker_select.load_selection()`, csv_sha256 checked against the Drive CSV) as
  text_frontend.prepare(ctg_text_nfc), char_map-excluded rows counted, not checked, vs
  the base voice's charset -> reports/oov_report.md (table, OOV char rate, share of clips),
  emitted; spec `notebooks/specs/c5_text_frontend_run1.py` (cells C5.0 fetch-data, C5.2;
  CPU) -> `notebooks/runs/C5_text-frontend_run1.ipynb`; own module text_oov.py because
  data_audit imports text_frontend (this commit). Narrower than the C5.0 plan, per the
  C5.2 promptbook: after-map text only, no per-split counts, no .json (add later if Gate G4
  needs them)
- [x] C5.H1 hand-off run 1: `notebooks/runs/C5_text-frontend_run1.ipynb` (CPU) rebuilt,
  unchanged since `86064cb`, pushed; link and save name
  `colab_runs/C5_text-frontend_run1.ipynb` given; no forms; duration not measured (this
  commit)
- [x] C5.V1 verify run 1: 7/7 PASS at `9683e31` (notebook built in `86064cb`), not stale;
  reports/oov_report.md: 7,506 selected clips, 0 excluded, OOV 0 of 251,223 characters
  (0.00%), 0 clips with an OOV character vs the male charset, char map 151ac177e2 (this
  commit)
- [x] C5.3 `apply_char_map(text, mapping=None)`, `prepare_text()` (= apply_char_map of
  normalise, whitespace re-collapsed; prepare() delegates) and `count_unknown(text,
  charset)` in text_frontend; configs/char_map.json unchanged (user: no new entries, keep
  the C3.8 map and its list format; digest 151ac177e2, no re-audit); DECISIONS "Character
  map" C5.3 note (this commit)
- [x] C5.4 code: `text_frontend.check_tokenizer(texts, config_path)`: coqui TTSTokenizer
  (`load_tokenizer`, lazy import, refuses a cleaner or phonemes) on prepare_text() of
  every kept selected clip (`text_oov.selected_texts()`); `round_trip()` strips bos/eos
  and blanks, counts characters encode() drops, compares decode with the text ->
  reports/tokenizer_check.md (unknown-char rate, failing clips with code points, first 10
  failures in full), emitted, "PASS: tokenizer check unknown_rate=..."; `boinga
  check-tokenizer` (config from `pretrained.fetch(base voice)`); spec
  `notebooks/specs/c5_text_frontend_run2.py` (cells C5.0, C5.4; CPU) ->
  `notebooks/runs/C5_text-frontend_run2.ipynb`; tests with a fake tokenizer
  (this commit). No token-length stats or .json (narrower than the C5.0 plan, per the
  C5.4 promptbook)
- [x] C5.H2 hand-off run 2: `notebooks/runs/C5_text-frontend_run2.ipynb` (CPU) rebuilt,
  unchanged since `49cc675`, pushed; link and save name
  `colab_runs/C5_text-frontend_run2.ipynb` given; no forms; duration not measured (this
  commit)
- [x] C5.V2 verify run 2: 7/7 PASS at `90b60c3` (notebook built in `49cc675`), not stale;
  reports/tokenizer_check.md: coqui TTSTokenizer from the male config.json, 7,506 clips,
  unknown 0 of 297,218 characters (spaces included), 0 round-trip failures (this commit)
- [x] C5.5 Gate G4 — text frontend: PASS, tokenizer unknown-char rate 0 of 297,218
  (0.00% < 1%), 0 round-trip failures (DECISIONS "Gate G4 — text frontend"; this commit).
  Rule from the C5.V2 promptbook: < 1% PASS, else FAIL -> widen char_map or extend the
  text embedding (C7.0)

### C5 follow-ups (from the whole-chunk review; for C6)
- [ ] `text_oov.selected_texts()` returns every selected clip, official test split
  included: G4 checked a superset of the training text (verdict holds), but C6 must
  filter on split_official when it builds training lists and not reuse it as-is
- [ ] One "unknown character" predicate: oov_report's inline loop, count_unknown and the
  tokenizer's _known agree in practice; the run-1 rate excludes spaces, the run-2 rate
  includes them; share the predicate and word both reports the same way
- [ ] prepare and prepare_text are the same function (text_oov.run uses prepare,
  check_tokenizer uses prepare_text): pick one name for C6 and inference
- [ ] cli `_oov` vs `_check_tokenizer` exception lists differ (FileNotFoundError is
  redundant next to OSError; `_oov` lacks RuntimeError/ImportError): align them
- [ ] The C4 follow-up note on reports/env.json is stale: since `e0beff4` it holds C5
  run 2's CPU environment (git 90b60c3)
- [ ] Inference/demo: check typed prompts with count_unknown() before synthesis (coqui
  drops unknown characters with only a warning; DECISIONS "Gate G4")
- [ ] Optional skip-if-no-TTS (or Colab) test of round_trip on the real TTSTokenizer, so a
  change in coqui's blank/bos/eos order is caught without a full run

## C6 — Preprocess audio and write training manifests
Branch `chunk/C6-prepare`. One CPU notebook: spec `notebooks/specs/c6_prepare.py` ->
`notebooks/runs/C6_prepare_run1.ipynb`. Input = the C4 selection (7,506 clips,
configs/speaker_selection.json, sha-checked), char_map-excluded rows dropped, text =
`text_frontend.prepare_text` (C5 follow-up: one name for C6 and inference). Target rate from
configs/pretrained_facts.json (male 22,050 Hz), never hard-coded. Local tests use synthetic
wavs/text; coqui is faked locally and only parse-checked on Colab. Line counts exclude tests,
the generated map/notebook and PROGRESS.
- [x] C6.1 `prepare.target_sample_rate()` (base voice in configs/speaker_selection.json
  -> sample_rate in configs/pretrained_facts.json) and `prepare.transform_audio(in_wav,
  out_wav, target_sr)`: mono mix, librosa resample soxr_hq when the rate differs, trim
  edge silence (top_db 40) keeping a 100 ms pad each side (clamped), peak to -1 dBFS,
  16-bit PCM written atomically; returns in_sr, out_sr, in_dur, out_dur, trimmed_s,
  gain_db, upsampled; empty/non-finite/silent -> ValueError. Peak, not RMS, normalisation
  per the C6.1 promptbook; module prepare.py (C6.4a's build() joins it). DECISIONS "Audio
  preprocessing". Tests `tests/test_prepare.py` (this commit)
- [x] C6.2 `prepare.make_splits(index_df, selection_ids, seed=settings.SEED)` ->
  (clip_id, split, duration_s): selected clips only, ctg_text_nfc duplicates dropped
  (longest kept, counted), official test if >= 20 clips else a seeded 5% (min 20) sample,
  val = seeded 5% (min 20) of the rest, train = the rest, asserts no transcript in two
  splits; `save_splits()` -> Drive `paths.splits_csv()` + configs/splits_summary.json
  (counts, minutes, csv_sha256, seed, test_source, dropped_duplicate_text), refuses to
  change a saved test set. In prepare.py (not splits.py), per the C6.2 promptbook.
  DECISIONS "Training splits". Tests in `tests/test_prepare.py` (this commit)
- [x] C6.3 code: `prepare.write_manifests(splits_df, index_df, out_root)` -> coqui ljspeech
  `metadata_{train,val,test}.csv` (`<clip_id>|<text>|<text>`, UTF-8, LF, no header, text =
  prepare_text(ctg_text); refuses missing/empty text and "|"/CR/LF), wavs at
  `<root>/wavs/<clip_id>.wav` (`manifest_wav`); formatter + file names in
  configs/pretrained_facts.json "dataset" (`manifest_names`); `prepare.parse_check()` +
  `tools/check_manifest_parse.py`: coqui load_tts_samples (0.28 needs a config with
  `datasets`) on all 3, counts, audio files exist, text equal (trailing "\n" allowed: the
  formatter keeps it); spec `notebooks/specs/c6_prepare.py` (cell C6.3, CPU) ->
  `notebooks/runs/C6_prepare_run1.ipynb`. DECISIONS "Training manifests". Tests in
  `tests/test_prepare.py` (fake loader) (this commit). Confirmed on Colab in C6.V
  - [x] C6.3 results: C6 run 1 (93d9796): coqui-tts 0.28.0 load_tts_samples parsed
    6,415 train / 338 val (+ 753 test) samples, every audio file present, text equal to
    the manifests; first sample text ends with "\n" as the source read predicted
    (this commit)
- [x] C6.4 code (one subchunk per the C6.4 promptbook; replaces the planned C6.4a/b):
  `prepare.build(workers=2)` + `boinga prepare [--workers N]`: corpus index, C4 selection
  (sha-checked) minus char_map-excluded rows, target_sample_rate(); make_splits +
  save_splits; transform_audio of every split clip into `paths.data_prepared()/wavs`
  (multiprocessing Pool, tqdm; a clip with a `clip_stats.csv` row and its wav, or an
  error, is skipped); failed clips counted by reason and left out of the manifests;
  write_manifests; tar of the prepared folder to Drive
  `paths.prepared_backup()/prepared-<splits sha256[:16]>.tar` (not rewritten if present);
  a fresh VM (no local clip_stats.csv) restores that tar instead of recomputing; prints
  clips/minutes per split, upsampled count, mean trimmed s, failures, 3 manifest lines
  per split; emits configs/splits_summary.json; "PASS: prepared N clips".
  `prepare.pick_prepared(n)` (seeded train clips + manifest text). Spec cells C6.0
  (`!boinga fetch-data`), C6.4 (`!boinga prepare`) before C6.3, and C6.4.play after it.
  DECISIONS "Prepared data backup". Tests `tests/test_prepare_build.py` (fake corpus:
  end to end, second run skips, lost wav redone, restore, pool, CLI) (this commit)
  - Not done from the C6 plan (promptbook scope): post-trim 1-15 s exclusions,
    reports/prepare_summary.md, the listening form, counting model-text collisions
    across splits (follow-ups). dialect-data-rules wants exclusions written to reports/:
    char_map exclusions and transform failures are only printed until
    reports/prepare_summary.md exists. Review suggestions not taken: retry OSError rows,
    delete stale Drive tars automatically, a clean FAIL in pick_prepared without manifests
  - [x] C6.4 fix: run 1 attempt 1 (1a9b636) failed in C6.4 with KeyError 'clip_id'
    (pandas 2 on Colab); `93d9796` fix(prepare), DEBUG_LOG entry
  - [x] C6.4 results: C6 run 1 (93d9796, CPU): 7,506 clips prepared at 22,050 Hz, 0
    failed, 0 char_map-excluded, 0 duplicates; train 6,415 / val 338 / test 753 clips
    (official test), prepared minutes 386.64 / 20.48 / 44.81 (source 419.62 / 22.35 /
    48.76); all upsampled, mean trim 0.310 s; transform 1:42; Drive tar
    prepared-fd93f4827c6d2b41.tar; configs/splits_summary.json; DECISIONS "Prepared
    dataset" (this commit)
- [x] C6.H hand-off: `notebooks/runs/C6_prepare_run1.ipynb` (CPU) rebuilt, unchanged
  since `f3564b5`, `--all --check` clean, pushed; link and save name
  `colab_runs/C6_prepare_run1.ipynb` given; no forms (listening reported in chat);
  duration not measured (estimate 10-30 min) (this commit)
- [x] C6.V verify (colab-verify): run 1 attempt 1 (1a9b636) FAIL in C6.4 (pandas 2
  KeyError, fixed `93d9796`); attempt 2: 9/9 PASS at `93d9796` (notebook built in
  `f3564b5`), not stale; results `00fd048` (C6.4) and this commit (C6.3). Listening:
  5 clips played (1217, 131, 1131, 2582, 2124); user (chat): "All sounded fine", no
  cut-offs or level problems

### C6 follow-ups (from the whole-chunk review; for C7)
- [x] C7 notebook runs `boinga fetch-data` and `boinga prepare` first (wavs and manifests
  live on the VM's fast disk); check it prints "restored ..." and "0 clip(s) to do"
  (C7 run 1, cell C7.pre: restored from prepared-fd93f4827c6d2b41.tar, 0 clip(s) to do)
- [ ] Restore depends on byte-identical splits.csv: a different pandas/numpy on the GPU VM
  could change its bytes (e.g. float format of duration_s), miss the tar, recompute and
  write a second ~1.2 GB tar; warn when another prepared-*.tar exists but none matches
- [x] C7 config: set min/max audio length on trimmed durations (no post-trim 1-15 s
  exclusion in C6; trimming can push a clip under the minimum) (C7.1: 0.5 / 15 s)
- [ ] env_report: record pandas, librosa and soxr versions (Colab's pandas was unknown
  when the C6.4 KeyError hit)
- [ ] reports/prepare_summary.md with exclusions by reason (dialect-data-rules; C6 run 1
  had 0, counts only in the run digest); count model-text collisions across splits; a
  clean FAIL in pick_prepared when manifests are missing
- [ ] cli `_prepare`: also catch `tarfile.TarError` (corrupt Drive tar on restore)
- [ ] Doc wording: prepare.py header comment covers only the audio transform; DECISIONS
  "Training splits" / PROGRESS C6.2 say "asserts" (code raises RuntimeError); DECISIONS
  "Training manifests" says the C6.3 cell stays last (C6.4.play follows it); DECISIONS
  "Audio preprocessing" mentions a C6.4 form (listening came by chat); tar size estimates
  1.3 vs 1.2 GB (measure it on Drive)

## C7 — Training runner, smoke test and resume
Branch `chunk/C7-train-runner`. One T4 notebook for the chunk: spec
`notebooks/specs/c7_train_runner.py` -> `notebooks/runs/C7_train-runner_run1.ipynb`. Every
subchunk that adds cells rebuilds it (pre-commit `--check`); only C7.H hands it over. Two new
modules: `train_config.py` (config from the pretrained config.json + a documented overrides
dict) and `train_runner.py` (loader check, coqui Trainer setup, resume, throughput). Local tests
never import torch/TTS/trainer: fake config dicts, fake loaders, fake Trainer factories, tmp run
folders. Training cells are Python calls that stop with `colab_runtime.fail()` (a failing `!`
line does not stop Run all). Every cell gets an `md()` explanation before it, and slow work
(loader check, smoke/resume training, eval synthesis) shows tqdm progress (CLAUDE.md). Base
voice = male (configs/speaker_selection.json), rate 22,050
from configs/pretrained_facts.json. Line counts are non-test estimates, excluding the generated
map/notebook and PROGRESS.
- [ ] C7.0 skipped: Gate G4 PASS (0 of 297,218 characters unknown) chose no embedding change
  (DECISIONS "Gate G4 — text frontend")
- [x] C7.1 code: `train_config.build(run_name)` + `boinga build-config --run NAME`: base
  voice config.json from `base_dir()` (`paths.pretrained()/<voice>_extended` if present, else
  `<voice>`; voice from configs/speaker_selection.json), `OVERRIDES` as value + reason (batch
  16 / eval 8, loader workers 2/1, fp16, run_eval, test_delay_epochs 0, print_step 25,
  save_step 500, keep 2, best after 1000, seed, epochs 1000, restore_path null) plus computed
  ones (lr_gen/lr_disc 0.5 x pretrained, min/max audio 0.5/15 s x sample_rate, first 5 val
  manifest texts as test sentences, ljspeech dataset at `paths.data_prepared()` language bn,
  output `paths.runs()`, run_name); refuses a change to audio, characters, use_phonemes,
  text_cleaner, add_blank, model_args or a foreign absolute path; writes
  configs/vits_ctg_finetune.json + reports/config_overrides.md (emitted) and replaces the
  table in DECISIONS "Fine-tune config"; `RUN_BUDGETS = {}` (C8). `tools/validate_config.py`
  (coqui `VitsConfig().load_json`, "PASS: coqui accepts config"). Spec
  `notebooks/specs/c7_train_runner.py` (slug `train-runner`: nbkit slugs allow no "_"),
  cells C7.pre (fetch-data + prepare) and C7.1. Tests `tests/test_train_config.py` (this
  commit). Results: C7.1 failed in run 1 (seed dropped, fixed `08f29d0`)
  - [x] C7.1 results: C7 re-run (f681627, RTX PRO 6000): "PASS: config written ... 23
    overrides", "sample_rate 22050, batch_size 16, lr_gen 0.0001, lr_disc 0.0001,
    mixed_precision True", "PASS: coqui accepts config"; training_seed 1234 (pretrained
    54321), the smoke log shows "Torch seed: 1234"; configs/vits_ctg_finetune.json and the
    DECISIONS overrides table (built for run smoke: print_step 5) (this commit)
- [x] C7.2 code: `train_runner.check_loader(config_path=None)` + `boinga check-loader`:
  coqui VitsConfig from configs/vits_ctg_finetune.json, `Vits(config)` (0.28 has no
  Vits.init_from_config), `load_tts_samples(config, eval_split=True)` (0.28 takes the config,
  not config.datasets), prints `inspect.signature(model.get_data_loader)` and passes only the
  keyword arguments it names (0.28: is_eval, samples, verbose), one train batch through
  format_batch + format_batch_on_device (spec/mel); `judge_batch()`: per tensor shape/dtype,
  token ids vs the text-embedding rows, token lengths, clip seconds, NaN/inf counts; "PASS:
  loader ok" or "FAIL: loader <reason>". Cell C7.2. Tests `tests/test_train_runner.py`
  (numpy batches, torch ones via importorskip, fake coqui modules) (`4c24b7a`).
  - [x] C7.2 results: C7 run 1 (87c0470, NVIDIA RTX PRO 6000, not a T4): signature
    `get_data_loader(config, *, is_eval, samples, verbose)` as expected; 6,415 train / 338
    eval samples; batch 16 x 29 tokens, ids 0..86 of vocab 87; 0.78-0.97 s clips; spec
    (16, 513, 83), mel (16, 80, 83), no NaN/inf; first batch 3.74 s; "PASS: loader ok".
    coqui warns "Character '
' not found in the vocabulary. Discarding it." (the
    ljspeech formatter keeps the line's trailing "
", C6.3): harmless (this commit)
- [x] C7.3a code: `train_runner.run(run_name, max_steps=None)` + `boinga train --run NAME
  [--max-steps N]`: train_config.build, VitsConfig, `Vits(config)`, load_tts_samples, coqui
  Trainer with `TrainerArgs(restore_path=base_dir()/model.pth)`, `parse_command_line_args=
  False`, output `paths.run_dir(run_name)`; `trainer._fit()` (fit() os._exits on interrupt);
  max_steps via an `on_train_step_end` callback raising StopTraining, then a forced
  save_checkpoint and "STOP at step N"; `boinga_run.json` per session (git, env, config,
  restore, max_steps, start/end, steps, peak GPU memory, error; sessions appended); an
  interrupt saves a checkpoint first; refuses a run folder with checkpoints (replaced by
  resume in C7.3b). DECISIONS "Training run". Tests `tests/test_train_run.py` (fake Trainer, no
  torch) (this commit). No notebook cell yet (C7.4 adds smoke/resume). Results: C7 run 1
- [x] C7.3b code: `train_runner.latest_run_folder(run_name)` -> (folder, step) of the
  highest-step `checkpoint_<step>.pth` under `paths.run_dir(run_name)` (not mtime; empty
  folders and best_model_* ignored); run() resumes with `TrainerArgs(continue_path=folder)`,
  never restore_path, from the folder's own config.json, printing "RESUME from step N in
  <folder>"; max_steps counts total steps; already at max_steps -> "PASS: run NAME already at
  step N" and no training. DECISIONS "Training resume". Tests in `tests/test_train_run.py`
  (fake folders: new vs resume, highest step, empty folders, already done) (this commit).
  Not handled: a truncated last checkpoint (C7.3 plan idea) is still chosen
- [x] C7.4 code: new module `train_report.py`: `parse_log()` reads coqui's
  `trainer_0_log.txt` train-step blocks (ANSI stripped, eval blocks and tensor values
  skipped), `summarise()` (mean step_time without step 0, steps/hour, coqui's running
  average, last losses, any NaN/inf loss), `throughput(run_name)` -> reports/throughput.json
  {run, folder, steps_done, gpu, batch_size, mixed_precision, sec_per_step_mean,
  n_step_samples, steps_per_hour, peak_mem_gb, any_nan_loss, last_losses, ...} (emitted),
  "PASS: throughput recorded" or "FAIL: NaN loss"; `listen_smoke(run_name)` (first test
  sentence by the pretrained base voice and the latest checkpoint, synth_basic, to
  `paths.eval_dir()/<run>`); `boinga throughput --run NAME`. Cells C7.4.a (smoke 50 steps),
  C7.4.b (resume to 60), C7.4.c (throughput), C7.4.d (listen), each after an md()
  explanation; a and b forbid "Partial model initialization". Tests
  `tests/test_train_report.py` (fake log snippet, fake run folder, fake synth) (this commit).
  With print_step 25 the smoke (stop after 50) and resume (51-59) would log only step 25
  after warm-up, so C7.4b sets print_step 5 for the smoke run (`1cc1fb5`).
  - [x] C7.4 results: C7 run 1 (87c0470, NVIDIA RTX PRO 6000 Blackwell 95 GB, not a T4):
    smoke "STOP at step 50", checkpoint_50 on Drive, peak 3.385 GB; resume "RESUME from
    step 50" (optimizer, scheduler, scaler restored, "Model restored from step 50"), "STOP
    at step 60", peak 3.262 GB; no "Partial model initialization"; throughput 0.3406
    s/step over 10 logged steps (10,570 steps/h), fp16, batch 16, no NaN loss, loss_mel
    30.9 at step 55; C7.4.d played both clips (no listening note yet). The run trained with
    Torch seed 54321 (dropped seed override, fixed `08f29d0`); speed is for this GPU, not a
    T4 (this commit)
  - [x] C7.4 fix: `boinga clean-run` + cell C7.4.0 so the re-run starts a fresh smoke run
    (`90c4504`)
  - [x] C7.4 re-run results (f681627, RTX PRO 6000, seed 1234): C7.4.0 "PASS: cleaned run
    smoke: removed .../runs/smoke (2.00 GB ...)"; "STOP at step 50" (peak 3.39 GB),
    "RESUME from step 50", "STOP at step 60" (peak 3.262 GB); throughput 0.3418 s/step
    over 10 steps (10,533 steps/h), no NaN loss, loss_mel 30.7 at step 55; C7.2 also PASS
    again (ids 0..86 of vocab 87); reports/throughput.json replaces run 1's (this commit)
- [x] C7.4b `train_config.RUN_OVERRIDES` (per-run, value + reason) with print_step 5 for run
  `smoke` only (user, chat 2026-11-08): steps 5-45 and 55 timed (~10 samples) instead of 1;
  `reasons(run_name)` fills the DECISIONS table; cell texts updated (this commit)
- [x] C7.H hand-off: `notebooks/runs/C7_train-runner_run1.ipynb` (T4, GPU required) rebuilt,
  unchanged since `b0ec918`, `--all --check` clean, pushed; link and save name
  `colab_runs/C7_train-runner_run1.ipynb` given; no forms; duration not measured (data
  restore + roughly 10-40 min of training, estimate). Kaggle is not a drop-in fallback: the
  bootstrap needs google.colab (Drive mount, GH_TOKEN secret) (this commit)
- [x] C7.V verify run 1 (colab-verify): 11/12 PASS at 87c0470 (notebook built in `b0ec918`),
  not stale; saved as colab_runs/C7_train_runner_run1.ipynb. C7.1 FAIL ("coqui loaded the
  config but not these overrides: ['seed']") -> fix `08f29d0` (training_seed) + clean smoke
  run `90c4504`; results `b2d1884` (C7.2 + digest), `ffdfe0e` (C7.4). GPU: RTX PRO 6000, not T4
- [x] C7.V2 re-run C7 (fresh smoke run): 13/13 PASS at f681627 (notebook built in `90c4504`),
  not stale, RTX PRO 6000; C7.1 "PASS: coqui accepts config"; results `705b617` (C7.1:
  config, overrides table, digest), `40a1a25` (C7.4: throughput)
- [x] C7.5 Gate G5 — ready to train: PASS (G1 PASS, splits_summary present, no NaN loss,
  RESUME seen); run 1 verdict `f681627`, conditions met in the re-run (DECISIONS "Gate G5 —
  ready to train"; this commit)

## C8 — Full fine-tune run
Branch `chunk/C8-finetune`. One GPU notebook per run: spec `notebooks/specs/c8_finetune_run1.py`
-> `notebooks/runs/C8_finetune_run1.ipynb` (RTX PRO 6000, user's pay-as-you-go GPU). Run name
`run1`; its settings are frozen once it starts (a resume uses the run folder's config), so any
change means a new run name (`run2`). Plan basis: reports/throughput.json (0.3418 s/step,
batch 16, fp16, 3.39 GB peak; batch 32 timed slower in C8.1c,
so run1 stays at 16) and configs/splits_summary.json (6,415 train clips -> ~401
steps/epoch). Coqui keeps only the last 2 checkpoints + best, so earlier steps are kept as
"milestones" (model + config, no optimizer) for listening and promotion. Line counts are non-test
estimates, excluding the generated map/notebook and PROGRESS.
- [x] C8.1 step budget (local): `train_config.budget_steps(steps_per_hour, hours=3.5)`
  (rounded down to a save_step multiple); `RUN_BUDGETS = {"run1": 36_500}` (10,533.4 steps/h
  x 3.5 h = 36,866.9 -> 36,500; 400.9 steps/epoch at batch 16 -> ~91 epochs); `run()` / `boinga
  train` use RUN_BUDGETS[run] when --max-steps is absent ("budget: N steps (RUN_BUDGETS)").
  DECISIONS "Run 1 budget". Tests `tests/test_train_config.py`, `tests/test_train_run.py`
  (`5d4b543`). Basis was batch 16; superseded by C8.1d after the RTX PRO 6000 settings (C8.1b)
- [x] C8.1b RTX PRO 6000 settings (local): OVERRIDES batch_size 32 / eval 16, loader workers
  8 / 4, reasons no longer T4-based (max_audio_len, mixed_precision); lr kept; env_report
  records `cpu_count`. DECISIONS "RTX PRO 6000 training settings". Tests
  `tests/test_train_config.py`, `tests/test_env_report.py` (this commit)
- [x] C8.1c timing notebook: spec `notebooks/specs/c8_timing.py` ->
  `notebooks/runs/C8_timing_run1.ipynb` (GPU; the header cell records env.json with cpu_count): cells C8T.pre (restore
  data), C8T.1 (build-config smoke + validate, expects batch_size 32), C8T.2 (loader check),
  C8T.3 (clean smoke), C8T.4 (smoke 50), C8T.5 (resume 60), C8T.6 (throughput, expects
  batch_size 32), each after an md() explanation (this commit)
- [x] C8.1c.H hand-off timing notebook (link, RTX PRO 6000, save name given; `946319e`)
- [x] C8.1c.V verify (colab-verify): 12/12 PASS at 946319e (built in 946319e), not stale, RTX
  PRO 6000, cpu_count 48. Batch 32: 0.7262 s/step over 10 steps (4,957.3 steps/h), peak
  5.642 GB, no NaN (loss_mel 29.8 at step 55); vs batch 16: 0.3418 s/step. Clips/s 44.1 at
  batch 32 vs 46.8 at batch 16: no throughput gain (GPU-bound; 8 workers did not help). Drive
  root 4.5 GB free (trash not emptied). Results: throughput, env, config, overrides table
  (`0df8c10`)
- [x] C8.1d batch back to 16 / eval 8 (user, chat, after C8.1c: batch 32 is 6% slower per
  clip); loader workers stay 8 / 4; `RUN_BUDGETS["run1"]` = 36,500 confirmed (batch-16 rate,
  ~91 epochs). DECISIONS "RTX PRO 6000 training settings" update (this commit)
- [x] C8.2 ckpt_sampler (local): new module `ckpt_sampler.py`: `checkpoints(run)` (step ->
  checkpoint_<N>, else best_model_<N>; best_model.pth not sampled twice), `sample(run)` (the 5
  test_sentences of configs/vits_ctg_finetune.json, synth_basic seeded, to
  `paths.eval_dir()/<run>/step_<N>/<i>.wav` + done.json; pretrained base voice once into
  step_0_pretrained/; "step N: mean X s", "PASS: sampled K checkpoints"), `parse_eval(log)`
  (eval blocks -> last logged train step, avg loss_mel, loss_total = avg_loss_0 + avg_loss_1,
  coqui's best-model sum), `candidates(run)` -> reports/candidates_<run>.md sorted by loss_mel
  (emitted, "PASS: candidates listed"), `players(run, steps=None)` ((wav, label) pairs:
  pretrained + 4 best + last); `boinga sample` / `boinga candidates --run NAME`. Tests
  `tests/test_ckpt_sampler.py` (fake folders, log, synth) (this commit). Eval-block format
  ("--> EVAL PERFORMANCE", "| > avg_loss_mel:") not yet seen in a real log (C8.3 checks it).
  Milestone copies (C8 plan) not in this subchunk: C8.2c
- [x] C8.2c milestone copies (local; reopened after the C8 merge, user: "finish C8.2c, no
  retraining"; the earlier skip in `72df4bd` is withdrawn): new module `milestones.py`:
  `keep(run, steps=None)` copies the checkpoint minus optimizer/scaler + config of each step to
  Drive `milestones/<run>/step_<N>/` (model.pth, config.json, keep.json written last), checks
  the copy tensor by tensor, skips steps already kept; default steps = the ones the listening
  form played (`ckpt_sampler.listened_steps`: 4 best by eval loss_mel + last);
  `paths.milestones()`; `boinga keep --run NAME [--steps N ...]` (this commit)
- [x] C8.2c.H keep notebook for run1 (no training, CPU runtime): spec
  `notebooks/specs/c8_keep_run1.py` -> `notebooks/runs/C8_keep_run1.ipynb`: C8K.1 candidates,
  C8K.2 `boinga keep --run run1`, C8K.3 list `milestones/run1/`, C8K.4 load the 36500 copy in
  the Synthesizer (CPU), form C8K-notes (FOUR_FOLDERS_PRESENT, NOTES) (this commit)
- [ ] C8.2c.V verify the keep run (colab-verify): FAILED, run1's checkpoints were never on
  Drive (DEBUG_LOG 2026-11-08); replaced by C8.5
- [x] C8.5a runs on the VM disk (local): paths.runs() -> fast()/runs, `boinga train` needs 20 GB
  free there, sampled audio tied to its run folder; DEBUG_LOG + DECISIONS "Runs train on the VM
  disk" (this commit)
- [x] C8.5b keep + checked upload inside the run-1 notebook: spec
  `notebooks/specs/c8_finetune_run1.py` gains C8.keep (`boinga keep --run run1`) and C8.flush
  (`drive.flush_and_unmount` with a 20 min timeout, remount, `boinga check-milestones --run run1`) after C8.cand;
  md text for VM-disk runs; `milestones.keep` refuses without Drive space, keep.json records
  bytes, `milestones.check`, a step kept from another run folder is refused; MIN_FREE_GB 30; save_* reasons no longer say Drive; separate keep notebook
  removed (this commit)
- [x] C8.5c check milestones through the Drive API: new module `drive_api.py` (Colab sign-in,
  folder walk from My Drive, name/size/md5Checksum per file); keep.json records the model md5;
  `milestones.check(run, remote)` checks every step in the VM-side manifest kept_milestones.json
  (a step whose whole folder failed to upload fails too); duplicate names and a missing
  folder raise; the signed-in account is printed; `boinga check-milestones --drive-api`, used by C8.flush
  (this commit)
- [x] C8.5d checkpoints on Drive (user, chat: "all checkpoints should be stored in the drive"):
  paths.runs() back on Drive; save_step 10,000, save_n_checkpoints 10 (never delete),
  best-model saves off (`_no_best_models`), `saves_needed` + Drive space check per session,
  budgets round to BUDGET_ROUND 500; DECISIONS "Checkpoints on Drive every 10,000 steps"
  (supersedes C8.5a's VM disk) (this commit)
- [x] C8.5e notebook: new module `run_check.py`: `record(run)` (size + md5 of every checkpoint
  and the config, on the VM disk at paths.manifests()), `check(run, remote)` (each recorded
  file in the Drive run folder, size + md5 via the Drive API); `boinga record-run` /
  `boinga check-run [--drive-api]`; run-1 notebook: C8.rec and C8.flush right after training
  (before sampling), C8.keep removed, md text for Drive checkpoints (this commit)
- [x] C8.H3 / C8.V3 retrain run1 (same name and budget), listen again: 15/15 PASS at 853ac3a, not stale;
  4 checkpoints (10k, 20k, 30k, 36.5k) checked on Drive through the Drive API (size + md5); user's
  verdict again "good: promote BEST_STEP" 36500; DECISIONS "Run 1 retrained: checkpoints on Drive" (this commit)
- [x] C8.3 code: spec `notebooks/specs/c8_finetune_run1.py` -> `notebooks/runs/C8_finetune_run1.ipynb`
  (GPU): C8.pre (restore data), md on healthy training and disconnects, C8.tb (TensorBoard on
  `paths.run_dir('run1')`, optional), C8.train (`boinga train --run run1`, RUN_BUDGETS 36,500,
  timeout 360 min, forbids "Partial model initialization"), C8.thr (throughput of run1),
  C8.sample, C8.cand, C8.play
  (`ckpt_sampler.players`), form C8-notes (BEST_STEP, VERDICT, CLEARER_THAN_PRETRAINED,
  MORE_CHITTAGONIAN_THAN_PRETRAINED, ARTEFACTS, NOTES), an md() before each cell (this commit).
  One `boinga train` to the budget (no milestone segments): C8.sample sees only the last 2
  checkpoints + best; earlier steps need C8.2c
- [x] C8.H hand-off run 1: `notebooks/runs/C8_finetune_run1.ipynb` rebuilt, unchanged since
  `adc2f48`, `--all --check` clean, pushed; link, runtime (RTX PRO 6000; user asked "T4"),
  ~3.5 h step time + eval/saves/sampling (not measured), disconnect tips, form instructions
  and save name `colab_runs/C8_finetune_run1.ipynb` given (this commit)
- [x] C8.V verify run 1 (colab-verify): 13/13 PASS at 8005e5d (built in adc2f48), not stale;
  36,500 steps in ~2.4 h wall (0.1501 s/step), final eval loss_mel 20.603, best 20.486 at
  20450; user: BEST_STEP 36500, "good: promote BEST_STEP", clearer and more Chittagonian than
  pretrained, no artefacts (DECISIONS "Run 1 verdict", Training runs table) (this commit)
- [x] C8.4 not needed (verdict "good")
- [ ] follow-up: every training/eval text logs "Character '\n' not found in the vocabulary":
  find where the newline enters (manifest text or coqui test_sentences) and strip it
- [x] follow-up: coqui kept 15 best_model_<N>.pth in run1: `save_all_best: true` comes from the
  pretrained config (by design); on the VM disk since C8.5a, so it costs no Drive space

### C7 follow-ups (from the whole-chunk review; for C8)
- [x] `run()` / `boinga train` ignore `train_config.RUN_BUDGETS` (fixed in C8.1): without --max-steps a run
  goes up to 1000 epochs; fall back to RUN_BUDGETS[run_name] (with a test); DECISIONS "Gate G5"
  and the `epochs` reason already assume it
- [ ] configs/vits_ctg_finetune.json and the DECISIONS overrides table describe the last built
  run (now `smoke`, print_step 5): commit them from C8's own build-config and say so in
  DECISIONS
- [x] T4-based reasons (batch_size, num_loader_workers, max_audio_len, mixed_precision; spec
  docstring "(T4)") (C8.1b changed the reasons; the C7 spec stays as the historical
  record; re-measured in C8.1c): revisit for the RTX PRO 6000 (batch 16 used 3.39 of 95 GB, 48 CPUs) and
  re-measure throughput after any change
- [ ] Drive space: 6.4 GB free at the C7 re-run; empty the Drive trash and check free space
  before C8 (~1 GB per checkpoint x 2 + best)
- [ ] A resume always uses the run folder's config: setting changes need a new run name
  (write it into the C8 plan / DECISIONS)
- [ ] train_report.throughput reports the run-wide peak memory over all sessions: label it so,
  or add the last session's peak
- [ ] PROGRESS: replace the remaining "(this commit)" markers of C7.4 / C7.4b with hashes
  (`1cc1fb5`, `b0ec918`)
- [ ] C7.4.d listening note (pretrained vs step-60 smoke clip) not given yet

## C9 — Voice bank and best checkpoint
Branch `chunk/C9-synth-best`. Promotes run1 step 36500 (user's BEST_STEP, reports/notes/C8-notes.json;
DECISIONS "Run 1 retrained: checkpoints on Drive") from
runs/run1/run1-November-08-2026_05+25AM-853ac3a/checkpoint_36500.pth to `paths.best_ckpt()`, then
proves the promoted copy loads and speaks on its own. One GPU notebook (user: gpu=True): spec
`notebooks/specs/c9_synth_best.py` -> `notebooks/runs/C9_synth-best_run1.ipynb`. Line counts are
non-test estimates, excluding the generated map/notebook and PROGRESS.
- [x] C9.1 synth.VoiceBank (local): new module `synth.py`: voices "baseline"
  (`train_config.base_dir()`) and "boinga" (`paths.best_ckpt()`, when it holds model.pth +
  config.json); `available()`; `speak(voice, text, seed)` -> (sample_rate, float32 wav) with the
  Synthesizer loaded once per voice (`synth_basic.load_synth`, pinned inference scales),
  `text_frontend.prepare_text` applied, seeded; `speak_to_file(voice, text, out_wav)` ->
  seconds; ValueError for empty text, unknown or absent voice. Tests `tests/test_synth.py`
  (fake loader) (`32e2dc0`)
- [x] C9.2 code (user's C9.2 replaces the planned 2a/2b): `train_runner.promote(run, step)`
  copies checkpoint_<N>.pth of the run's latest folder (whole file, tqdm, md5 while copying)
  as `best/model.pth` + the run's config.json into flat `paths.best_ckpt()` (the folder
  VoiceBank "boinga" reads), says what it replaces, skips a re-copy of the same run folder +
  step + size, refuses without Drive space, writes provenance.json last (run, step, folder,
  md5, nearest eval losses, base model id + licence, dataset DOI + licence, speaker selection,
  char map digest, seed, inference scales, git commit, UTC date), emits
  reports/best_provenance.json, "PASS: promoted <run> step <N>"; `boinga promote --run NAME
  --step N`. Spec `notebooks/specs/c9_synth_best.py` cell C9.2 (`!boinga promote --run run1
  --step 36500`; slug `synth-best`, nbkit refuses `_`). Tests `tests/test_promote.py`
  (`1c3bd15`)
- [ ] optional: check best/ on Drive through the Drive API (size + md5 vs provenance.json),
  as run_check does for checkpoints (planned C9.2b, not in the user's C9.2)
- [x] C9.3 code: `tools/reload_check.py` (own process): VoiceBank "boinga" speaks the first
  3 rows of metadata_test.csv into `paths.eval_dir()/reload_check/` (+ reload_check.json for
  the play cell); "PASS: reload check" if every wav > 0.5 s and RMS > -50 dBFS (and finite),
  else FAIL; a stale list is removed first. `synth.reload_dir()` / `synth.reload_players()` feed
  the play cell (one call). Spec cells C9.pre (restore prepared data: the manifest is on the VM disk), C9.3,
  C9.3.play, form C9-notes (INTELLIGIBLE, NOTES). Tests `tests/test_reload_check.py`
  (`a985044`)
- [x] C9.H hand-off C9 run 1: `notebooks/runs/C9_synth-best_run1.ipynb` rebuilt, unchanged
  since `a985044`, `--all --check` clean, pushed; link, runtime (user: T4), a few minutes
  (not measured; C9.pre restores the prepared data first), form C9-notes and save name
  `colab_runs/C9_synth-best_run1.ipynb` given (`dcb26e5`)
- [x] C9.V verify C9 run 1 (colab-verify): 10/10 PASS at dcb26e5 (built in a985044), not stale,
  RTX PRO 6000 (user asked T4). C9.2: replaces nothing, 998 MB copied in 13 s, md5
  66970fd5..., provenance in reports/best_provenance.json (`42dd33e`). C9.3: 3 wavs 2.94 /
  5.23 / 2.71 s at -17.3 / -18.4 / -21.7 dBFS, "PASS: reload check (3 wavs)"; C9-notes
  INTELLIGIBLE "yes", NOTES "none" (`a95535f`)
- [x] C9.4 Gate G6 — usable checkpoint: PASS (reload check PASS, INTELLIGIBLE "yes");
  DECISIONS "Gate G6 — usable checkpoint" (`09e4cbc`)

## C10 — Blind A/B evaluation
Branch `chunk/C10-eval`. Blind comparison of the pretrained Bangla voice ("baseline") and the
fine-tuned voice ("boinga", Drive best/, run1 step 36500) on 20 test-split prompts that were never
trained on, each with its real recording (R) as the reference (user, after C10.V1 failed:
"All 20 from our test"). Raters see R, A and B. A and B are the
two voices, swapped per prompt by a seeded, balanced draw (10 prompts with boinga as A, 10 as B).
The key stays on Drive and is not printed until run 3. Three notebooks (user, after C10.1:
review the prompts before any audio is made): run 1 `notebooks/specs/c10_eval_run1.py` ->
`notebooks/runs/C10_eval_run1.ipynb` (CPU: prompts + review), run 2 `c10_eval_run2.py` (GPU:
replace rejected prompts, audio, blind pack) and run 3 `c10_eval_run3.py` (CPU: scoring).
Raters fill the sheet between runs 2 and 3. Line counts are non-test estimates, excluding the generated
map/notebook and PROGRESS.
- [x] C10.1 code (local + spec): new module `evaluate.py`: `pick_prompts(seed, exclude_ids)`:
  P01-P15 = clean clips of our test split (Drive splits.csv), has_reference true; P16-P20 =
  clean official-test clips of other speakers (not in the selection, not in splits.csv), one
  per voice cluster first, has_reference false (replaced by C10.1b: all 20 from our test
  split). Both groups: no text of a train/val clip,
  distinct text, 5-15 words first, then lowest difflib ratio of ctg_text_nfc vs bn_text
  (`baseline.bangla_similarity`), seeded ties. Writes `configs/eval_prompts.json` (list of
  id, clip_id, ctg_text verbatim, bn_text, has_reference; "replaced" lists clips a prompt
  replaced) and emits it. An existing file is kept; `--exclude` ids get the next-best unused
  clip of their group, other ids stay the same. `boinga pick-eval-prompts [--exclude
  P03,P11]` ("PASS: 20 prompts"); `evaluate.show_prompts()`. Spec
  `notebooks/specs/c10_eval_run1.py`: C10.pre, C10.1, C10.1.show, form C10.1-notes
  (REVIEWED_BY, IDS_TO_REPLACE, NOTES). Tests `tests/test_evaluate_prompts.py`. ~225 lines
  incl. the spec (over the ~150 budget; kept as one commit as the user asked). DECISIONS
  "C10 evaluation prompts" (this commit)
- [x] decided (user, chat: "Review first, hand off"): run 1 = prompts + review only (CPU);
  C10.2/C10.3 move to run 2, which first runs `boinga pick-eval-prompts --exclude <IDS>` when
  the review lists any
- [x] C10.H1 hand-off run 1: spec retitled "prompt review", gpu=False; link, CPU runtime,
  a few minutes (not measured; C10.pre restores the prepared data), form C10.1-notes and save
  name `colab_runs/C10_eval_run1.ipynb` given (this commit)
- [x] C10.V1 attempt 1 FAILED at 7829b2f (not stale): C10.pre PASS, C10.1 "need 5
  other-speaker candidates, found 2": the C4 clustering has 7,506 of 7,513 clean clips in
  cluster 1 (DEBUG_LOG 2026-11-09)
- [x] C10.1b fix (user: "All 20 from our test"): P01-P20 all from our test split,
  has_reference true; clusters and the selection no longer read; spec md updated; tests
  `tests/test_evaluate_prompts.py` (this commit)
- [x] C10.V1 verify run 1 (colab-verify, re-run after C10.1b): 9/9 PASS at b9d78c1 (built in
  b9d78c1), not stale, CPU. 20 prompts P01-P20 (5-11 words, all has_reference) in
  configs/eval_prompts.json; C10.1-notes: REVIEWED_BY "Me", IDS_TO_REPLACE "None" (no
  replacements; run 2 needs no `--exclude`), NOTES "All okay" (this commit). P08 (clip 29)
  and P10 (clip 6) are C9 reload-check clips the user has heard before
- [x] C10.2 code (local + spec): `evaluate.generate()`: per prompt VoiceBank "baseline" ->
  Drive `eval/final/A_baseline/<id>.wav`, "boinga" -> `B_boinga/` (fixed seed + inference
  scales, raw model output), and for has_reference the corpus clip through
  `prepare.transform_audio` -> `R_real/`; each wav > 0.5 s and > -50 dBFS, else ValueError;
  `final/index.csv` (id, set, path, duration_s, rms_db, text); a wav listed there with the
  same text is kept (a replaced prompt is redone); reports/eval_generation.md (count, mean /
  min / max duration, mean RMS per set) emitted; `evaluate.players(ids)`; `boinga
  eval-generate` ("PASS: generated A=20 B=20 R=20": all prompts have R since C10.1b, the
  user's "R=15" predates it). Cells C10.pre (fetch-data), C10.2, C10.2.play in new spec
  `notebooks/specs/c10_eval_run2.py` (GPU), not run 1 (run 1 is verified; audio comes after
  the review, user's "Review first"). No loudness matching here: C10.3 decides it for the
  blind pack. Tests `tests/test_evaluate_generate.py` (fake VoiceBank) (this commit)
- [x] C10.3 code (local + spec): `evaluate.make_rating_sheet(seed)`: per prompt the
  A_baseline / B_boinga wavs of final/index.csv become Drive `eval/final/blind/<id>_clip1.wav`
  / `_clip2.wav` in a seeded order with B first for half the prompts, each scaled to -23 dBFS
  RMS (peak <= 0.99); `final/key.csv` (prompt_id, clip, set, source) outside blind/, never
  printed (sha256 only), not in the repo; `blind/rating_sheet.csv` (rater, prompt_id, clip,
  text verbatim, intelligibility, naturalness, dialect_authenticity, dialect_words, comments;
  40 empty rows); `blind/INSTRUCTIONS.md` English + Bangla, emitted as
  reports/rater_instructions.md. An edited sheet is kept; a reshuffle under it is refused.
  R_real is not in the pack (not asked). `boinga eval-sheet` ("PASS: rating sheet 40 rows").
  Cell C10.3 in the run-2 spec (not run 1, as for C10.2), forbids "A_baseline"/"B_boinga" in
  its output. DECISIONS "C10 blind rating sheet". Tests `tests/test_evaluate_sheet.py`
  (this commit)
- [x] C10.H2 hand-off run 2: `notebooks/runs/C10_eval_run2.ipynb` (C10.pre, C10.2,
  C10.2.play, C10.3) unchanged since `5a679fc`, `--all --check` clean, pushed; link, GPU
  runtime (T4 or the RTX PRO 6000), duration not measured (user estimate ~15 min), rater
  instructions (share blind/ view-only, a Google Sheets copy of rating_sheet.csv per rater)
  and save name `colab_runs/C10_eval_run2.ipynb` given. The user's C10.H1 text (build run 1,
  review prompts) was already done: run 1 verified in `97e9cee` (this commit)
- [x] C10.V2 verify run 2 (colab-verify): 9/9 PASS at 05720c3 (built in 5a679fc), not
  stale. C10.2: A=20 B=20 R=20 in 1:25 (4.29 s/prompt); mean duration A_baseline 3.59 s,
  B_boinga 2.83 s, R_real 2.90 s; mean RMS -24.1 / -17.7 / -16.6 dBFS
  (reports/eval_generation.md). C10.3: 40 blind clips, key sha256 printed, no voice name in
  its output, reports/rater_instructions.md (this commit and the next)
- [x] (user) rater fills the sheet: one outside rater filled it in Google Sheets; run 3's
  C10.R.import saved it as Drive `eval/final/ratings/me.csv` (C10.4d, C10.V3)
- [x] C10.4 code (local + spec run 3): `evaluate.score()`: reads every Drive
  `eval/final/ratings/*.csv`, checks each covers exactly the 40 key rows and every score
  is a whole number 1-5 (bad cells listed by file, spreadsheet row and column; nothing is
  guessed or scored), unblinds with final/key.csv, raters anonymised R1..Rn (sheets sorted
  by file name); reports/eval.md: mean ± sd per criterion (+ overall) A vs B, per-prompt
  mean of the 4 criteria and B win rate (share of raters scoring B above A, tie 0.5), all
  comments by set, Limitations (user's list); emitted; `boinga eval-score` ("PASS: scored N
  raters"). New spec `notebooks/specs/c10_eval_run3.py` (CPU, cell C10.4), not
  c10_eval_run2.py, which already holds the executed audio + sheet run. User's design
  replaces the planned sign test / bootstrap / eval.json. Tests `tests/test_evaluate_score.py`
  (this commit)
- [x] C10.4b interactive rating (user, chat: no rater panel, "Rate the 40 blind clips
  yourself", "build a notebook where I can rate interactively"): new module `rating_app.py`:
  `load_or_start(rater)` (resume ratings/<rater>.csv, else a copy of blind/rating_sheet.csv),
  `record` (four scores 1-5 only, comment whitespace collapsed), `progress`, `next_prompt`,
  `save` (atomic), `check` (PASS only when every clip has four scores), `rate(rater)`
  (ipywidgets screen: sentence, clip1/clip2 players, 1-5 buttons, comment, Previous / Save &
  next, saved to Drive after each prompt; no voice names shown). Run-3 spec: C10.R,
  C10.R.check, then C10.4; title "rating and scoring". DECISIONS "C10 self-rating". Tests
  `tests/test_rating_app.py` (widget test needs ipywidgets: skipped locally, passed with a
  scratch install) (this commit)
- [x] C10.5 optional MCD (user: "Yes, build it"; local + spec run 4): new module
  `eval_mcd.py`: `mcd_dtw(a_wav, r_wav)` (librosa MFCC 1-13, silence trimmed, DTW with
  `librosa.sequence.dtw`, MCD in dB = sqrt(2) x Euclidean of dB-scaled MFCCs) and
  `mcd_table()` for all 20 prompts with a reference (A vs R, B vs R), replacing or
  appending "Objective check (relative only)" in reports/eval.md; `boinga eval-mcd`
  ("PASS: mcd table"). New spec `notebooks/specs/c10_eval_run4.py` (CPU, cell C10.5)
  instead of run 2 (already verified); 20 prompts, not 15 (every prompt has R). Tests
  `tests/test_eval_mcd.py` (this commit)
- [x] C10.4c ratings typed into blind/rating_sheet.csv: `rating_app.adopt_blind_sheet`, called
  by `check`, copies it to ratings/<rater>.csv when it has more rated clips (older sheet kept
  as .bak) (e18615b)
- [x] C10.4d ratings saved as a Google Sheet (run 3 failed: no blind/rating_sheet.csv, the
  folder held a Google Sheet `rating_sheet`): `rating_app.import_google_sheet` exports it
  through the Drive API; new cell C10.R.import; DEBUG_LOG entry (9e17032)
- [x] C10.H3 hand-off run 3 (CPU; save name `colab_runs/C10_eval_run3.ipynb`)
- [x] C10.V3 verify run 3 (colab-verify): 9/9 PASS at 9e17032 (built in 9e17032), not stale.
  C10.R.import copied 40 rated clips from the Google Sheet; C10.R.check "PASS: rated 40 of
  40 clips"; C10.4 "PASS: scored 1 raters". reports/eval.md: overall A 1.12 vs B 1.73
  (+0.60), B win rate 0.82 over 20 prompts. DECISIONS "C10 evaluation result" (b3e5ffa)
- [x] C10.H4 hand-off run 4 (CPU; save name `colab_runs/C10_eval_run4.ipynb`)
- [x] C10.V4 verify run 4 (colab-verify): 6/6 PASS at 2487e29 (built in 2487e29), not stale.
  C10.5 "PASS: mcd table (20 prompts)" in 0:45 (2.30 s/prompt). MCD vs R: A 143.21 ± 14.84,
  B 107.23 ± 24.91; B closer on 20 of 20 prompts (reports/eval.md) (this commit)

## C11 — Gradio demo
Branch `chunk/C11-demo`. A Gradio page that speaks typed Chittagonian text with the fine-tuned
voice (Drive best/, run1 step 36500) and, for comparison, the pretrained baseline, through
`synth.VoiceBank` (prepare_text, fixed seed and inference scales). The input check rejects
empty text and text over 300 characters, and since C11.3 characters the voice lacks (coqui
would drop them silently; C5 follow-up, DECISIONS "Gate G4"). New module
`app_gradio.py` (not `gradio.py`: it would shadow the library); gradio is imported lazily and
pinned in the `[train]` extra (the bootstrap installs it). One GPU notebook: spec `notebooks/specs/c11_demo.py` ->
`notebooks/runs/C11_demo_run1.ipynb`. Local tests use a fake VoiceBank, never coqui or the model; the Blocks test needs gradio (skipped when it is not installed). Line counts
are non-test estimates, excluding the generated map/notebook and PROGRESS. The user's C11.1
promptbook replaces the planned C11.1a/b split: one subchunk, gradio pinned in `[train]` (no
`[demo]` extra, so no install cell in C11.2), check_input = empty / > 300 characters only.
- [x] C11.1 new module `app_gradio.py`: `check_input(text)` -> message or None ("Please enter
  some Bengali text." for empty/blank, "Please keep it under 300 characters." over 300, edge
  spaces not counted); `synthesize(bank, text, voice)` (check, then VoiceBank.speak; ValueError
  messages kept, other errors "Synthesis failed (<type>)", logged); `make_handler` (DemoError ->
  gr.Error, no traceback); `build_app(bank)` -> gr.Blocks (title, proof-of-concept description,
  Textbox, Radio of bank.available() default boinga, Synthesize, Audio, gr.Examples from the
  first 5 of configs/eval_prompts.json if present, footer from settings); `launch(share)`;
  `boinga demo [--share]`; `settings.PRETRAINED_AUTHOR`; `gradio==6.31.0` in `[train]`.
  DECISIONS "C11 demo". Tests `tests/test_app_gradio.py` (fake bank; gradio tests ran with
  gradio 6.31.0 in .venv) (this commit). Not done: rejecting characters outside the voice's
  charset (coqui drops them silently; C5 follow-up)
- [x] C11.2 code (user's C11.2 promptbook replaces the planned edge_checks/report):
  `tools/demo_selfcheck.py` calls `app_gradio.make_handler` (the page's own callback) on the
  real VoiceBank, no UI: first held-out eval prompt in boinga + baseline (audio > 0.5 s,
  finite), empty and 400 characters (exact friendly messages), prompt + Latin letters and
  prompt + Bengali digits (audio or a friendly message); only gr.Error may leave the handler;
  "CASE <name>: ok <s> s|error <message>", "PASS: demo selfcheck (6 cases)". `nbkit.code(...,
  allow_interrupt=True)` (metadata key only when set, older notebooks unchanged) and
  `read_colab_run._status`: a KeyboardInterrupt in such a cell is PASS, expect/forbid checked
  on its output before the stop, no CELL-OK needed (a fallback: gradio's block_thread usually
  catches Stop itself and the cell ends normally). Spec `notebooks/specs/c11_demo.py` ->
  `notebooks/runs/C11_demo_run1.ipynb` (GPU): C11.2.check, C11.2.live, form C11-notes
  (DEMO_WORKED, NOTES). Changes to the user's cell text: the live cell launches with
  `debug=True` (without it gradio returns at once in Colab and Run all runs on into the form),
  its md says to (re)run the form after filling it (after Stop the cell usually ends
  normally, so Run all may already have run the form empty; record_notes saves what it holds); `app_gradio.make_handler` raises gr.Error with
  `print_exception=False` (gradio otherwise prints a traceback for every friendly error,
  which the live cell's "Traceback" forbid would flag). Tests `tests/test_demo_selfcheck.py`,
  test_nbkit, test_read_colab_run (this commit)
- [x] C11.H hand-off: `notebooks/runs/C11_demo_run1.ipynb` rebuilt, unchanged since `a1bc809`,
  `--all --check` clean, pushed; link, runtime (user: T4), notebook instructions, form
  C11-notes and save name `colab_runs/C11_demo_run1.ipynb` given; duration not measured (this
  commit)
- [x] C11.V verify run 1 (colab-verify): 8/8 PASS at 0280e8c (built in a1bc809), not stale,
  RTX PRO 6000 (user asked T4), Colab Python 3.13.16. C11.2.check: normal-boinga ok 3.21 s,
  normal-baseline ok 3.08 s, empty and too-long gave the friendly messages, latin-mixed ok
  5.06 s, bengali-digits ok 5.52 s, "PASS: demo selfcheck (6 cases)" in 3:00 (mostly model
  loading). C11.2.live: public gradio.live link up; gradio caught Stop ("Keyboard interruption
  in main thread... closing server.") and the cell ended with CELL-OK, so the allow_interrupt
  fallback was not needed. C11-notes: DEMO_WORKED "yes", NOTES "none" (this commit)
- [x] follow-up: Latin letters are dropped silently (coqui logs "Character 'h' not found in the
  vocabulary. Discarding it." per letter): C11.3 `app_gradio.check_input(text, charset)` lists
  characters the voice lacks after prepare_text ("The voice cannot say these characters,
  please remove them: h, e, ..."); `model_charset()` = base voice's charset from
  configs/pretrained_facts.json; `make_handler` uses it by default; all 20 eval prompts pass;
  selfcheck latin-mixed now expects that message; C11.2.check md updated, notebook rebuilt
  (run 1 not re-run with it). Bengali digits ০-৬ and ৮ are in the charset (৭, ৯ are not), so
  "১২৩" still speaks, untrained (digit rows were excluded) (this commit)

## C12 — End-to-end notebook
Branch `chunk/C12-notebook`. One notebook, `notebooks/boinga_translator.ipynb`, built from spec
`notebooks/specs/final.py`, runs the whole pipeline in 17 sections from cells the C1-C11 specs
already define, through existing `boinga` commands. A `RUN_TRAINING` flag (default False)
skips smoke test, fine-tune, samples and candidates, so the notebook uses the promoted Drive
`best/`; promote stays manual. Over budget as one commit (5+ files), so C12.1 is split a/b.
- [x] C12.1a code (tooling): `nbkit.Notebook(output=...)` (a `notebooks/...ipynb` path, not
  under specs/) and `build_notebook.output_path()`, used by the builder, the title cell's save
  name and `read_colab_run` (staleness, spec lookup, digest name); `nbkit.code(...,
  skip_unless="FLAG")` wraps src in `if not FLAG: print("SKIPPED (FLAG=False)") else: ...`
  (metadata key only when set; older notebooks byte-identical); the reader counts a skip line
  in such a cell as PASS without checking expect (forbid and CELL-OK still checked). DECISIONS
  "C12 end-to-end notebook". Tests test_nbkit, test_build_notebook, test_read_colab_run (this
  commit)
- [ ] C12.1b code: `notebooks/specs/final.py` (17 sections S1-S17, RUN_TRAINING flag cell) ->
  `notebooks/boinga_translator.ipynb`; test: 17 headings, every `!boinga <cmd>` exists in cli
- [ ] C12.H hand-off: fresh runtime, Run all with RUN_TRAINING=False
- [ ] C12.V verify (colab-verify); Gate G7 — end-to-end notebook

## C13
- [ ] C13.x (subchunks not yet defined)

## F — Fallback: MMS stack (only if the VITS fine-tune path is abandoned)
- [ ] F.x (subchunks not yet defined; trigger rules in the debug-protocol skill)
