# Progress

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
  - [ ] C4.2a results: embeddings count from the verified run (C4.V)
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
  - [ ] C4.2b results: chosen threshold and cluster minutes from the verified run (C4.V)
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
  - [ ] C4.3 results: the chosen cluster and base voice from the verified run (C4.V)
  - Moved out of C4 (C4.3 promptbook): per-split training manifests, cross-split duplicate
    text on `ctg_text_model`, text_frontend.prepare, resampling (training-data prep, C5)
- [x] C4.H hand-off: `notebooks/runs/C4_speaker_run1.ipynb` (T4) rebuilt, unchanged since
  `d856b3b`, pushed; link, form instructions and save name `colab_runs/C4_speaker_run1.ipynb`
  given; duration not measured (embedding estimate 20-40 min) (this commit)
- [ ] C4.V verify run 1 (colab-verify): every cell PASS and not stale; save
  reports/speaker_ranking.md, configs/speaker_selection.json, reports/notes/C4.3-notes.json
- [ ] C4.4 Gate G3 — speaker (DECISIONS, supersedes or confirms "Speakers"). Proposed PASS
  rule, to confirm before the run: configs/speaker_selection.json exists (so
  `ONE_CONSISTENT_VOICE` was "yes"); its `minutes_by_official_split` has train >= 60 and
  val >= 6; the official test split is kept out of training by the C5 data prep (the
  selection lists the speaker's test clips on purpose). FAIL -> the next candidate, mixed
  voices (DECISIONS "Speakers" A), or the fallback corpus, decided in the G3 entry.

## C5
- [ ] C5.x (subchunks not yet defined)

## C6
- [ ] C6.x (subchunks not yet defined)

## C7
- [ ] C7.x (subchunks not yet defined)

## C8
- [ ] C8.x (subchunks not yet defined)

## C9
- [ ] C9.x (subchunks not yet defined)

## C10
- [ ] C10.x (subchunks not yet defined)

## C11
- [ ] C11.x (subchunks not yet defined)

## C12
- [ ] C12.x (subchunks not yet defined)

## C13
- [ ] C13.x (subchunks not yet defined)

## F — Fallback: MMS stack (only if the VITS fine-tune path is abandoned)
- [ ] F.x (subchunks not yet defined; trigger rules in the debug-protocol skill)
