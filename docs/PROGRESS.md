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
- [ ] C7 notebook runs `boinga fetch-data` and `boinga prepare` first (wavs and manifests
  live on the VM's fast disk); check it prints "restored ..." and "0 clip(s) to do"
- [ ] Restore depends on byte-identical splits.csv: a different pandas/numpy on the GPU VM
  could change its bytes (e.g. float format of duration_s), miss the tar, recompute and
  write a second ~1.2 GB tar; warn when another prepared-*.tar exists but none matches
- [ ] C7 config: set min/max audio length on trimmed durations (no post-trim 1-15 s
  exclusion in C6; trimming can push a clip under the minimum)
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
  commit). Results: C7 run 1
- [ ] C7.2 `train_runner.check_loader(cfg_path, n_batches=3)` + `boinga check-loader`:
  load_tts_samples + Vits.init_from_config + model.get_data_loader (train, no model weights
  needed beyond init); per batch: shapes, max token id < num_chars, no NaN in wavs/specs, text
  and audio length bounds, sec/batch; sample counts after the length filter vs
  configs/splits_summary.json. Cell C7.2. Tests with an injected fake loader. ~70 lines
- [ ] C7.3a `train_runner.run(run_name, max_steps=None)` for a new run: TrainerArgs
  `restore_path` = cached pretrained model.pth, `continue_path` empty; output under
  `paths.run_dir(run_name)` on Drive; restore check logs how many `disc.`/`dec.`/text-encoder
  keys loaded (G1: 111 disc keys expected; none -> log, longer warm-up, consider F);
  `max_steps` stop via a trainer step-end callback, mechanism confirmed first against the
  installed coqui-tts-trainer source (it must leave a checkpoint behind; DECISIONS entry).
  Factories injected so tests use a fake Trainer. ~110 lines
- [ ] C7.3b auto-resume: `latest_run_folder(run_dir)` finds the coqui run folder with the
  newest readable `checkpoint_*.pth` (0-byte/unreadable last file -> previous one); found ->
  `continue_path` and never `restore_path`; none -> new run (C7.3a). Refuses both set. Tests
  on tmp folders with fake checkpoint files. ~60 lines
- [ ] C7.4 throughput + `boinga train --run NAME [--max-steps N]`: a step-end callback
  records step time and `torch.cuda.max_memory_allocated`, NaN/inf in any loss -> stop + FAIL
  (fp16 -> fp32 is then a logged config change, not automatic); `throughput_report()` ->
  `reports/throughput.json` (gpu, batch, fp16, steps, median sec/step after 5 warm-up steps,
  peak GB, eval sec, hours per 10k steps), emitted. Cells: C7.0 fetch-data, C7.0p prepare
  (expects "restored" + "0 clip(s) to do", C6 follow-up), C7.0m fetch-pretrained male,
  C7.4.smoke (run "smoke", 50 steps), C7.4.resume (same run, to step 60: log shows
  continue_path, first step > 50), C7.4.report, C7.4.play (test sentences of the smoke
  checkpoint: noise expected, checks the synth path only). ~100 lines
- [ ] C7.H hand-off: notebook (T4) built, `--all --check` clean, pushed; link, save name
  `colab_runs/C7_train-runner_run1.ipynb`; duration not measured (smoke ~15-30 min estimate)
- [ ] C7.V verify (colab-verify): every cell PASS, not stale; reports/throughput.json and
  configs/vits_ctg_finetune.json committed as results
- [ ] C7.5 Gate G5 — training runner: PASS if 50 smoke steps + resumed steps finish with
  finite losses, resume continues from the saved step (no second restore), peak memory fits
  the T4 with margin and sec/step is measured; record hours for the C8 run from throughput.
  FAIL -> debug-protocol (batch/fp32/length limits), Chunk F only by its trigger rules

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
