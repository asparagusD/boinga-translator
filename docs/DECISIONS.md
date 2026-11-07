# Decisions

One entry per decision, newest last. Config overrides, character mappings
(configs/char_map.json) and version changes must be logged here.

Template:

```
## YYYY-MM-DD — <decision>
**Context:** <what forced a choice>
**Decision:** <what we chose>
**Alternatives:** <what else was considered and why not>
**Consequences:** <what this commits us to, risks, follow-ups>
```

## 2026-11-07 — Stack: fine-tune Bangla VITS on the Chittagonian corpus with coqui-tts on free Colab
**Context:** We want a single-speaker Chittagonian TTS proof of concept with
a $0 budget. Chittagonian is written in Bengali script, and we have no
Chittagonian TTS model to start from.
**Decision:** Fine-tune the pretrained Bangla VITS model
`tts_models/bn/custom/vits-{male,female}` (Apache 2.0) using
`coqui-tts==0.28.0` (Idiap fork, import path `TTS`). Train on "A
Synchronized Chittagonian Speech Corpus" (Mendeley n9xttk45df v1,
DOI 10.17632/n9xttk45df.1, CC BY 4.0) on the free Colab T4 GPU. Local
development is CPU-only, so coqui-tts lives only in the `[train]` extra.
**Alternatives:** Training VITS from scratch needs far more data and
compute than we have. The MMS stack is kept as fallback chunk F. Paid GPUs
and APIs are out of budget; the one exception allowed by CLAUDE.md is Colab
pay-as-you-go for an A100.
**Consequences:** coqui-tts stays pinned, and any version change is logged
here. We keep the checkpoint's sample rate, STFT/mel parameters and
character set, so Chittagonian characters outside that set need
configs/char_map.json. Training must be resumable from Drive, because free
Colab sessions drop. Every README and model card cites the dataset DOI and
its authors.

## 2026-11-07 — Naming: boinga-translator / boinga_translator
**Context:** The repo, distribution, import package and docs need one
consistent name.
**Decision:** The repo, project and distribution name is
`boinga-translator`; the import package is `boinga_translator`; the console
script is `boinga`.
**Alternatives:** none considered.
**Consequences:** The code-reviewer subagent treats any other project name
as BLOCKING. Module names must not shadow libraries (no `tts.py`,
`trainer.py`, `datasets.py`, `gradio.py`, `audio.py` or `config.py`), and no
local `TTS` module.

## 2026-11-07 — Pinned facts: model ids, release URLs, dataset file, hash and seed
**Context:** Downloads and evaluation must be reproducible and checkable, and
every later module needs the same ids, URLs and hash. They now live in
`src/boinga_translator/settings.py`.
**Decision:** Pin these values:
- coqui-tts `0.28.0`; pretrained ids `tts_models/bn/custom/vits-female` and
  `tts_models/bn/custom/vits-male` (Apache-2.0); release zips
  `.../releases/download/v0.13.3_models/tts_models--bn--custom--vits_{female,male}.zip`.
  Source: the Coqui model registry in the Idiap coqui-ai-TTS repo.
- Dataset "A Synchronized Chittagonian Speech Corpus for Dialect ASR and
  Speech-to-Speech Translation" by Dipto Saha, Md Ariful Islam and
  Md. Milon Islam (DOI 10.17632/n9xttk45df.1, CC-BY-4.0). One zip file:
  843,601,272 bytes, sha256
  `6ae668d22492f725cc3846e809788555d48b1b2c096af66cd8b521b71c0ccb0d`.
  Source: the Mendeley Data files API (file id
  `4b27c3de-f162-4028-a8aa-b7db252ece1e`).
- `SEED = 1234`. Inference scales: noise 0.667, noise_w 0.8, length 1.0 (the
  usual VITS inference defaults), fixed for all evaluation audio.

**Alternatives:** Leaving the hash unset and trusting the first download was
rejected: a pinned hash catches a corrupted or changed upload at download
time.
**Consequences:** The download step must check the size and sha256 and fail
loudly on a mismatch. These values were supplied with the C1.2 task and have
not yet been checked against a real download; the first download (Colab)
confirms them. If Mendeley or the release changes, update `settings.py` and
add a new entry here.

## 2026-11-07 — Colab run reader: a STALE run fails verification
**Context:** `tools/read_colab_run.py` checks an executed notebook from
`colab_runs/`. Its staleness check compares the commit the run used
(RUN-HEADER) with the last commit that built the notebook. The C1.8 task
said "exit 0 if all spec cells PASS"; a run of an older notebook build can
pass every cell without verifying the current one.
**Decision:** The reader exits 1 when the run is STALE, even if every cell
passed. UNKNOWN (no RUN-HEADER, or the run's commit is not in the local
clone) does not change the exit code; it is printed and written to the digest.
**Alternatives:** Cell status only (the original task wording); a separate
exit code 2 for STALE. Rejected: one non-zero code keeps scripts and hooks
simple, and the digest says why.
**Consequences:** After rebuilding a notebook, re-run it on Colab before
committing its results. The colab-verify skill (C1.9) must say this.

## 2026-11-07 — Colab work runs as whole generated notebooks
**Context:** Pasting Colab output cell by cell was slow and error-prone
(truncated pastes, cells run on the wrong branch, no record of what ran).
**Decision:** Each chunk run on Colab is one notebook generated from a spec
(`notebooks/specs/cN_<slug>.py`, `tools/nbkit.py`, `tools/build_notebook.py`)
with the bootstrap cells and branch baked in. The user runs it with "Run all"
and downloads the executed copy into `colab_runs/` (git-ignored).
`tools/read_colab_run.py` checks every cell against the spec's expect/forbid
patterns, writes a digest to `reports/colab_runs/`, and saves report blocks.
Human judgement goes in form cells. The `colab-handoff` and `colab-verify`
skills describe the loop.
**Alternatives:** Keep pasting cell output (rejected: the reason for C1N);
commit executed notebooks (rejected: large audio/base64 outputs and the risk
of leaking secrets).
**Consequences:** Cells must be idempotent, so "Run all" can always be
repeated after a fix. Executed notebooks never reach git; only the digest and
reports do. tools/save_pasted_reports.py stays as a fallback.


## 2026-11-07 — Colab cells stop the run themselves on failure
**Context:** In a generated notebook, a cell that only prints `FAIL: ...`
lets Colab's "Run all" carry on into later cells, which then work on bad
state. The reader marks such a cell FAIL afterwards, but the run has already
wasted time (and GPU quota). `cell_ok()` cannot see the cell's output, so it
cannot stop the run.
**Decision:** A cell that detects a failure stops the run: it calls
`colab_runtime.fail(msg)` (prints `FAIL: msg`, raises `SystemExit(1)`) or lets
library code raise. `colab_runtime.expect_files()` stops by default
(`stop=False` returns a bool instead). The reader still checks every cell's
output against the spec, so a forgotten `fail()` is caught at verification (FAIL);
a stopped cell shows as ERROR (`SystemExit: 1`), with its `FAIL:` line in the digest.
**Alternatives:** Capture each cell's stdout from `run_header()` and make
`cell_ok()` raise on a `FAIL:` line (rejected for now: depends on IPython
output internals that cannot be tested locally); reader-only checking
(rejected: the run keeps going after a failure).
**Consequences:** colab-handoff tells spec writers to use `fail()`. Not yet
seen on Colab: the first failing cell in a real run will confirm that
`SystemExit` stops "Run all".

## 2026-11-07 — Run notebooks stay tied to their chunk branch
**Context:** A built notebook's BOOT.2 checks out the branch its spec names
(e.g. `chunk/C1N-notebook-pipeline`). After the chunk merges, `main` carries
that notebook, but the code it ran lives on the chunk branch.
**Decision:** Chunk branches are kept on origin after merging (never deleted),
so every run notebook can be re-opened and re-run on exactly the code it was
built for. Run notebooks on `main` are a record; new Colab work gets a new
spec (or a bumped `run=`) on the new chunk branch. Only bootstrap-style specs
use `branch="main"`.
**Alternatives:** Retarget every spec to `main` in a last commit before the
merge (rejected: rebuilding makes the verified run STALE); delete run
notebooks at merge (rejected: loses the link between digest and notebook).
**Consequences:** Don't delete `chunk/*` branches on origin. A STALE result
for an old run after a later rebuild is expected; its committed digest stays
the record of what was verified.


## 2026-11-07 — Baseline prompts: standard Bangla now, Chittagonian from the corpus after C3
**Context:** C2 synthesises a baseline with the pretrained Bangla voices before
any fine-tuning. The corpus (and so verbatim Chittagonian text) only arrives in
C3, and dialect-data-rules forbid writing or "standardising" Chittagonian text.
**Decision:** `configs/baseline_prompts.json` holds three everyday standard-Bangla
sentences written for this project (STD_1-3, NFC, one question) and three
placeholders CTG_1-3 = null. `boinga baseline` skips null prompts. The user chose
this over supplying Chittagonian lines by hand.
**Alternatives:** User-written Chittagonian lines (unverified against the corpus
spelling); delaying the baseline until after C3 (delays Gate G1).
**Consequences:** Gate G1 judges the pretrained voices on standard Bangla only.
After C3, CTG_1-3 get verbatim held-out corpus transcripts and the baseline is
re-run (C2 follow-up in PROGRESS.md).

## 2026-11-07 — Pretrained download: both Bangla VITS voices cached on Drive
**Context:** C2.1 fetches `tts_models/bn/custom/vits-{female,male}` once and
caches them under `pretrained/<voice>/` on Drive (C2 run 1, commit `68ab521`).
**Decision:** Keep the coqui ModelManager download as the source: it succeeded
for both voices (no WARN, so the pinned release-zip fallback was not used).
Sizes from the C2.1 cell: female `model.pth` 997.77 MB + `config.json` 0.01 MB;
male `model.pth` 997.77 MB + `config.json` 0.01 MB (about 2 GB of Drive).
**Alternatives:** Downloading the release zips directly (kept as the fallback).
**Consequences:** The checkpoints are large because they carry optimizer and
grad-scaler state (C2.2). The cached files' sha256 were not recorded in this
run; pinning them stays a C2 follow-up.

## 2026-11-07 — Baseline smoke listening: both pretrained voices intelligible
**Context:** C2.4 synthesised STD_1-3 (standard Bangla) with both pretrained
voices at the pinned seed and inference scales on a T4 (C2 run 1, `68ab521`).
Every wav passed the smoke rule (> 0.3 s, > -50 dBFS): female 4.15 / 4.41 /
4.04 s at -22.9 / -20.9 / -19.8 dBFS; male 5.13 / 4.48 / 5.46 s at
-20.2 / -18.1 / -19.4 dBFS.
**Decision:** Record the user's listening form (reports/notes/C2.4-notes.json):
female intelligible "yes", male intelligible "yes", artefacts "none",
preferred voice quality "same". Neither voice is ruled out by quality.
**Alternatives:** none; this is a measurement.
**Consequences:** The fine-tune voice can be chosen by fit to the corpus
speaker (C3/C4) rather than by baseline quality. The sentence-final danda
"।" is the config's EOS token and not in characters/punctuations, so the
baseline WARNed on it for STD_1 and STD_3; how coqui's tokenizer treats it
is checked when the text front-end is built (C3). Durations may include
coqui's per-sentence padding, so they are not used as throughput figures.

## 2026-11-07 — Gate G1 — pretrained model
**Context:** Before any fine-tuning, Gate G1 checks that the pretrained Bangla
VITS checkpoints are usable for a Chittagonian fine-tune. Evidence: C2 run 1
(`68ab521`, 10/10 cells PASS), configs/pretrained_facts.json,
reports/pretrained_inspect.md, reports/notes/C2.4-notes.json.
**Findings:**
- Sample rate: female 22050 Hz, male 22050 Hz (fft 1024, hop 256, win 1024,
  80 mels, fmin 0, fmax null). Our data must be resampled to 22050 Hz.
- Text input: graphemes (`TTS.tts.utils.text.characters.Graphemes`),
  `use_phonemes` false, `text_cleaner` null (no cleaner at all), `add_blank`
  true, `enable_eos_bos_chars` false; pad `<PAD>`, blank `<BLNK>`, bos `<BOS>`,
  eos `।`. Single speaker (`num_speakers` 0, no speaker embedding).
- Discriminator: present in both checkpoints (111 `disc.` keys, 46,747,132
  params of 83.05 M total). A normal warm-up applies; this does not make
  Chunk F likelier.
- Standard Bangla intelligible (user's form): female "yes", male "yes",
  artefacts "none", voice quality "same".
**Surprises vs the expectation** (a character-based single-speaker Apache-2.0
VITS trained on IITM Bangla):
- Matches: character (grapheme) input, single speaker, VITS. The licence comes
  from the coqui registry entry; the training corpus cannot be confirmed from
  the config (dataset paths are `bn_tts/{female,male}/mono` on the author's
  machine, no corpus name).
- No text cleaner: the model sees raw text, so our NFC is the only
  normalisation, as the data rules want.
- The EOS symbol `।` is not in characters/punctuations; tokenizer handling
  to be checked in C3 (the baseline was intelligible either way).
- Both charsets hold precomposed য়/ড়/ঢ় (U+09DF/09DC/09DD) and the separate
  nukta U+09BC. These three letters are Unicode composition exclusions, so NFC
  always decomposes them (e.g. য় -> U+09AF U+09BC); NFC text never contains the
  precomposed symbols. Which form the model learned is unverified, so
  precomposed vs decomposed is a C3 decision (char_map + DECISIONS entry).
  The C2 baseline prompts fed the decomposed form and were still intelligible.
- The two vocabularies differ: male adds U+200D (ZWJ), one extra embedding
  row (192 params). Each voice must keep its own character set.
- Checkpoints are about 1 GB each because they keep optimizer and grad-scaler
  state; long training histories (female step 811000, male step 910000).
- Configs carry the author's absolute paths (`output_path`, dataset paths,
  `restore_path`); our fine-tune config must override them.
**Decision:** G1: PASS. Both voices are usable; the fine-tune voice is chosen
by fit to the corpus speaker in C3/C4.
**Alternatives:** G1: FAIL -> Chunk F (MMS) was not needed.
**Consequences:** C3 can build the data pipeline at 22050 Hz with grapheme
input against the chosen voice's charset; any characters outside it go to
configs/char_map.json with a DECISIONS entry. The fine-tune config must copy
the full `characters` block from the pretrained config.json (including flags
such as `is_unique`/`is_sorted` that fix the vocabulary order and so the
embedding rows), not rebuild it from configs/pretrained_facts.json.
(Nukta wording corrected at the C2 merge review.)

## 2026-11-07 — Dataset download: pinned Mendeley zip, verified, unpacked once on Drive
**Context:** C3.1 must put the pinned corpus (Mendeley n9xttk45df v1, CC BY 4.0) on
Colab reproducibly. Run 1's first attempt failed: data.mendeley.com (Cloudflare)
answers 403 to urllib's default `Python-urllib/3.x` User-Agent (DEBUG_LOG 2026-11-07).
**Decision:** `boinga fetch-data` (data_download.fetch) streams the pinned URL with
`User-Agent: boinga-translator/<version> (research dataset download)` (no personal
data) via Mendeley's 302 to S3, resuming a `.part` file with HTTP Range; checks size
843,601,272 B and the pinned sha256; unzips once to Drive data/unzipped; copies to
/content/boinga-fast/data per VM. Markers hold the pinned sha256; a redone step redoes
the later ones. Verified on Colab (C3 run 1, commit f06163c, CPU): download 844 MB in
23 s (35.2 MB/s), `DONE verify`, `PASS: dataset ready at /content/boinga-fast/data,
8005 files` = 8,004 .wav + 1 .xlsx (matches the 8,004 clips on the dataset page).
**Alternatives:** Download with a browser and upload to Drive (manual, not
reproducible); extract the Drive zip straight to /content each session (skips the
Drive tree; not measured, revisit if the per-VM copy is slow).
**Consequences:** Drive holds the zip and the unpacked tree (~1.7 GB of the 11.1 GB
free before the run). Per-step timings for unzip/copy were not captured (the cell
prints no durations); add them if the copy becomes a bottleneck. If a Colab IP is ever
refused, the IP, not the agent, is the next suspect.

## 2026-11-07 — Corpus layout: folders mark the splits, one xlsx holds the text, no ids or speakers
**Context:** C3.3 (corpus.load_index) must be written against the real layout.
reports/corpus_layout.txt (C3 run 1, f06163c) records it; only the clip count, the
transcript/translation columns and the 6,403/801/800 splits were known before.
**Decision:** The layout is:
- Audio: `Dataset/Speech/{training,validation,test}/<integer>.wav`, 6,403 / 801 / 800
  files (8,004). Sampled: 16,000 Hz, mono, PCM_16 (1.50 s, 3.76 s, 4.67 s). Stems seen:
  `990`, `1001`, `4596`.
- Text: one workbook `Dataset/Chittagonian_translated.xlsx`, one sheet
  `Chittagonian_translated`, 8,009 data rows, 3 columns.
- Clip id: no id or file-name column. The row <-> wav link is presumably the row
  number = wav stem, but unverified (base 0 or 1, and 8,009 rows vs 8,004 wavs).
- Chittagonian text: column `Chittagonian Dialect` (training text, verbatim).
- Standard Bangla text: column `Standard Bangla` (metadata only, never trained on).
- Also present: `Translated English` (metadata only; not on the dataset page).
- Split: no column; marked by the folder (`training` / `validation` / `test`); counts
  match the dataset page.
- Speaker: no speaker field (and no gender/region/age field).
**Alternatives:** none; this records the data as published.
**Consequences:**
- C3.3 must establish the row <-> wav mapping before any audit: per-folder stem range
  and gaps, which 5 rows have no wav (empty/duplicate/header-like rows?), 0- vs 1-based,
  plus a sanity check (clip duration vs transcript length) and a listening spot-check.
- Unknown speaker count ("native speakers", plural): the single-speaker goal may need
  a speaker-consistent subset; C3.5's listening form asks for it.
- 16 kHz audio vs the model's 22,050 Hz: resampling up adds no content above 8 kHz;
  the fine-tuned voice will be band-limited. Accepted for a proof of concept; noted for
  Gate G2.
- The first rows read like translated encyclopedic prose about Chittagonian, so the
  text is read speech of written sentences.

**Verified (C3 run 2, commit f4c7ab2, `boinga index`):** 8,009 index rows: train 6,403,
val 801, test 800, no split 5 (the rows without a wav); 8,004 with audio; speaker ids:
none ("no speaker field").

## 2026-11-07 — Corpus row <-> wav mapping: wav stem = Excel row number
**Context:** The corpus has no clip-id column (DECISIONS "Corpus layout"); C3.3 needs
the link between Dataset/Chittagonian_translated.xlsx rows and Dataset/Speech wavs.
**Decision:** wav `<n>.wav` belongs to Excel row n of the first sheet (header = row 1);
`corpus.load_index` keys on it (`clip_id` = n). Evidence (probe of the pinned zip over
HTTP Range: file listing, the xlsx and 40 random wavs, nothing committed):
- wav stems run 2..8010, all unique across the three split folders; the xlsx has
  8,009 data rows = Excel rows 2..8010.
- Exactly 5 stems are absent: 1934, 3289, 3352, 4062, 4223 (8,009 - 5 = 8,004 wavs).
- Duration vs `Chittagonian Dialect` length over 40 random wavs, Pearson r:
  stem = row -2: 0.13, -1: 0.17, **+0: 0.87**, +1: -0.02, +2: 0.33.
- The 5 rows without audio are all long sentences (221-256 characters).
- Local `boinga index` on the real xlsx + empty placeholder files at the real wav paths: train 6403, val 801,
  test 800, (none) 5; `PASS: index has 8009 rows (8004 with audio)`.
**Alternatives:** 0-based data-row index (r = 0.17), sequential order (no gaps
explained); pairing by transcript length alone (not unique).
**Consequences:** The index has 8,009 rows; the 5 without audio have no split and
`wav_exists` False and are excluded from training. `load_index` raises if a wav has no
row, a stem appears in two splits, a wav name is not a number, a split folder is
missing, or a row with a wav has no dialect text. The C3.5 listening check is the
final confirmation (clip audio vs its transcript).

## 2026-11-07 — Dialect baseline prompts: 3 held-out test-split lines, least like Bangla
**Context:** DECISIONS "Baseline prompts" planned verbatim held-out Chittagonian lines
for CTG_1-3 once the corpus was in (C3.6).
**Decision:** `boinga pick-baseline-ctg` takes clips that pass `audit_summary.clean_mask`
and have `split_official == "test"`, prefers 5-12 words, and ranks by
`difflib` similarity of `ctg_text_nfc` to the normalised `Standard Bangla` text, lowest
first (ties: seeded shuffle, `settings.SEED`). CTG_1-3 hold the raw `ctg_text` verbatim
(not NFC; outer/inner spaces kept); `_sources` in configs/baseline_prompts.json records
their clip ids. Local preview on the real text (no audit): 478 test clips with 5-12
words, similarity median 0.77; the lowest were 0.19, 0.31, 0.35.
**Alternatives:** random test lines (often near-standard Bangla, a weak dialect probe);
user-written lines (unverified against the corpus); NFC text (closer to the training
input, but the prompt rule asks for the transcript as published).
**Consequences:** The `_sources` clips are evaluation prompts: C4 must keep the official
test split out of training (dialect-data-rules: the test split is never trained on).
Re-running the pick after the audit changes may choose other lines; `_sources` shows
which ones a baseline used.

## 2026-11-07 — Audit listening: clean audio, many speakers, speech heard as standard Bangla
**Context:** C3 run 2 (f4c7ab2) form C3.5-notes, after listening to 20 seeded random clean
clips (C3.5.play) labelled with their Chittagonian transcript and Standard Bangla
translation; reports/audit_summary.md gives the numbers.
**Decision:** Record the listening result as given: TRANSCRIPTS_MATCH_AUDIO "mostly",
SOUNDS_CHITTAGONIAN "no, standard Bangla", NOISE_LEVEL "clean", SPEAKERS_HEARD "many",
NOTES "none". Audit numbers: 8,004 decodable clips, 9.04 h (16 kHz mono PCM_16); clean
2,881 clips, 4.10 h (train 3.30, val 0.37, test 0.42); excluded 5,128 of 8,009 rows
(64%): oov 5,015 (4,711 of them only because of the danda), speech rate 167, too long
29, duplicate text 28, too short 8, clipped 7, undecodable/no wav 5; no speaker field.
**Alternatives:** none (an observation record, not a choice).
**Consequences:**
- Open tension for Gate G2: the dialect transcripts "mostly" match audio that sounds like
  standard Bangla. Either the speakers read the Chittagonian text with standard
  pronunciation, or the audio follows the Standard Bangla column; the form cannot tell
  these apart, and the duration/text-length mapping check (r = 0.87) cannot either,
  because both columns have similar lengths.
- "Many" speakers and no speaker ids: a single-speaker fine-tune cannot select one
  speaker from this corpus's metadata.
- The oov rule mostly measures the danda; a char_map decision on "।" changes clean hours
  substantially (up to ~4,700 more clips).

## 2026-11-07 — Baseline on dialect text: pretrained voices read Chittagonian "partly"
**Context:** C3 run 2 (f4c7ab2) cell C3.6 picked CTG_1-3 from clean test-split clips
(DECISIONS "Dialect baseline prompts") and both pretrained voices read all 6 prompts on
the T4; form C3.6-notes recorded the judgement after C3.6.play.
**Decision:** Record the result: picks CTG_1 clip 386 "অ্যাক্কানা হুইত পাইজ্জম না, ছার? ",
CTG_2 clip 29 (a 10-word line about Roman script), CTG_3 clip 625 "আগর ডইল্লা দেইন ন ফার";
all 12 wavs passed (2.30-6.56 s, -24.6 to -18.1 dBFS); HANDLES_DIALECT "partly",
NOTES "none". Both voices warned that "।" (STD_1, STD_3, CTG_2) is outside
characters/punctuations, as expected for the eos symbol.
**Alternatives:** none (an observation record).
**Consequences:** The pretrained Bangla voices give a usable but imperfect starting point
for dialect text; fine-tuning has something to improve. Clips 386, 29 and 625 are
evaluation prompts and must stay out of training (official test split).

## 2026-11-07 — Gate G2 — data
**Context:** Gate G2 decides whether "A Synchronized Chittagonian Speech Corpus"
(n9xttk45df v1) is fit for the Chittagonian fine-tune. Rule (C3.V2 promptbook): PASS if
(1) undecodable < 1% and (2) transcripts are paired with the audio and Chittagonian,
judged from the C3.5 listening form. Evidence: reports/audit_summary.md,
reports/notes/C3.5-notes.json, DECISIONS "Audit listening".
- Undecodable: 5 of 8,009 rows (0.06%); all 8,004 wavs decode, the 5 rows have no wav.
  Holds.
- Paired: TRANSCRIPTS_MATCH_AUDIO "mostly"; row <-> wav mapping r = 0.87 (DECISIONS
  "Corpus row <-> wav mapping"). Holds.
- Chittagonian: SOUNDS_CHITTAGONIAN "no, standard Bangla". Does not hold.
- Total clean: 2,881 clips, 4.10 h (of 9.04 h decodable).
- Speaker ids: none (no speaker field; SPEAKERS_HEARD "many").
- Excluded: 5,128 of 8,009 rows (64.0%). Top 3 reasons: oov 5,015 (4,711 only because of
  the danda "।", the model's eos symbol), speech rate 167, too long 29.
**Decision:** G2: FAIL (condition 2: the audio was heard as standard Bangla, not
Chittagonian). Recommended fallback: "Bangla Regional Dialects Speech Dataset" (Mendeley
777wsgjgtm v1, Chittagong subset, CC BY 4.0); its transcripts may be standard Bangla,
so it gets the same C3 download, layout, audit and listening checks before it is used.
**Alternatives:**
- G2: PASS, reading "standard Bangla" as speakers' accent while they read dialect text
  (transcripts "mostly" match): would keep this corpus. Not chosen: the rule is decided
  by the form, and the dialect is the project's point.
- Fine-tune anyway on the 4.10 h (or ~8 h after a danda char_map): a voice that reads
  dialect spelling with standard pronunciation would not be a Chittagonian voice.
**Consequences:**
- C4 does not start on this corpus. Next: evaluate the fallback dataset with the C3
  pipeline (download, layout, index, audit, listening) and decide G2 again.
- Reusable regardless: data_download, the layout inspector, the audit and summary
  (clean_mask), the dialect baseline picker; corpus.load_index is specific to this layout.
- If the user revises the C3.5 judgement (e.g. dialect words read with a standard
  accent count as Chittagonian), G2 can be re-decided in a new entry; the open
  danda char_map and the many-speakers issue then remain to be solved before C4.

## 2026-11-07 — Gate G2 — data (re-decided): conditional PASS
**Context:** The user re-decided G2 after the FAIL entry above. Asked what the C3.5
answer "no, standard Bangla" meant, the user clarified in chat: the 20 clips contain
the dialect words of the Chittagonian transcripts, read with a standard-Bangla
pronunciation/intonation ("dialect words, standard accent"); the audio does not follow
the Standard Bangla column. This matches TRANSCRIPTS_MATCH_AUDIO "mostly" and the
row <-> wav mapping (r = 0.87). Condition (1) still holds (undecodable 0.06%).
**Decision:** G2: PASS with conditions; this supersedes "Gate G2 — data" (FAIL). The
corpus stays. Conditions to meet before C4 (training) starts:
1. Danda: decide how "।" (U+0964, the model's eos symbol, in 6,254 rows) is handled, in
   configs/char_map.json with a DECISIONS entry; 4,711 clips are excluded for it alone,
   so clean hours (4.10 h now) depend on it. The other out-of-charset characters
   (ৰ U+09F0, ৷ U+09F7, digits 7/9, quotes, brackets) are decided in the same entry.
2. Speakers: many speakers and no speaker ids. Decide between training the
   single-speaker model on mixed voices and selecting a consistent subset (e.g. by
   clustering speaker embeddings), with a DECISIONS entry.
3. Test split: the official test split (incl. baseline clips 386, 29, 625) stays out of
   training.
**Alternatives:** Keep FAIL and evaluate the fallback "Bangla Regional Dialects Speech
Dataset" (777wsgjgtm v1): not chosen, the dialect is in the words of this corpus and the
fallback's transcripts may be standard Bangla. Unconditional PASS: not chosen, the danda
and speaker issues decide how much usable single-voice data exists.
**Consequences:** The fine-tuned voice will say Chittagonian words with a largely
standard-Bangla accent (what the corpus contains); the proof of concept targets dialect
vocabulary and spelling, not regional accent. The fallback stays the plan B if the
speaker condition leaves too little consistent data.

## 2026-11-07 — Character map: danda to ".", typo fixes, silent punctuation, precomposed nukta
**Context:** Gate G2 condition 1. coqui-tts 0.28.0 source (TTS/tts/utils/text/
characters.py `_create_vocab`, tokenizer.py `encode`): the vocab is pad, eos, bos,
blank, the characters and the punctuations, so the eos symbol "।" is a single-character
token that text can hit; a character outside the vocab is dropped with only a log
warning, so text and audio would silently disagree. The pretrained config has no text
cleaner. Corpus counts (rows): danda 6,254; ৰ U+09F0 59; ৱ U+09F1 1; ৷ U+09F7 4; quotes
' 40, " 46, “ 2; brackets 45/44; en dash 41; slash 19; colon 7; * 1; … 1; any digit 255
(৭ and ৯ are not in the model). Nukta letters in the raw text: precomposed য় 5,825
(3,816 rows), ড় 1,043, ঢ় 2; bare nukta 500 (360 rows); NFC decomposes all of them, and
both forms are in both voices' charsets.
**Decision:** configs/char_map.json, applied in order to ctg_text_nfc, then whitespace
collapsed and stripped again (user choices in chat):
- "।" and its lookalike "৷" -> "." (the user chose the ASCII full stop over keeping the
  eos token).
- Typos: ৰ -> র, ৱ -> র (contexts: আগ্ৰহ, গ্ৰাউন্ড, ৱেষ্টুরেন্টে).
- Silent marks removed: ' " “ ” ( ) …; en dash – -> em dash —.
- For inference prompts only (not in the corpus): | -> ".", ; -> ",", ZWJ removed (it
  only shapes a conjunct; the female voice lacks it).
- Kept on purpose (both voices know them): ‘ ’ (’ marks elision, e.g. দু’রাত, 37 rows)
  and ZWNJ (11 rows).
- Nukta: after NFC, য+় -> য় (U+09DF), ড+় -> ড় (U+09DC), ঢ+় -> ঢ় (U+09DD), the form of
  about 93% of the raw corpus.
- Rows containing "/", ":", "*" or any digit (ASCII or Bengali) are excluded from
  training, not rewritten (278 rows). The colon rows mix punctuation (যেঅন:), a typed
  visarga (দু:খ, বস্তুত:) and a time (২:৩০), so no single mapping fits (review finding;
  a first draft mapped ":" to a comma, which would have split দু:খ).
Preview on the real transcripts (scratchpad): 7,133 of 8,004 rows change; after the map
and the exclusions, 0 rows contain a character outside either voice's charset.
**Alternatives:** keep "।" as the eos token (no mapping): it is in the vocab, but the user
chose ".". NFC-decomposed nukta (no mapping): fewer map entries, but differs from 93% of
the published text. Exclude every row with any listed punctuation (about 230 rows) or
map "/" to a comma (keeps 19 rows, risks text/audio mismatch where "বা" was spoken).
Spell digits out: rewrites transcripts, against dialect-data-rules.
**Consequences:** Text front-end (C3.8b) applies this map for training and inference
alike; the audit then judges mapped text and replaces the oov_ratio > 0.02 rule with
"any character still unknown after mapping" plus the exclusion list, and is re-run on
Colab (C3.8c). The mapped text is not NFC-stable (precomposed nukta): nothing may
normalise it again after the map, and C3.8b tests that applying it twice changes
nothing. The model's "." token replaces its "।" token at sentence ends; inference
prompts get the same map. Whether the precomposed nukta rows of the checkpoint are the
trained ones can be checked from embedding norms in C4.
