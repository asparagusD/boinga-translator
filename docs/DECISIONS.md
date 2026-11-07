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
**Later (C3.8b):** these wavs and this listening predate the char map; baseline runs now
synthesise `text_frontend.prepare(text)`, so STD_1/STD_3 end in "." instead of "।".

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

**Verified (C3 run 3, commit eb3d3e0, `boinga audit` + `audit-summary` on the mapped
text):** clean 7,513 clips, 8.18 h (train 6,010 / 6.56 h, val 748 / 0.81 h, test 755 /
0.81 h), up from 2,881 clips / 4.10 h before the map. Excluded character 279 rows
(0.62 h); unknown to the model after the map 114 rows, all of them also excluded-character
rows (৭ ৯ / : *), so no unknown character reaches training. Gate G2 condition 1 is met.
(279 vs the 278 above: the preview counted the 8,004 rows with audio, the audit counts
all 8,009 index rows; one of the 5 rows without a wav has an excluded character.)

**Reviewed in C5.3 (2026-11-07): no new entries; the map above stays as it is.** C5 run 1
(reports/oov_report.md, commit 5d35740) checked the training speaker's 7,506 clips,
mapped with this map, against the base voice's (male) charset: 0 of 251,223 characters
unknown, 0 clips affected. The user declined (chat, C5.V1) eight optional inference-only
entries that the corpus never uses: ‐ U+2010, ‑ U+2011 and − U+2212 -> "-"; ― U+2015 -> —;
and dropping U+200B, U+FEFF, U+00AD, U+2060. The user also chose (chat, C5.3) to keep
the file and its list format instead of an empty {map, drop} file: an empty file would
undo the decisions above, and any change to the parsed map changes map_digest and makes
audit.csv stale. The approved mappings are therefore the 18 entries listed above, in
order:
- "।" (U+0964) and "৷" (U+09F7) -> "."
- ৰ (U+09F0) and ৱ (U+09F1) -> র (U+09B0)
- ' " “ ” ( ) … removed
- – (U+2013) -> — (U+2014)
- | -> "."; ; -> ","; ZWJ (U+200D) removed
- য+় -> য় (U+09DF), ড+় -> ড় (U+09DC), ঢ+় -> ঢ় (U+09DD)

The row exclusions ("/", ":", "*", digits) are unchanged. text_frontend now exposes the
steps as apply_char_map() (the replacements only), prepare_text() (=
apply_char_map(normalise(text)), whitespace collapsed again; prepare() is the same) and
count_unknown(); behaviour and map digest (151ac177e2) are unchanged.

## 2026-11-07 — Speakers: fine-tune the single-speaker model on the mixed voices
**Context:** Gate G2 condition 2. The corpus has no speaker field and the C3.5 listening
heard "many" speakers. Options offered (chat): A mixed voices (all clean clips into the
single-speaker VITS); B the largest consistent speaker cluster from a speaker encoder;
C a multi-speaker VITS with cluster ids (architecture change from the single-speaker
checkpoint). Measuring clusters first (C3.9a) was offered and declined.
**Decision:** A, decided by the user in chat: fine-tune the pretrained single-speaker VITS
on the clean clips regardless of speaker, with no speaker filtering or speaker ids. The
training pool is the clean official train split only (6,010 clips, 6.56 h); the clean
val split (748 clips, 0.81 h) is for validation; the clean test split (755 clips,
0.81 h, incl. baseline clips 386, 29, 625) is held out (G2 condition 3). The 7,513 clean
clips / 8.18 h is the total over all three splits, not the training pool.
**Alternatives:** B (consistent voice, but an unknown and possibly thin subset plus a new
dependency); C (most work, needs speaker embeddings in a model trained without them).
Measure-first (one Colab run) was declined.
**Consequences:**
- The fine-tuned voice may average speakers: timbre or prosody can drift between
  utterances. C4's smoke-test listening checks this explicitly.
- If C4 hears an unstable voice, the fallback is B: embed clips with SpeechBrain ECAPA
  (speechbrain/spkrec-ecapa-voxceleb, the user's choice of encoder; Apache-2.0, licence
  and Python 3.13 support to be verified before install) and retrain on the largest
  cluster. This needs a new DECISIONS entry.
- Gate G2 condition 2 is met; condition 3 (official test split out of training, incl.
  clips 386, 29, 625) is a C4 data-split rule.

## 2026-11-07 — Speaker encoder: resemblyzer 0.1.4, SpeechBrain ECAPA as fallback
**Context:** C4.2a needs one speaker embedding per clean clip to find a consistent training
speaker (the corpus has no speaker ids). The C4.2a promptbook names resemblyzer first and
speechbrain/spkrec-ecapa-voxceleb as the fallback. Checked from the PyPI files: Resemblyzer
0.1.4 (2023-10-12) is Apache-2.0 (classifier and LICENSE file), ships its weights
(pretrained.pt, 256-dim d-vectors), loads them with torch.load (a plain state dict) and
resamples with keyword arguments (librosa >= 0.10 safe). Its dependency webrtcvad 2.0.10 is
an sdist (built with gcc on Colab) whose module does `import pkg_resources`, which newer
setuptools may no longer provide. speechbrain 1.1.2 is Apache-2.0 and needs torch >= 2.7.1.
**Decision:** resemblyzer==0.1.4 and setuptools<81 in the [train] extra (installed by BOOT.3).
`speaker_select.load_encoder()` uses resemblyzer (VoiceEncoder on cuda + preprocess_wav);
on ImportError it falls back to SpeechBrain ECAPA, pinned as speechbrain==1.1.2 in a
separate `speaker-fallback` extra, so the training install never pulls it in. Embeddings
are cached per clip on Drive (`speaker_emb/<clip_id>.npy`) with an `encoder.txt` marker; a
cache from one encoder is never extended with the other, and .npy files without a marker
are refused. The loop stops as soon as more than 1% of clean clips have failed (e.g. CUDA
errors), not after the whole pass.
**Alternatives:** ECAPA only (the earlier plan; better speaker-verification accuracy, but a
model download and a torch/torchaudio constraint); pin speechbrain in [train] too (could
upgrade torch under coqui-tts); webrtcvad-wheels (resemblyzer still requires the
`webrtcvad` distribution).
**Consequences:** If the C4 run prints "WARN: resemblyzer unavailable", the fallback needs
`pip install -e '.[speaker-fallback]'` first, otherwise the cell FAILs fast; speechbrain
may upgrade torch/torchaudio on that VM, so the fallback runs in its own runtime, never in
the one that trains. The cluster
step (C4.2b) must read the marker and treat 256-dim (resemblyzer) and 192-dim (ECAPA)
vectors alike. Model weights stay in the pip package / Drive, never in the repo.

**Verified (C4 run 1, commit ad690fd, T4, Python 3.13.16, torch 2.14.1):** resemblyzer
imported (no fallback); 7,513 of 7,513 clean clips embedded, 0 failed. webrtcvad's
`import pkg_resources` only warned ("slated for removal"), so the setuptools<81 pin
still matters.

## 2026-11-07 — Speaker clustering: average-linkage cosine, threshold chosen by silhouette
**Context:** C4.2b groups the clean clips' speaker embeddings (no speaker ids) into voices
and ranks them, so C4.3 can train on one consistent voice. The right distance threshold
for this corpus and encoder is unknown before the run.
**Decision:** (C4.2b promptbook) scikit-learn AgglomerativeClustering, metric "cosine",
linkage "average", on L2-normalised embeddings, at thresholds 0.20, 0.25, ..., 0.50. Per
threshold: cluster count, clean minutes of the largest cluster, silhouette (cosine; NaN
when every clip is alone or all are in one cluster). Chosen: the best silhouette among
thresholds whose largest cluster has >= 30 clean minutes; if none does, the threshold
whose largest cluster has the most minutes; ties go to the lower threshold. Clusters are
ranked 1..K by clean minutes over all three splits (ties: first clip id); labels go to
Drive `speaker_clusters.csv`, the table and the top 20 clusters (minutes per split, 5
seeded clip ids each) to reports/speaker_ranking.md. The tree is built once per run
(sklearn `memory` cache); 7,500 random 256-dim vectors took 9 s locally.
**Alternatives:** one fixed threshold (the earlier plan; no evidence for a value); HDBSCAN
or k-means (the promptbook fixes agglomerative); ranking by train minutes only (the
ranking table shows train/val/test minutes, so C4.3 and Gate G3 can apply a train rule).
**Consequences:** "Largest cluster minutes" counts val and test clips too; Gate G3 must
check the train minutes of the chosen cluster. A high silhouette at a high threshold can
still merge similar voices: listening to the top clusters (C4.3) decides, not the score.

## 2026-11-07 — Speaker choice: saved from the listening form, base voice = speaker gender
**Context:** C4.3. The clusters (C4.2b) are ranked by minutes, not by how consistent they
sound; only listening can tell one voice from two similar voices merged. The pretrained
voice to fine-tune (female or male) was still open (C2 follow-up; G1 quality "same").
**Decision:** (C4.3 promptbook) Form C4.3-notes, after listening to 10 seeded clips of
each of the top 3 clusters, saves the choice only when CHOSEN_ID and GENDER are filled
and ONE_CONSISTENT_VOICE is "yes"; otherwise it prints "NOTES-MISSING choice incomplete"
and saves nothing, so the cell's expected "PASS: selection saved" is missing and the
run reader flags it. The base voice is the pretrained voice matching the chosen
speaker's gender. The selection is the cluster's clean clips over all official splits:
Drive `speaker_selection.csv` (clip_id, split_official, duration_s) and
configs/speaker_selection.json with its sha256, so later chunks can check they read the
same file. `kind` is "cluster" (C4.1, speaker labels, was skipped: the corpus has none).
`boinga save-speaker --id N --base-voice female|male` does the same from a shell.
**Alternatives:** "mixed voices" as a form option (DECISIONS "Speakers" A; not offered by
the promptbook: if no candidate is one voice, Gate G3 (C4.4) decides the fallback);
choosing the base voice by listening comparison (no evidence G1 preferred either).
**Consequences:** The selection includes the cluster's val and test clips; the training
prep (C5) must keep the official test split out of training (G2 condition 3) and Gate G3
must judge the train minutes. If the audit charset voice (female) differs from the base
voice, re-run the audit with `--voice male` before training (C2 follow-up).

## 2026-11-07 — Training speaker: mixed voices (cluster 1), base voice male
**Context:** C4 run 1 (ad690fd) embedded all 7,513 clean clips (resemblyzer) and the
C4.2b rule chose threshold 0.50 (best silhouette, 0.196, since every threshold's largest
cluster had >= 30 min). That merged 7,506 clips into cluster 1; clusters 2 and 3 had 5
and 2 clips. In form C4.3-notes the user picked cluster 2 (CHOSEN_ID 2, one consistent
voice, male, "some noise", notes "none"): 0.18 clean minutes, under the 10-minute G3 floor.
Smaller thresholds split the corpus into 71-1,464 clusters (largest 52-128 min), unheard.
**Decision:** The user chose in chat (C4.V): train on mixed voices, i.e. cluster 1, with
the male pretrained voice as the base (kept from the form). C4 run 2 (fcb64e5, CPU) saved
it with `boinga save-speaker --id 1 --base-voice male`: configs/speaker_selection.json,
kind "cluster", id 1, 7,506 clips, 490.73 clean min (train 393.72, val 48.25, test 48.76),
csv_sha256 e170000e...592e of Drive speaker_selection.csv. This confirms DECISIONS
"Speakers" (A, mixed voices); the 7 clean clips of clusters 2-3 stay out.
**Alternatives:** keep cluster 2 (11 s of one voice: G3 FAIL, Chunk F); re-cluster with a
cap on the largest cluster's share and listen again (one more run; not chosen).
**Consequences:**
- The fine-tuned voice may drift between speakers; C7-C9 listening checks it, and the
  re-clustering option stays open if it does (PROGRESS follow-up).
- The C3 audit checked text against the female charset: C5 re-runs `boinga audit --voice
  male` before building training lists (C2 follow-up).
- The selection lists the official val and test clips too; C5 keeps the official test
  split (incl. clips 386, 29, 625) out of training (G2 condition 3).
- The run-1 selection (cluster 2) is superseded: run 2 overwrote the Drive copy.

## 2026-11-07 — Gate G3 — speaker data
**Context:** End of C4. Rule (C4.V promptbook), on the clean minutes of
configs/speaker_selection.json: >= 30 -> PASS; 10-30 -> PASS WITH RISK; < 10 -> FAIL,
do C5 and C6, then Chunk F instead of C7-C9 (MMS fine-tuning is reported to work with
80-150 clips). Selection (DECISIONS "Training speaker", d386d13): cluster 1, mixed voices,
7,506 clips, 490.73 clean min (train 393.72, val 48.25, test 48.76), base voice male.
**Decision:** G3: PASS (490.73 >= 30). The VITS fine-tune path (C5 onwards) continues.
**Alternatives:** the form's single-voice pick (cluster 2, 0.18 min) would have been
G3: FAIL; the user chose mixed voices instead (DECISIONS "Training speaker").
**Consequences:** The PASS is on minutes, not on voice consistency: the selection is many
speakers (DECISIONS "Speakers" A). Training minutes come from the train split only
(393.72 min); the official test split stays out of training (C5). The C4 plan's own
G3 draft (train >= 60, val >= 6 min) is also met. If C7-C9 hear an unstable voice, the
fallback is re-clustering with a cap on the largest cluster's share (PROGRESS follow-up).

## 2026-11-07 — Gate G4 — text frontend
**Context:** End of C5. Rule (C5.V2 promptbook), on the unknown-character rate of the
real coqui tokenizer over the training speaker's text: < 1% -> PASS; otherwise FAIL with
(a) widen configs/char_map.json (user approval; redo C5.3, C5.4 and run 2) or (b) extend
the model's text embedding as C7.0. Evidence: C5 run 2 (90b60c3, reports/
tokenizer_check.md, e0beff4): coqui-tts 0.28.0 TTSTokenizer from the male voice's
config.json on text_frontend.prepare_text() of all 7,506 selected clips (0 excluded by
char_map, char map 151ac177e2): 0 of 297,218 characters unknown (0.000000, spaces
included), 0 round-trip failures. C5 run 1 (reports/oov_report.md, 5d35740) agrees: 0 of
251,223 non-space characters outside the male charset.
**Decision:** G4: PASS (0.00% < 1%). The text front-end (normalise -> apply_char_map,
DECISIONS "Character map") is the one used for training and inference; no embedding
change. The VITS fine-tune path continues with C6.
**Alternatives:** (a) and (b) above were not needed. The eight optional inference-only
map entries (dash variants, invisible format characters) were declined in C5.V1.
**Consequences:** Training (C6 onwards) must feed prepare_text() output to coqui and
must not re-normalise it (it is not NFC-stable). Any later char_map change changes
map_digest, makes audit.csv stale and needs this check (C5 run 2) again. Inference text
outside the corpus (typed prompts) can still hold characters the voice lacks: coqui drops
them with only a warning, so the demo should check prompts with count_unknown().

## 2026-11-07 — Audio preprocessing: mono, soxr_hq to the model rate, 100 ms pad, -1 dBFS peak
**Context:** C6.1. The corpus is 16 kHz mono PCM_16 (audit_summary.md); the base voice
(male) was trained at 22,050 Hz (pretrained_facts.json), and the model's sample rate and
STFT/mel settings must not change (vits-finetune). dialect-data-rules asks for mono,
resampling to the checkpoint's rate, edge-only silence trimming that keeps >= 100 ms, and
peak-safe normalisation.
**Decision:** `prepare.transform_audio()`: average the channels to mono; resample with
librosa `res_type="soxr_hq"` only when the rate differs; trim leading/trailing silence
with librosa.effects.trim at top_db 40 (the audit's threshold, data_audit.TRIM_TOP_DB) and
keep 100 ms on each side, clamped at the clip edges; scale so the peak is -1 dBFS; write
16-bit PCM atomically (`.tmp` then rename). The target rate comes from
`prepare.target_sample_rate()` (base voice in configs/speaker_selection.json ->
pretrained_facts.json), never a constant. Empty, non-finite or all-zero clips raise
ValueError (counted as exclusions by C6.4). Stats per clip: in/out rate and duration,
trimmed seconds, gain_db, upsampled.
**Alternatives:** Loudness (RMS/LUFS) normalisation was in the C6 plan draft; the C6.1
promptbook chose peak normalisation (simpler, no new dependency, cannot clip). Other
resamplers (kaiser_best, scipy) were not needed: soxr ships with librosa >= 0.10. Trimming
before resampling was not chosen: trimming at the output rate makes the 100 ms pad exact
in output samples.
**Consequences:** Every training clip is upsampled 16 -> 22.05 kHz, so it has no content
above 8 kHz; the model will learn to generate band-limited audio (DECISIONS "Corpus
layout"). Peak normalisation leaves loudness varying with each clip's crest factor across
the mixed voices; if listening (C6.4 form) finds big level jumps, revisit with RMS/LUFS.
trim's centred frames can keep up to about half a frame (~46 ms at 22,050 Hz) of extra
silence on each side; the 100 ms pad is a minimum.

## 2026-11-07 — Training splits: dedupe by transcript, official test, seeded 5% val
**Context:** C6.2. dialect-data-rules: no transcript in more than one split; the test split
is fixed once and never trained on (Gate G2 condition 3, incl. baseline clips 386, 29,
625). The C4 selection (7,506 clips) spans all three official splits.
**Decision:** `prepare.make_splits(index_df, selection_ids, seed=settings.SEED)` keeps the
selected clips, drops clips whose ctg_text_nfc repeats (keeping the longest; ties: lowest
clip id) and counts them; test = the official test clips when >= 20 remain, else a seeded
5% sample (min 20) of the unique texts; val = a seeded 5% (min 20) of what is left after
test, official val clips included; train = the rest. It asserts no transcript is in two
splits. `save_splits()` writes Drive `splits.csv` (`paths.splits_csv()`; clip_id, split,
duration_s, LF line ends) and `configs/splits_summary.json` (counts, minutes, csv_sha256,
seed, test_source, dropped_duplicate_text), and refuses to overwrite a splits.csv that
fixes a different test set.
**Alternatives:** Keeping the official val split as val (48 min, ~10%) was not chosen: the
C6.2 promptbook asks for a 5% seeded val, which leaves more audio for training. Dedupe on
the model text (prepare_text) instead of ctg_text_nfc was not chosen (promptbook); see
consequences.
**Consequences:** Duplicates are matched on ctg_text_nfc, so two different transcripts
that char_map turns into the same model text could still land in two splits; the audit's
duplicate_text exclusion already removed every ctg_text_nfc duplicate from the clean set,
so this is the only residual leak path (follow-up: count it in the C6.4 summary). Changing
the seed or the selection later needs a deliberate delete of splits.csv, and a new test
set makes earlier evaluations incomparable.
