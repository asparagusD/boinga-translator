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
