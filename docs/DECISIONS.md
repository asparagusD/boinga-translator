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
