# Colab run C10 run 3 — rating and scoring

- File: `C10_eval_run3.ipynb` · spec `notebooks/specs/c10_eval_run3.py`
- Run: commit `9e17032` · branch `chunk/C10-eval` · GPU `none` · time `2026-11-09T01:24:39+00:00`
- Staleness: OK: the run used 9e17032, notebook built in 9e17032
- Result: 9/9 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C10.R.import | Import scores from the Google Sheet | PASS |  |
| C10.R | Rate the blind clips | PASS |  |
| C10.R.check | Check the ratings are complete | PASS |  |
| C10.4 | Score ratings | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
9e17032 fix(eval): import ratings saved as a Google Sheet
PASS: /content/boinga-translator on chunk/C10-eval
```

### BOOT.3 — install pinned dependencies (coqui-tts==0.28.0 via the [train] extra)

```text
torch      2.14.1+cpu
coqui-tts  0.28.0
numpy      2.4.6
PASS: dependencies installed
```

### BOOT.4 — create project folders (the HDR cell records the environment)

```text
PASS: 13 project folders ready under /content/drive/MyDrive/boinga-translator
```

### HDR — Run header

```text
python             3.13.18
platform           Linux-6.6.159+-x86_64-with-glibc2.39
cpu_count          2
is_colab           True
torch              2.14.1+cpu
torch_cuda         None
torch_error        None
gpu_name           none
gpu_mem_gb         None
coqui_tts          0.28.0
numpy              2.4.6
git_commit         9e17032
disk_free_gb_root  3.5
disk_free_gb_fast  107.5
utc_time           2026-11-09T01:24:39+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 2,
  "disk_free_gb_fast": 107.5,
  "disk_free_gb_root": 3.5,
  "git_commit": "9e17032",
  "gpu_mem_gb": null,
  "gpu_name": "none",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.18",
  "torch": "2.14.1+cpu",
  "torch_cuda": null,
  "torch_error": null,
  "utc_time": "2026-11-09T01:24:39+00:00"
}
=== END reports/env.json ===
PASS: env recorded
WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run needs one
RUN-HEADER chunk=C10 run=3 branch=chunk/C10-eval commit=9e17032 gpu=none utc=2026-11-09T01:24:39+00:00
{'python': '3.13.18',
 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39',
 'cpu_count': 2,
 'is_colab': True,
 'torch': '2.14.1+cpu',
 'torch_cuda': None,
 'torch_error': None,
 'gpu_name': 'none',
 'gpu_mem_gb': None,
 'coqui_tts': '0.28.0',
 'numpy': '2.4.6',
 'git_commit': '9e17032',
 'disk_free_gb_root': 3.5,
 'disk_free_gb_fast': 107.5,
 'utc_time': '2026-11-09T01:24:39+00:00'}
```

### C10.R.import — Import scores from the Google Sheet

```text
copied 40 rated clips from Google Sheet 'rating_sheet' to /content/drive/MyDrive/boinga-translator/eval/final/ratings/me.csv
CELL-OK C10.R.import
```

### C10.R — Rate the blind clips

```text
CELL-OK C10.R
```

### C10.R.check — Check the ratings are complete

```text
PASS: rated 40 of 40 clips (/content/drive/MyDrive/boinga-translator/eval/final/ratings/me.csv)
CELL-OK C10.R.check
```

### C10.4 — Score ratings

```text
=== BEGIN reports/eval.md ===
# Blind listening test

1 raters (R1), 20 prompts, 40 ratings. A_baseline = pretrained Bangla voice, B_boinga = fine-tuned voice (Drive best/). Scores 1-5; mean ± sd over all ratings.

| criterion | A_baseline | B_boinga | B − A |
|---|---|---|---|
| intelligibility | 1.40 ± 0.50 | 1.90 ± 0.79 | +0.50 |
| naturalness | 1.10 ± 0.31 | 1.80 ± 0.70 | +0.70 |
| dialect_authenticity | 1.00 ± 0.00 | 1.60 ± 0.68 | +0.60 |
| dialect_words | 1.00 ± 0.00 | 1.60 ± 0.75 | +0.60 |
| overall | 1.12 ± 0.17 | 1.73 ± 0.60 | +0.60 |

## Per prompt

Mean of the 4 criteria; B win rate = share of raters scoring B above A (tie = 0.5).

| prompt | A_baseline | B_boinga | B win rate |
|---|---|---|---|
| P01 | 1.50 | 1.00 | 0.00 |
| P02 | 1.50 | 1.00 | 0.00 |
| P03 | 1.00 | 1.25 | 1.00 |
| P04 | 1.25 | 1.50 | 1.00 |
| P05 | 1.25 | 1.00 | 0.00 |
| P06 | 1.00 | 2.50 | 1.00 |
| P07 | 1.00 | 1.50 | 1.00 |
| P08 | 1.25 | 2.50 | 1.00 |
| P09 | 1.00 | 2.75 | 1.00 |
| P10 | 1.00 | 1.75 | 1.00 |
| P11 | 1.00 | 2.25 | 1.00 |
| P12 | 1.25 | 1.50 | 1.00 |
| P13 | 1.25 | 2.00 | 1.00 |
| P14 | 1.25 | 1.75 | 1.00 |
| P15 | 1.00 | 1.50 | 1.00 |
| P16 | 1.00 | 1.00 | 0.50 |
| P17 | 1.00 | 1.50 | 1.00 |
| P18 | 1.00 | 1.25 | 1.00 |
| P19 | 1.00 | 2.25 | 1.00 |
| P20 | 1.00 | 2.75 | 1.00 |

B_boinga win rate over prompts: 0.82

## Comments

### A_baseline

(none)

### B_boinga

(none)

## Limitations

- Informal panel of 1 rater(s), not a formal MOS test; no significance testing, so small
  differences between A and B may be noise.
- One target voice (fine-tuned on the corpus's mixed clean clips): this measures unseen
  sentences in a known voice, not new speakers.
- The base voice (tts_models/bn/custom/vits-male) was trained on Indian Bengali data; its
  accent can colour both A and B.
- The model reads graphemes (Bengali script, NFC + char map), not phonemes, so spelling
  variation in the dialect text reaches the model directly.
- ASR word error rate (WER) is not used: standard-Bangla ASR penalises correct Chittagonian forms.
=== END reports/eval.md ===
PASS: scored 1 raters
CELL-OK C10.4
```
