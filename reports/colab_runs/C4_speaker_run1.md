# Colab run C4 run 1 — C4 — training speaker

- File: `C4_speaker_run1.ipynb` · spec `notebooks/specs/c4_speaker.py`
- Run: commit `ad690fd` · branch `chunk/C4-speaker` · GPU `Tesla_T4` · time `2026-11-07T05:27:28+00:00`
- Staleness: OK: the run used ad690fd, notebook built in d856b3b
- Result: 10/10 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C4.0 | Ensure dataset on this VM | PASS |  |
| C4.2a | Speaker embeddings (20–40 min, resumable) | PASS |  |
| C4.2b | Cluster voices | PASS |  |
| C4.3.play | Listen: top 3 candidates, 10 clips each | PASS |  |
| C4.3-notes | Your choice — training speaker | PASS |  |

## Form values

### C4.3-notes — Your choice — training speaker

- `CHOSEN_ID` = "2"
- `ONE_CONSISTENT_VOICE` = "yes"
- `GENDER` = "male"
- `AUDIO_QUALITY` = "some noise"
- `NOTES` = "none"

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
ad690fd chore(notebook): build C4 run 1 notebook
PASS: /content/boinga-translator on chunk/C4-speaker
```

### BOOT.3 — install pinned dependencies (coqui-tts==0.28.0 via the [train] extra)

```text
torch      2.14.1+cu130
coqui-tts  0.28.0
numpy      2.4.6
PASS: dependencies installed
```

### BOOT.4 — create project folders (the HDR cell records the environment)

```text
PASS: 12 project folders ready under /content/drive/MyDrive/boinga-translator
```

### HDR — Run header

```text
python             3.13.16
platform           Linux-6.6.159+-x86_64-with-glibc2.39
is_colab           True
torch              2.14.1+cu130
torch_cuda         13.0
torch_error        None
gpu_name           Tesla T4
gpu_mem_gb         14.6
coqui_tts          0.28.0
numpy              2.4.6
git_commit         ad690fd
disk_free_gb_root  9.4
disk_free_gb_fast  112.5
utc_time           2026-11-07T05:27:28+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 9.4,
  "git_commit": "ad690fd",
  "gpu_mem_gb": 14.6,
  "gpu_name": "Tesla T4",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T05:27:28+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C4 run=1 branch=chunk/C4-speaker commit=ad690fd gpu=Tesla_T4 utc=2026-11-07T05:27:28+00:00
{'python': '3.13.16',
 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39',
 'is_colab': True,
 'torch': '2.14.1+cu130',
 'torch_cuda': '13.0',
 'torch_error': None,
 'gpu_name': 'Tesla T4',
 'gpu_mem_gb': 14.6,
 'coqui_tts': '0.28.0',
 'numpy': '2.4.6',
 'git_commit': 'ad690fd',
 'disk_free_gb_root': 9.4,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-07T05:27:28+00:00'}
```

### C4.0 — Ensure dataset on this VM

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
CELL-OK C4.0
```

### C4.2a — Speaker embeddings (20–40 min, resumable)

```text
/usr/local/lib/python3.13/dist-packages/webrtcvad.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
embedding 7513 of 7513 clean clips with resemblyzer-0.1.4
500/7513 embedded
1000/7513 embedded
1500/7513 embedded
2000/7513 embedded
2500/7513 embedded
3000/7513 embedded
3500/7513 embedded
4000/7513 embedded
4500/7513 embedded
5000/7513 embedded
5500/7513 embedded
6000/7513 embedded
6500/7513 embedded
7000/7513 embedded
7500/7513 embedded
PASS: 7513 embeddings (resemblyzer-0.1.4, 0 failed)
CELL-OK C4.2a
```

### C4.2b — Cluster voices

```text
=== BEGIN reports/speaker_ranking.md ===
# Speaker clusters

Encoder: resemblyzer-0.1.4. Clean clips with an embedding: 7513. Agglomerative clustering, cosine distance, average linkage. Chosen threshold 0.50: best silhouette among thresholds whose largest cluster has >= 30 clean minutes, else the one with the most.

| threshold | clusters | largest cluster min | silhouette | chosen |
|---|---|---|---|---|
| 0.20 | 1464 | 51.64 | 0.042 |  |
| 0.25 | 568 | 65.70 | 0.049 |  |
| 0.30 | 187 | 72.08 | 0.125 |  |
| 0.35 | 71 | 128.35 | 0.108 |  |
| 0.40 | 20 | 395.17 | 0.011 |  |
| 0.45 | 8 | 452.77 | 0.109 |  |
| 0.50 | 3 | 490.73 | 0.196 | yes |

## Clusters ranked by clean minutes (top 20)

| rank | clips | clean min | train min | val min | test min | clip ids |
|---|---|---|---|---|---|---|
| 1 | 7506 | 490.73 | 393.72 | 48.25 | 48.76 | 4351, 5466, 6654, 7478, 941 |
| 2 | 5 | 0.18 | 0.12 | 0.00 | 0.06 | 5048, 5101, 5604, 6659, 6671 |
| 3 | 2 | 0.11 | 0.00 | 0.11 | 0.00 | 6490, 6504 |
=== END reports/speaker_ranking.md ===
PASS: ranked 3 clusters (threshold 0.50)
CELL-OK C4.2b
```

### C4.3.play — Listen: top 3 candidates, 10 clips each

```text
=== candidate 1: 490.73 clean min, 7506 clips ===
AUDIO 7478 [train] CTG: পুলিশ হুইন্নী ঘুষ খায়।:
<IPython.lib.display.Audio object>
AUDIO 5466 [val] CTG: আঁই চোখ বন্ধ গরি গম্ভীর ভঙ্গিত কিছুক্ষণ বিড় বিড় গরি গাড়ির ডেসবোর্ডত দুয়ো ফুঁ দি দিলাম।:
<IPython.lib.display.Audio object>
AUDIO 6654 [train] CTG: না দিয়ি জয়াব ।:
<IPython.lib.display.Audio object>
AUDIO 941 [train] CTG: আঁআর দাঁত ভাঙ্গি গিয়ি।:
<IPython.lib.display.Audio object>
AUDIO 4351 [train] CTG: আঁরা গাড়ি লৈ বাইর হোই গিয়ি ।:
<IPython.lib.display.Audio object>
AUDIO 5969 [train] CTG: এহন ঘাবড়ি গিয়ি।:
<IPython.lib.display.Audio object>
AUDIO 3771 [val] CTG: এ সমত ইতারা পোইরত পড়ি পানিত ডুবি যায়।:
<IPython.lib.display.Audio object>
AUDIO 3597 [train] CTG: যা খাইতু ইচ্ছে গরে খা।:
<IPython.lib.display.Audio object>
AUDIO 1490 [train] CTG: মঙ্গলবার বিয়ালে সাতক্ষীরা সদর হাসপাতালত লইবার পথত ইবা মারা গিয়ি।:
<IPython.lib.display.Audio object>
AUDIO 1894 [train] CTG: আঁই ন খাই ।:
<IPython.lib.display.Audio object>
=== candidate 2: 0.18 clean min, 5 clips ===
AUDIO 6671 [test] CTG: তুঁই বিদায় হ।:
<IPython.lib.display.Audio object>
AUDIO 5048 [train] CTG: আঁই হাইসসিলাম।:
<IPython.lib.display.Audio object>
AUDIO 5101 [train] CTG: কি হোইয়ী ?:
<IPython.lib.display.Audio object>
AUDIO 5604 [test] CTG: কোঁইয়া ভাজি।:
<IPython.lib.display.Audio object>
AUDIO 6659 [train] CTG: খাইতে খাইতে কই ।:
<IPython.lib.display.Audio object>
=== candidate 3: 0.11 clean min, 2 clips ===
AUDIO 6490 [val] CTG: ইতারা বোথ ডর পাইয়ী।:
<IPython.lib.display.Audio object>
AUDIO 6504 [val] CTG: রেনু হাত-পা এলিয়ে ঘুমার।:
<IPython.lib.display.Audio object>
CELL-OK C4.3.play
```

### C4.3-notes — Your choice — training speaker

```text
=== BEGIN reports/notes/C4.3-notes.json ===
{
  "AUDIO_QUALITY": "some noise",
  "CHOSEN_ID": "2",
  "GENDER": "male",
  "NOTES": "none",
  "ONE_CONSISTENT_VOICE": "yes"
}
=== END reports/notes/C4.3-notes.json ===
=== BEGIN configs/speaker_selection.json ===
{
  "kind": "cluster",
  "id": 2,
  "n_clips": 5,
  "clean_minutes": 0.18,
  "minutes_by_official_split": {
    "train": 0.12,
    "val": 0.0,
    "test": 0.06
  },
  "csv_sha256": "c32de346c80bd45e2e43a4d0f5b0fe599eadf3633c1386190bd83d3f6d1a3059",
  "base_voice": "male"
}
=== END configs/speaker_selection.json ===
PASS: selection saved
CELL-OK C4.3-notes
```
