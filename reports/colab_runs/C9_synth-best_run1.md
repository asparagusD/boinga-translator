# Colab run C9 run 1 — C9 — best checkpoint

- File: `C9_synth_best_run1.ipynb` · spec `notebooks/specs/c9_synth_best.py`
- Run: commit `dcb26e5` · branch `chunk/C9-synth-best` · GPU `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` · time `2026-11-08T08:00:31+00:00`
- Staleness: OK: the run used dcb26e5, notebook built in a985044
- Result: 10/10 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C9.pre | Restore prepared data | PASS |  |
| C9.2 | Promote the chosen checkpoint | PASS |  |
| C9.3 | Reload best checkpoint in a fresh process | PASS |  |
| C9.3.play | Listen: reload check | PASS |  |
| C9-notes | Your observations — best checkpoint | PASS |  |

## Form values

### C9-notes — Your observations — best checkpoint

- `INTELLIGIBLE` = "yes"
- `NOTES` = "none"

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
dcb26e5 chore(notebook): build C9 run 1 notebook
PASS: /content/boinga-translator on chunk/C9-synth-best
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
PASS: 13 project folders ready under /content/drive/MyDrive/boinga-translator
```

### HDR — Run header

```text
python             3.13.16
platform           Linux-6.6.159+-x86_64-with-glibc2.39
cpu_count          48
is_colab           True
torch              2.14.1+cu130
torch_cuda         13.0
torch_error        None
gpu_name           NVIDIA RTX PRO 6000 Blackwell Server Edition
gpu_mem_gb         95.0
coqui_tts          0.28.0
numpy              2.4.6
git_commit         dcb26e5
disk_free_gb_root  4.5
disk_free_gb_fast  112.5
utc_time           2026-11-08T08:00:31+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 48,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 4.5,
  "git_commit": "dcb26e5",
  "gpu_mem_gb": 95.0,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-08T08:00:31+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C9 run=1 branch=chunk/C9-synth-best commit=dcb26e5 gpu=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition utc=2026-11-08T08:00:31+00:00
{'python': '3.13.16',
 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39',
 'cpu_count': 48,
 'is_colab': True,
 'torch': '2.14.1+cu130',
 'torch_cuda': '13.0',
 'torch_error': None,
 'gpu_name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
 'gpu_mem_gb': 95.0,
 'coqui_tts': '0.28.0',
 'numpy': '2.4.6',
 'git_commit': 'dcb26e5',
 'disk_free_gb_root': 4.5,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-08T08:00:31+00:00'}
```

### C9.pre — Restore prepared data

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
copy to fast disk: 100% 8005/8005 [03:14<00:00, 41.15file/s]
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
selection: 7506 clips, 0 excluded by char_map
restore: 100% 7512/7512 [00:04<00:00, 1511.09file/s]
restored /content/boinga-fast/prepared from /content/drive/MyDrive/boinga-translator/data/prepared/prepared-fd93f4827c6d2b41.tar
transform: 0 clip(s) to do, 7506 already done
0it [00:00, ?it/s]
train: 6415 clips, 386.64 min
val: 338 clips, 20.48 min
test: 753 clips, 44.81 min
upsampled: 7506 of 7506 clips; mean trimmed 0.310 s
failed: 0 clip(s), by reason: {}
--- metadata_train.csv (first 3 lines)
2|চাটগাঁইয়া ডঅর চাটগাঁ অঞ্চলর আঞ্চলিক ভাষা.|চাটগাঁইয়া ডঅর চাটগাঁ অঞ্চলর আঞ্চলিক ভাষা.
3|বাঁআলা ভাষার হোঁয়ারে চাটগাঁইয়া ভাষার বউত শব্দর বেশাবেশি মিল থয়নে ইবারে আঞ্চলিক ভাষা ইসাবে ধরা অয়.|বাঁআলা ভাষার হোঁয়ারে চাটগাঁইয়া ভাষার বউত শব্দর বেশাবেশি মিল থয়নে ইবারে আঞ্চলিক ভাষা ইসাবে ধরা অয়.
4|তয় কেউ কেউ এ ভাষারে ওউগগা স্বাধীন ভাষার মজ্জাদা দিবাল্লাই পঅল.|তয় কেউ কেউ এ ভাষারে ওউগগা স্বাধীন ভাষার মজ্জাদা দিবাল্লাই পঅল.
--- metadata_val.csv (first 3 lines)
23|এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.|এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.
50|অনে কন ডইল্যা বেয়ারি?|অনে কন ডইল্যা বেয়ারি?
66|ইত্তিরি বেআন.|ইত্তিরি বেআন.
--- metadata_test.csv (first 3 lines)
6|ইবা নিজর মতো লেকনর কন কুদরত নাই.|ইবা নিজর মতো লেকনর কন কুদরত নাই.
29|যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি.|যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি.
58|ইত্তিরি হাঁজুইন্না সাততো.|ইত্তিরি হাঁজুইন্না সাততো.
=== BEGIN configs/splits_summary.json ===
{
  "counts": {
    "train": 6415,
    "val": 338,
    "test": 753
  },
  "minutes": {
    "train": 419.62,
    "val": 22.35,
    "test": 48.76
  },
  "csv_sha256": "fd93f4827c6d2b413a51b3703c0debcf294eb436c0ec2391dbe69c85e77e016e",
  "seed": 1234,
  "test_source": "official",
  "dropped_duplicate_text": 0
}
=== END configs/splits_summary.json ===
PASS: prepared 7506 clips
CELL-OK C9.pre
```

### C9.2 — Promote the chosen checkpoint

```text
source: /content/drive/MyDrive/boinga-translator/runs/run1/run1-November-08-2026_05+25AM-853ac3a/checkpoint_36500.pth (997.8 MB)
replaces: nothing (/content/drive/MyDrive/boinga-translator/best holds no model)
copy checkpoint_36500.pth: 100% 998M/998M [00:13<00:00, 71.4MB/s]
=== BEGIN reports/best_provenance.json ===
{
  "run": "run1",
  "step": 36500,
  "run_folder": "run1-November-08-2026_05+25AM-853ac3a",
  "source": "checkpoint_36500.pth",
  "bytes": 997792643,
  "md5": "66970fd54dd14a5447517fa6b8cb3e85",
  "eval": {
    "step": 36475,
    "loss_mel": 20.5995,
    "loss_total": 43.8919
  },
  "base_model": {
    "voice": "male",
    "id": "tts_models/bn/custom/vits-male",
    "license": "Apache-2.0",
    "coqui_tts": "0.28.0"
  },
  "dataset": {
    "name": "A Synchronized Chittagonian Speech Corpus for Dialect ASR and Speech-to-Speech Translation",
    "doi": "10.17632/n9xttk45df.1",
    "license": "CC-BY-4.0"
  },
  "speaker_selection": {
    "kind": "cluster",
    "id": 1,
    "csv_sha256": "e170000ef948cfa3051edabdd05b4e6376ee704763ebbf7af27a47964dac592e"
  },
  "char_map": "151ac177e2",
  "seed": 1234,
  "inference": {
    "inference_noise_scale": 0.667,
    "inference_noise_scale_dp": 0.8,
    "length_scale": 1.0
  },
  "git_commit": "dcb26e5",
  "promoted_utc": "2026-11-08T08:07:01+00:00"
}
=== END reports/best_provenance.json ===
PASS: promoted run1 step 36500
CELL-OK C9.2
```

### C9.3 — Reload best checkpoint in a fresh process

```text
reload check:   0% 0/3 [00:00<?, ?wav/s]synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
0_6.wav: 2.94 s, RMS -17.3 dBFS, ok: ইবা নিজর মতো লেকনর কন কুদরত নাই.
reload check:  33% 1/3 [00:07<00:14,  7.11s/wav]1_29.wav: 5.23 s, RMS -18.4 dBFS, ok: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি.
2_58.wav: 2.71 s, RMS -21.7 dBFS, ok: ইত্তিরি হাঁজুইন্না সাততো.
reload check: 100% 3/3 [00:07<00:00,  2.41s/wav]
PASS: reload check (3 wavs)
CELL-OK C9.3
```

### C9.3.play — Listen: reload check

```text
AUDIO #0 6: ইবা নিজর মতো লেকনর কন কুদরত নাই.:
<IPython.lib.display.Audio object>
AUDIO #1 29: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি.:
<IPython.lib.display.Audio object>
AUDIO #2 58: ইত্তিরি হাঁজুইন্না সাততো.:
<IPython.lib.display.Audio object>
CELL-OK C9.3.play
```

### C9-notes — Your observations — best checkpoint

```text
=== BEGIN reports/notes/C9-notes.json ===
{
  "INTELLIGIBLE": "yes",
  "NOTES": "none"
}
=== END reports/notes/C9-notes.json ===
CELL-OK C9-notes
```
