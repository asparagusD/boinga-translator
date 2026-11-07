# Colab run C6 run 1 — C6 — prepared dataset

- File: `C6_prepare_run1.ipynb` · spec `notebooks/specs/c6_prepare.py`
- Run: commit `93d9796` · branch `chunk/C6-prepare` · GPU `none` · time `2026-11-07T09:55:25+00:00`
- Staleness: OK: the run used 93d9796, notebook built in f3564b5
- Result: 9/9 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C6.0 | Ensure dataset on this VM | PASS |  |
| C6.4 | Prepare dataset | PASS |  |
| C6.3 | Coqui parses our manifests | PASS |  |
| C6.4.play | Listen: 5 prepared clips | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
93d9796 fix(prepare): keep the clip_id column name on pandas 2 in make_splits
PASS: /content/boinga-translator on chunk/C6-prepare
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
PASS: 12 project folders ready under /content/drive/MyDrive/boinga-translator
```

### HDR — Run header

```text
python             3.13.18
platform           Linux-6.6.159+-x86_64-with-glibc2.39
is_colab           True
torch              2.14.1+cpu
torch_cuda         None
torch_error        None
gpu_name           none
gpu_mem_gb         None
coqui_tts          0.28.0
numpy              2.4.6
git_commit         93d9796
disk_free_gb_root  9.4
disk_free_gb_fast  107.5
utc_time           2026-11-07T09:55:25+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 107.5,
  "disk_free_gb_root": 9.4,
  "git_commit": "93d9796",
  "gpu_mem_gb": null,
  "gpu_name": "none",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.18",
  "torch": "2.14.1+cpu",
  "torch_cuda": null,
  "torch_error": null,
  "utc_time": "2026-11-07T09:55:25+00:00"
}
=== END reports/env.json ===
PASS: env recorded
WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run needs one
RUN-HEADER chunk=C6 run=1 branch=chunk/C6-prepare commit=93d9796 gpu=none utc=2026-11-07T09:55:25+00:00
{'python': '3.13.18',
 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39',
 'is_colab': True,
 'torch': '2.14.1+cpu',
 'torch_cuda': None,
 'torch_error': None,
 'gpu_name': 'none',
 'gpu_mem_gb': None,
 'coqui_tts': '0.28.0',
 'numpy': '2.4.6',
 'git_commit': '93d9796',
 'disk_free_gb_root': 9.4,
 'disk_free_gb_fast': 107.5,
 'utc_time': '2026-11-07T09:55:25+00:00'}
```

### C6.0 — Ensure dataset on this VM

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
CELL-OK C6.0
```

### C6.4 — Prepare dataset

```text
selection: 7506 clips, 0 excluded by char_map
transform: 7506 clip(s) to do, 0 already done
100% 7506/7506 [01:42<00:00, 73.51it/s] 
backed up /content/boinga-fast/prepared to /content/drive/MyDrive/boinga-translator/data/prepared/prepared-fd93f4827c6d2b41.tar
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
CELL-OK C6.4
```

### C6.3 — Coqui parses our manifests

```text
train: 6415 samples from metadata_train.csv
val: 338 samples from metadata_val.csv
test: 753 samples from metadata_test.csv
first train sample: text='চাটগাঁইয়া ডঅর চাটগাঁ অঞ্চলর আঞ্চলিক ভাষা.\n' audio_file=/content/boinga-fast/prepared/wavs/2.wav
PASS: coqui parsed 6415 train / 338 val samples (+ 753 test)
CELL-OK C6.3
```

### C6.4.play — Listen: 5 prepared clips

```text
AUDIO 1217: মোবাইল ফোন গান বাইর গরিয়েরে ইতের ছবি তুলি ফেলাই.:
<IPython.lib.display.Audio object>
AUDIO 131: গেইয়িদে কালিয়া:
<IPython.lib.display.Audio object>
AUDIO 1131: ওসি সাব শূন্যদৃষ্টিত ইবার সামনে দেবালত ঝুলানো ক্যালেণ্ডারর মিক্ষে চাই আছেন.:
<IPython.lib.display.Audio object>
AUDIO 2582: গত দুয়ো পূর্ণিমাত আঁই ঘরত ত্থুন্ বাইর ন হোই .:
<IPython.lib.display.Audio object>
AUDIO 2124: আঁই খতা দির .:
<IPython.lib.display.Audio object>
CELL-OK C6.4.play
```
