# Colab run C3 run 1 — Download and layout

- File: `C3_data_audit_run1.ipynb` · spec `notebooks/specs/c3_data_audit_run1.py`
- Run: commit `f06163c` · branch `chunk/C3-data-audit` · GPU `none` · time `2026-11-07T02:05:40+00:00`
- Staleness: OK: the run used f06163c, notebook built in f06163c
- Result: 7/7 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C3.1 | Download and unpack corpus | PASS |  |
| C3.2 | Inspect corpus layout | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
f06163c fix(notebook): stop Run all when the C3.1 download fails
PASS: /content/boinga-translator on chunk/C3-data-audit
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
git_commit         f06163c
disk_free_gb_root  11.1
disk_free_gb_fast  107.6
utc_time           2026-11-07T02:05:40+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 107.6,
  "disk_free_gb_root": 11.1,
  "git_commit": "f06163c",
  "gpu_mem_gb": null,
  "gpu_name": "none",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.18",
  "torch": "2.14.1+cpu",
  "torch_cuda": null,
  "torch_error": null,
  "utc_time": "2026-11-07T02:05:40+00:00"
}
=== END reports/env.json ===
PASS: env recorded
WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run needs one
RUN-HEADER chunk=C3 run=1 branch=chunk/C3-data-audit commit=f06163c gpu=none utc=2026-11-07T02:05:40+00:00
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
 'git_commit': 'f06163c',
 'disk_free_gb_root': 11.1,
 'disk_free_gb_fast': 107.6,
 'utc_time': '2026-11-07T02:05:40+00:00'}
```

### C3.1 — Download and unpack corpus

```text
Dataset.zip: 100%|██████████| 844M/844M [00:23<00:00, 35.2MB/s]
DONE download
DONE verify
DONE unzip
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
CELL-OK C3.1
```

### C3.2 — Inspect corpus layout

```text
=== BEGIN reports/corpus_layout.txt ===
Corpus layout of /content/boinga-fast/data
macOS zip leftovers skipped (__MACOSX/, ._*): 0 files

== Tree (depth 3; files per extension) ==
./  ((none) 1)
  Dataset/  (.xlsx 1)
    Speech/  (no files)
      test/  (.wav 800)
      training/  (.wav 6403)
      validation/  (.wav 801)

== Wav headers (first, middle, last) ==
wav files: 8004
Dataset/Speech/test/1001.wav: samplerate=16000 channels=1 duration=1.498s format=WAV subtype=PCM_16
Dataset/Speech/training/4596.wav: samplerate=16000 channels=1 duration=3.762s format=WAV subtype=PCM_16
Dataset/Speech/validation/990.wav: samplerate=16000 channels=1 duration=4.672s format=WAV subtype=PCM_16

== Metadata columns (speaker/gender/region/age/split; top 20 values) ==
none found

== Tables (< 5 MB; first 5 rows; long cells cut for display only) ==
Dataset/Chittagonian_translated.xlsx [Chittagonian_translated]: 8009 rows, 3 columns
  columns: Standard Bangla, Translated English, Chittagonian Dialect
                               Standard Bangla                       Translated English                     Chittagonian Dialect
    0  চাটগাঁইয়া বৃহত্তর চট্টগ্রাম অঞ্চলের ...  Chittagonian is the regional languag...  চাটগাঁইয়া ডঅর চাটগাঁ অঞ্চলর আঞ্চলিক ...
    1  বাংলা ভাষার সঙ্গে চাটগাঁইয়া ভাষার শা...  It is considered regional Bengali in...  বাঁআলা ভাষার হোঁয়ারে চাটগাঁইয়া ভাষার...
    2  তবে কেউ কেউ একে স্বতন্ত্র ভাষার মর্য...  But some are keen to give it the sta...  তয় কেউ কেউ এ ভাষারে ওউগগা স্বাধীন ভা...
    3               চাটগাঁইয়া মূলত মৌখিক ভাষা।  Chittagonian is primarily an oral la...                চাটগাঁইয়া আদতে মুখর ভাষা।
    4               এর নিজস্ব লিপি পদ্ধতি নেই।  It does not have its own scripting s...         ইবা নিজর মতো লেকনর কন কুদরত নাই।
=== END reports/corpus_layout.txt ===
PASS: layout inspected
CELL-OK C3.2
```
