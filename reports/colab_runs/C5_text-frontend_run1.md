# Colab run C5 run 1 — C5 run 1 — OOV report

- File: `C5_text_frontend_run1.ipynb` · spec `notebooks/specs/c5_text_frontend_run1.py`
- Run: commit `9683e31` · branch `chunk/C5-text-frontend` · GPU `none` · time `2026-11-07T06:49:27+00:00`
- Staleness: OK: the run used 9683e31, notebook built in 86064cb
- Result: 7/7 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C5.0 | Ensure dataset on this VM | PASS |  |
| C5.2 | OOV report | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
9683e31 chore(notebook): build C5 run 1 notebook
PASS: /content/boinga-translator on chunk/C5-text-frontend
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
git_commit         9683e31
disk_free_gb_root  9.4
disk_free_gb_fast  107.5
utc_time           2026-11-07T06:49:27+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 107.5,
  "disk_free_gb_root": 9.4,
  "git_commit": "9683e31",
  "gpu_mem_gb": null,
  "gpu_name": "none",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.18",
  "torch": "2.14.1+cpu",
  "torch_cuda": null,
  "torch_error": null,
  "utc_time": "2026-11-07T06:49:27+00:00"
}
=== END reports/env.json ===
PASS: env recorded
WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run needs one
RUN-HEADER chunk=C5 run=1 branch=chunk/C5-text-frontend commit=9683e31 gpu=none utc=2026-11-07T06:49:27+00:00
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
 'git_commit': '9683e31',
 'disk_free_gb_root': 9.4,
 'disk_free_gb_fast': 107.5,
 'utc_time': '2026-11-07T06:49:27+00:00'}
```

### C5.0 — Ensure dataset on this VM

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
CELL-OK C5.0
```

### C5.2 — OOV report

```text
=== BEGIN reports/oov_report.md ===
# OOV report

Characters of the model text (text_frontend.prepare, char map 151ac177e2) missing from the base voice **male**'s charset (characters | punctuations | space). Made by `boinga oov`.

- Clips: 7506 selected clips checked; 0 selected clips excluded by char_map exclude_rows_with (not checked: they never reach training)
- OOV character rate: 0 of 251223 characters (0.00%)
- Clips with any OOV character: 0 of 7506 clips (0.00%)

No character is missing from the charset.
=== END reports/oov_report.md ===
PASS: oov report: 0 missing character(s) in 0 of 7506 clips
CELL-OK C5.2
```
