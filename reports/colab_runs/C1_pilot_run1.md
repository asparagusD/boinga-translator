# Colab run C1 run 1 — Pipeline pilot

- File: `C1_pilot_run1.ipynb` · spec `notebooks/specs/c1_pilot.py`
- Run: commit `e5568ea` · branch `chunk/C1N-notebook-pipeline` · GPU `Tesla_T4` · time `2026-11-07T00:41:13+00:00`
- Staleness: OK: the run used e5568ea, notebook built in 21113ba
- Result: 8/8 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C1.10.a | Env report | PASS |  |
| C1.10.b | Drive write test | PASS |  |
| C1.10-notes | Your observations | PASS |  |

## Form values

### C1.10-notes — Your observations

- `GPU_SHOWN` = "Tesla T4"
- `ANY_PROBLEMS` = "none"

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
e5568ea docs(notebook): close C1N review notes before the pilot re-run
PASS: /content/boinga-translator on chunk/C1N-notebook-pipeline
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
git_commit         e5568ea
disk_free_gb_root  13.0
disk_free_gb_fast  112.5
utc_time           2026-11-07T00:41:13+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 13.0,
  "git_commit": "e5568ea",
  "gpu_mem_gb": 14.6,
  "gpu_name": "Tesla T4",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T00:41:13+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C1 run=1 branch=chunk/C1N-notebook-pipeline commit=e5568ea gpu=Tesla_T4 utc=2026-11-07T00:41:13+00:00
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
 'git_commit': 'e5568ea',
 'disk_free_gb_root': 13.0,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-07T00:41:13+00:00'}
```

### C1.10.a — Env report

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
git_commit         e5568ea
disk_free_gb_root  13.0
disk_free_gb_fast  112.5
utc_time           2026-11-07T00:41:19+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 13.0,
  "git_commit": "e5568ea",
  "gpu_mem_gb": 14.6,
  "gpu_name": "Tesla T4",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T00:41:19+00:00"
}
=== END reports/env.json ===
PASS: env recorded
CELL-OK C1.10.a
```

### C1.10.b — Drive write test

```text
PASS: drive ok (/content/drive/MyDrive/boinga-translator/reports/pilot.txt)
CELL-OK C1.10.b
```

### C1.10-notes — Your observations

```text
=== BEGIN reports/notes/C1.10-notes.json ===
{
  "ANY_PROBLEMS": "none",
  "GPU_SHOWN": "Tesla T4"
}
=== END reports/notes/C1.10-notes.json ===
CELL-OK C1.10-notes
```
