# Colab run C4 run 2 — C4 — save the mixed-voice selection

- File: `C4_speaker_run2.ipynb` · spec `notebooks/specs/c4_speaker_run2.py`
- Run: commit `fcb64e5` · branch `chunk/C4-speaker` · GPU `none` · time `2026-11-07T05:58:27+00:00`
- Staleness: OK: the run used fcb64e5, notebook built in fcb64e5
- Result: 6/6 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C4.3b | Save cluster 1 (mixed voices) with base voice male | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
fcb64e5 chore(notebook): build C4 run 2 to save the mixed-voice selection
PASS: /content/boinga-translator on chunk/C4-speaker
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
git_commit         fcb64e5
disk_free_gb_root  9.4
disk_free_gb_fast  107.6
utc_time           2026-11-07T05:58:27+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 107.6,
  "disk_free_gb_root": 9.4,
  "git_commit": "fcb64e5",
  "gpu_mem_gb": null,
  "gpu_name": "none",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.18",
  "torch": "2.14.1+cpu",
  "torch_cuda": null,
  "torch_error": null,
  "utc_time": "2026-11-07T05:58:27+00:00"
}
=== END reports/env.json ===
PASS: env recorded
WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run needs one
RUN-HEADER chunk=C4 run=2 branch=chunk/C4-speaker commit=fcb64e5 gpu=none utc=2026-11-07T05:58:27+00:00
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
 'git_commit': 'fcb64e5',
 'disk_free_gb_root': 9.4,
 'disk_free_gb_fast': 107.6,
 'utc_time': '2026-11-07T05:58:27+00:00'}
```

### C4.3b — Save cluster 1 (mixed voices) with base voice male

```text
=== BEGIN configs/speaker_selection.json ===
{
  "kind": "cluster",
  "id": 1,
  "n_clips": 7506,
  "clean_minutes": 490.73,
  "minutes_by_official_split": {
    "train": 393.72,
    "val": 48.25,
    "test": 48.76
  },
  "csv_sha256": "e170000ef948cfa3051edabdd05b4e6376ee704763ebbf7af27a47964dac592e",
  "base_voice": "male"
}
=== END configs/speaker_selection.json ===
PASS: selection saved
CELL-OK C4.3b
```
