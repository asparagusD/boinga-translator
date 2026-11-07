# Colab run C5 run 2 — C5 run 2 — tokenizer check

- File: `C5_text_frontend_run2.ipynb` · spec `notebooks/specs/c5_text_frontend_run2.py`
- Run: commit `90b60c3` · branch `chunk/C5-text-frontend` · GPU `none` · time `2026-11-07T08:22:08+00:00`
- Staleness: OK: the run used 90b60c3, notebook built in 49cc675
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
| C5.4 | Tokenizer round trip | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
90b60c3 chore(notebook): build C5 run 2 notebook
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
git_commit         90b60c3
disk_free_gb_root  9.4
disk_free_gb_fast  107.6
utc_time           2026-11-07T08:22:08+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 107.6,
  "disk_free_gb_root": 9.4,
  "git_commit": "90b60c3",
  "gpu_mem_gb": null,
  "gpu_name": "none",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.18",
  "torch": "2.14.1+cpu",
  "torch_cuda": null,
  "torch_error": null,
  "utc_time": "2026-11-07T08:22:08+00:00"
}
=== END reports/env.json ===
PASS: env recorded
WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run needs one
RUN-HEADER chunk=C5 run=2 branch=chunk/C5-text-frontend commit=90b60c3 gpu=none utc=2026-11-07T08:22:08+00:00
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
 'git_commit': '90b60c3',
 'disk_free_gb_root': 9.4,
 'disk_free_gb_fast': 107.6,
 'utc_time': '2026-11-07T08:22:08+00:00'}
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

### C5.4 — Tokenizer round trip

```text
voice male: 7506 clips, 0 excluded by char_map
male: model.pth 997.77 MB, config.json 0.01 MB
PASS: male cached (/content/drive/MyDrive/boinga-translator/pretrained/male)
=== BEGIN reports/tokenizer_check.md ===
# Tokenizer check

coqui TTSTokenizer from `/content/drive/MyDrive/boinga-translator/pretrained/male/config.json` on prepare_text() of each clip (char map 151ac177e2). Made by `boinga check-tokenizer`.

- Clips checked: 7506
- Unknown character rate: 0 of 297218 characters (0.000000)
- Round-trip failures: 0 of 7506 clips

Every clip round-trips: decode(encode(text)) == text, nothing dropped.
=== END reports/tokenizer_check.md ===
PASS: tokenizer check unknown_rate=0.000000 round_trip_failures=0
CELL-OK C5.4
```
