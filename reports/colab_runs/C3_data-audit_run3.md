# Colab run C3 run 3 — Char-mapped re-audit

- File: `C3_data_audit_run3.ipynb` · spec `notebooks/specs/c3_data_audit_run3.py`
- Run: commit `eb3d3e0` · branch `chunk/C3-data-audit` · GPU `none` · time `2026-11-07T04:13:18+00:00`
- Staleness: OK: the run used eb3d3e0, notebook built in eb3d3e0
- Result: 8/8 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C3.8c.fetch | Ensure dataset on this VM | PASS |  |
| C3.8c.audit | Re-audit all clips on the mapped text (~1 min) | PASS |  |
| C3.8c.summary | Audit summary with the char map | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
eb3d3e0 chore(notebook): build C3 run 3 notebook for the char-mapped re-audit
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
git_commit         eb3d3e0
disk_free_gb_root  9.4
disk_free_gb_fast  107.6
utc_time           2026-11-07T04:13:18+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 107.6,
  "disk_free_gb_root": 9.4,
  "git_commit": "eb3d3e0",
  "gpu_mem_gb": null,
  "gpu_name": "none",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.18",
  "torch": "2.14.1+cpu",
  "torch_cuda": null,
  "torch_error": null,
  "utc_time": "2026-11-07T04:13:18+00:00"
}
=== END reports/env.json ===
PASS: env recorded
WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run needs one
RUN-HEADER chunk=C3 run=3 branch=chunk/C3-data-audit commit=eb3d3e0 gpu=none utc=2026-11-07T04:13:18+00:00
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
 'git_commit': 'eb3d3e0',
 'disk_free_gb_root': 9.4,
 'disk_free_gb_fast': 107.6,
 'utc_time': '2026-11-07T04:13:18+00:00'}
```

### C3.8c.fetch — Ensure dataset on this VM

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
CELL-OK C3.8c.fetch
```

### C3.8c.audit — Re-audit all clips on the mapped text (~1 min)

```text
voice female
skipped 0 rows already in /content/drive/MyDrive/boinga-translator/reports/audit.parts (female-a4543ed63b-151ac177e2)
audit: 100%|██████████| 8009/8009 [01:02<00:00, 128.53it/s]
wrote 8009 rows to /content/drive/MyDrive/boinga-translator/reports/audit.parts
PASS: audited 8009 clips
CELL-OK C3.8c.audit
```

### C3.8c.summary — Audit summary with the char map

```text
=== BEGIN reports/audit_summary.md ===
# Corpus audit summary

Charset: female-a4543ed63b-151ac177e2. Rows: 8009; decodable clips: 8004, 9.04 h.

## Per split

| split | clips | hours | clean clips | clean hours |
|---|---|---|---|---|
| (none) | 5 | 0.00 | 0 | 0.00 |
| test | 800 | 0.89 | 755 | 0.81 |
| train | 6403 | 7.26 | 6010 | 6.56 |
| val | 801 | 0.89 | 748 | 0.81 |

## Audio formats

| sample_rate | channels | subtype | clips |
|---|---|---|---|
| 16000 | 1 | PCM_16 | 8004 |

## Exclusions (a clip can have several reasons)

| reason | clips | hours |
|---|---|---|
| undecodable (or no wav) | 5 | 0.00 |
| too short (< 1 s) | 8 | 0.00 |
| too long (> 15 s) | 29 | 0.14 |
| clipped (clip_ratio > 0.001) | 7 | 0.01 |
| quiet (rms_db < -45) | 0 | 0.00 |
| empty text | 0 | 0.00 |
| excluded character (char_map: / : * digits) | 279 | 0.62 |
| unknown to the model after char_map | 114 | 0.24 |
| speech rate (chars/s outside 1st-99th percentile, or no speech) | 167 | 0.13 |
| duplicate text | 28 | 0.02 |

Clean: 7513 clips, 8.18 h.

## Speakers

no speaker field in the corpus (DECISIONS "Corpus layout")

## Characters not in the model after char_map (top 20, rows containing)

| char | code | name | rows |
|---|---|---|---|
| ৭ | U+09ED | BENGALI DIGIT SEVEN | 50 |
| ৯ | U+09EF | BENGALI DIGIT NINE | 48 |
| / | U+002F | SOLIDUS | 19 |
| : | U+003A | COLON | 7 |
| * | U+002A | ASTERISK | 1 |
=== END reports/audit_summary.md ===
PASS: summary written
CELL-OK C3.8c.summary
```
