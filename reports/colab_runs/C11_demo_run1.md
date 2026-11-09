# Colab run C11 run 1 — C11 — demo

- File: `C11_demo_run1.ipynb` · spec `notebooks/specs/c11_demo.py`
- Run: commit `0280e8c` · branch `chunk/C11-demo` · GPU `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` · time `2026-11-09T02:29:13+00:00`
- Staleness: OK: the run used 0280e8c, notebook built in a1bc809
- Result: 8/8 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C11.2.check | Automated demo checks | PASS |  |
| C11.2.live | Live demo (stop the cell when done) | PASS |  |
| C11-notes | Your observations — demo | PASS |  |

## Form values

### C11-notes — Your observations — demo

- `DEMO_WORKED` = "yes"
- `NOTES` = "none"

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
0280e8c chore(notebook): build C11 run 1 notebook
PASS: /content/boinga-translator on chunk/C11-demo
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
git_commit         0280e8c
disk_free_gb_root  3.5
disk_free_gb_fast  112.5
utc_time           2026-11-09T02:29:13+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 48,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 3.5,
  "git_commit": "0280e8c",
  "gpu_mem_gb": 95.0,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-09T02:29:13+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C11 run=1 branch=chunk/C11-demo commit=0280e8c gpu=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition utc=2026-11-09T02:29:13+00:00
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
 'git_commit': '0280e8c',
 'disk_free_gb_root': 3.5,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-09T02:29:13+00:00'}
```

### C11.2.check — Automated demo checks

```text
demo selfcheck:   0% 0/6 [00:00<?, ?case/s]synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
CASE normal-boinga: ok 3.21 s
demo selfcheck:  17% 1/6 [00:53<04:29, 53.94s/case]Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
CASE normal-baseline: ok 3.08 s
demo selfcheck:  33% 2/6 [03:00<06:27, 96.77s/case]CASE empty: error Please enter some Bengali text.
CASE too-long: error Please keep it under 300 characters.
hello world
Character 'h' not found in the vocabulary. Discarding it.
hello world
Character 'e' not found in the vocabulary. Discarding it.
hello world
Character 'l' not found in the vocabulary. Discarding it.
hello world
Character 'o' not found in the vocabulary. Discarding it.
hello world
Character 'w' not found in the vocabulary. Discarding it.
hello world
Character 'r' not found in the vocabulary. Discarding it.
hello world
Character 'd' not found in the vocabulary. Discarding it.
CASE latin-mixed: ok 5.06 s
CASE bengali-digits: ok 5.52 s
demo selfcheck: 100% 6/6 [03:00<00:00, 30.13s/case]
PASS: demo selfcheck (6 cases)
CELL-OK C11.2.check
```

### C11.2.live — Live demo (stop the cell when done)

```text
Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().
* Running on public URL: https://5fca477759529c38e0.gradio.live

This share link is temporary and will last for up to 1 week (best effort). For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
<IPython.core.display.HTML object>
Keyboard interruption in main thread... closing server.
Killing tunnel 127.0.0.1:7860 <> https://5fca477759529c38e0.gradio.live
CELL-OK C11.2.live
```

### C11-notes — Your observations — demo

```text
=== BEGIN reports/notes/C11-notes.json ===
{
  "DEMO_WORKED": "yes",
  "NOTES": "none"
}
=== END reports/notes/C11-notes.json ===
CELL-OK C11-notes
```
