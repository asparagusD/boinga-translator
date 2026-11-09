# Colab run C10 run 2 — evaluation audio

- File: `C10_eval_run2.ipynb` · spec `notebooks/specs/c10_eval_run2.py`
- Run: commit `05720c3` · branch `chunk/C10-eval` · GPU `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` · time `2026-11-08T21:12:59+00:00`
- Staleness: OK: the run used 05720c3, notebook built in 5a679fc
- Result: 9/9 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C10.pre | Restore data | PASS |  |
| C10.2 | Generate A/B/R audio | PASS |  |
| C10.2.play | Listen: 3 prompts side by side | PASS |  |
| C10.3 | Build blind rating sheet | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
05720c3 chore(notebook): hand off C10 run 2 notebook (audio and blind sheet)
PASS: /content/boinga-translator on chunk/C10-eval
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
git_commit         05720c3
disk_free_gb_root  3.6
disk_free_gb_fast  112.5
utc_time           2026-11-08T21:12:59+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 48,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 3.6,
  "git_commit": "05720c3",
  "gpu_mem_gb": 95.0,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-08T21:12:59+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C10 run=2 branch=chunk/C10-eval commit=05720c3 gpu=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition utc=2026-11-08T21:12:59+00:00
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
 'git_commit': '05720c3',
 'disk_free_gb_root': 3.6,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-08T21:12:59+00:00'}
```

### C10.pre — Restore data

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
copy to fast disk: 100% 8005/8005 [01:15<00:00, 106.42file/s]
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
CELL-OK C10.pre
```

### C10.2 — Generate A/B/R audio

```text
eval audio:   0% 0/20 [00:00<?, ?prompt/s]Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
eval audio: 100% 20/20 [01:25<00:00,  4.29s/prompt]
=== BEGIN reports/eval_generation.md ===
# Evaluation audio

20 prompts (configs/eval_prompts.json), audio on Drive eval/final/. Voices use the fixed seed and inference scales; R_real is the corpus clip through prepare.transform_audio.

| set | wavs | mean duration s | min s | max s | mean RMS dBFS |
|---|---|---|---|---|---|
| A_baseline | 20 | 3.59 | 2.38 | 6.76 | -24.1 |
| B_boinga | 20 | 2.83 | 1.61 | 5.22 | -17.7 |
| R_real | 20 | 2.90 | 1.87 | 4.63 | -16.6 |
=== END reports/eval_generation.md ===
PASS: generated A=20 B=20 R=20
CELL-OK C10.2
```

### C10.2.play — Listen: 3 prompts side by side

```text
AUDIO P01 A_baseline: অ্যাক্কানা হুইত পাইজ্জম না, ছার? :
<IPython.lib.display.Audio object>
AUDIO P01 B_boinga: অ্যাক্কানা হুইত পাইজ্জম না, ছার? :
<IPython.lib.display.Audio object>
AUDIO P01 R_real: অ্যাক্কানা হুইত পাইজ্জম না, ছার? :
<IPython.lib.display.Audio object>
AUDIO P02 A_baseline: হেঁততে উগগো পোয়া ছুটি আইসসি।:
<IPython.lib.display.Audio object>
AUDIO P02 B_boinga: হেঁততে উগগো পোয়া ছুটি আইসসি।:
<IPython.lib.display.Audio object>
AUDIO P02 R_real: হেঁততে উগগো পোয়া ছুটি আইসসি।:
<IPython.lib.display.Audio object>
AUDIO P03 A_baseline: অনরে  ফেলাই  ধান পরিবু বুলি  ন ধাই ।:
<IPython.lib.display.Audio object>
AUDIO P03 B_boinga: অনরে  ফেলাই  ধান পরিবু বুলি  ন ধাই ।:
<IPython.lib.display.Audio object>
AUDIO P03 R_real: অনরে  ফেলাই  ধান পরিবু বুলি  ন ধাই ।:
<IPython.lib.display.Audio object>
CELL-OK C10.2.play
```

### C10.3 — Build blind rating sheet

```text
blind clips: 100% 40/40 [00:00<00:00, 221.62wav/s]
=== BEGIN reports/rater_instructions.md ===
# Rating instructions / রেটিং নির্দেশিকা

## English

You will hear 20 short sentences in Chittagonian. Each sentence is spoken by two computer
voices, in `<id>_clip1.wav` and `<id>_clip2.wav`. The order is random and you are not told
which voice is which: please do not try to guess. Read the sentence in `rating_sheet.csv`,
listen to each clip as often as you like, then fill in its row:

- `rater`: your name or initials (the same in every row).
- `intelligibility` (1-5): how easily you understand the words. 1 = nothing, 5 = every word.
- `naturalness` (1-5): how human and fluent it sounds. 1 = very robotic or broken,
  5 = like a real person.
- `dialect_authenticity` (1-5): how much it sounds like a native Chittagonian speaker.
  1 = not at all (e.g. standard Bangla or a foreign accent), 5 = a native speaker.
- `dialect_words` (1-5): are the Chittagonian-specific words said correctly?
  1 = mostly wrong, 5 = all correct.
- `comments` (optional): anything you noticed, e.g. a wrong word, noise, speed.

Write the scores as the digits 1, 2, 3, 4 or 5. Rate each clip on its own. Please do not
share the files or your ratings with other raters until everyone has finished.

## বাংলা

আপনি চাটগাঁইয়া ভাষায় ২০টি ছোট বাক্য শুনবেন। প্রতিটি বাক্য দুটি কম্পিউটার কণ্ঠে বলা
হয়েছে: `<id>_clip1.wav` এবং `<id>_clip2.wav`। ক্রম এলোমেলো, আর কোনটি কোন কণ্ঠ তা
জানানো হবে না: দয়া করে অনুমান করার চেষ্টা করবেন না। `rating_sheet.csv`-এ বাক্যটি পড়ুন, প্রতিটি
ক্লিপ যতবার ইচ্ছা শুনুন, তারপর তার সারিটি পূরণ করুন:

- `rater`: আপনার নাম বা নামের আদ্যক্ষর (প্রতিটি সারিতে একই)।
- `intelligibility` (1-5): শব্দগুলো কত সহজে বোঝা যায়। 1 = কিছুই না, 5 = প্রতিটি শব্দ।
- `naturalness` (1-5): কতটা মানুষের মতো ও সাবলীল শোনায়। 1 = খুব যান্ত্রিক বা ভাঙা,
  5 = সত্যিকারের মানুষের মতো।
- `dialect_authenticity` (1-5): কতটা একজন স্থানীয় চাটগাঁইয়া ভাষীর মতো শোনায়।
  1 = একেবারেই না (যেমন প্রমিত বাংলা বা বিদেশি টান), 5 = স্থানীয় ভাষীর মতো।
- `dialect_words` (1-5): চাটগাঁইয়া-নির্দিষ্ট শব্দগুলো কি ঠিকভাবে বলা হয়েছে?
  1 = বেশিরভাগ ভুল, 5 = সবগুলো ঠিক।
- `comments` (ঐচ্ছিক): যা লক্ষ করেছেন, যেমন ভুল শব্দ, নয়েজ, গতি।

নম্বরগুলো 1, 2, 3, 4 বা 5 অঙ্কে লিখুন। প্রতিটি ক্লিপ আলাদাভাবে মূল্যায়ন করুন। সবার
রেটিং শেষ না হওয়া পর্যন্ত অন্য রেটারদের সাথে ফাইল বা রেটিং শেয়ার করবেন না।

---
Sentences: "A Synchronized Chittagonian Speech Corpus" (Mendeley Data,
doi:10.17632/n9xttk45df.1), CC BY 4.0.
=== END reports/rater_instructions.md ===
blind clips in /content/drive/MyDrive/boinga-translator/eval/final/blind; key sha256 f48be82e0ca8624a9da31e17520090cadbd444479a05a161418bac2093e1f309
PASS: rating sheet 40 rows
CELL-OK C10.3
```
