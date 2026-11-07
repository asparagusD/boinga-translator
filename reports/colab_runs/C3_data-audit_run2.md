# Colab run C3 run 2 — Index, audit, summary

- File: `C3_data_audit_run2.ipynb` · spec `notebooks/specs/c3_data_audit_run2.py`
- Run: commit `f4c7ab2` · branch `chunk/C3-data-audit` · GPU `Tesla_T4` · time `2026-11-07T03:25:35+00:00`
- Staleness: OK: the run used f4c7ab2, notebook built in f4c7ab2
- Result: 14/14 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C3.2.fetch | Ensure dataset on this VM | PASS |  |
| C3.3 | Index corpus | PASS |  |
| C3.4c | Audit all clips (10–30 min) | PASS |  |
| C3.5 | Audit summary | PASS |  |
| C3.5.play | Listen: 20 random clean clips | PASS |  |
| C3.6 | Pick dialect prompts and synthesise them | PASS |  |
| C3.6.play | Listen: pretrained voices on dialect text | PASS |  |
| C3.5-notes | Your observations — corpus | PASS |  |
| C3.6-notes | Your observations — baseline on dialect text | PASS |  |

## Form values

### C3.5-notes — Your observations — corpus

- `TRANSCRIPTS_MATCH_AUDIO` = "mostly"
- `SOUNDS_CHITTAGONIAN` = "no, standard Bangla"
- `NOISE_LEVEL` = "clean"
- `SPEAKERS_HEARD` = "many"
- `NOTES` = "none"

### C3.6-notes — Your observations — baseline on dialect text

- `HANDLES_DIALECT` = "partly"
- `NOTES` = "none"

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
f4c7ab2 chore(notebook): build C3 run 2 notebook
PASS: /content/boinga-translator on chunk/C3-data-audit
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
git_commit         f4c7ab2
disk_free_gb_root  9.4
disk_free_gb_fast  112.5
utc_time           2026-11-07T03:25:35+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 9.4,
  "git_commit": "f4c7ab2",
  "gpu_mem_gb": 14.6,
  "gpu_name": "Tesla T4",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T03:25:35+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C3 run=2 branch=chunk/C3-data-audit commit=f4c7ab2 gpu=Tesla_T4 utc=2026-11-07T03:25:35+00:00
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
 'git_commit': 'f4c7ab2',
 'disk_free_gb_root': 9.4,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-07T03:25:35+00:00'}
```

### C3.2.fetch — Ensure dataset on this VM

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
CELL-OK C3.2.fetch
```

### C3.3 — Index corpus

```text
rows by split_official:
  train 6403
  val 801
  test 800
  (none) 5
wav_exists True 8004
wav_exists False 5
speakers: no speaker field
sample rows:
  2 train চাটগাঁইয়া ডঅর চাটগাঁ অঞ্চলর আঞ্চলিক ভাষা।
  3 train বাঁআলা ভাষার হোঁয়ারে চাটগাঁইয়া ভাষার বউত শব্দর বেশাবেশি মি
  4 train তয় কেউ কেউ এ ভাষারে ওউগগা স্বাধীন ভাষার মজ্জাদা দিবাল্লাই প
  5 val চাটগাঁইয়া আদতে মুখর ভাষা।
  6 test ইবা নিজর মতো লেকনর কন কুদরত নাই।
PASS: index has 8009 rows (8004 with audio)
CELL-OK C3.3
```

### C3.4c — Audit all clips (10–30 min)

```text
voice female
skipped 0 rows already in /content/drive/MyDrive/boinga-translator/reports/audit.parts (female-a4543ed63b)
audit: 100%|██████████| 8009/8009 [00:47<00:00, 168.69it/s]
wrote 8009 rows to /content/drive/MyDrive/boinga-translator/reports/audit.parts
PASS: audited 8009 clips
CELL-OK C3.4c
```

### C3.5 — Audit summary

```text
=== BEGIN reports/audit_summary.md ===
# Corpus audit summary

Charset: female-a4543ed63b. Rows: 8009; decodable clips: 8004, 9.04 h.

## Per split

| split | clips | hours | clean clips | clean hours |
|---|---|---|---|---|
| (none) | 5 | 0.00 | 0 | 0.00 |
| test | 800 | 0.89 | 299 | 0.42 |
| train | 6403 | 7.26 | 2324 | 3.30 |
| val | 801 | 0.89 | 258 | 0.37 |

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
| oov (oov_ratio > 0.02) | 5015 | 4.74 |
| speech rate (chars/s outside 1st-99th percentile, or no speech) | 167 | 0.13 |
| duplicate text | 28 | 0.02 |

oov only because of the danda । (the model's eos symbol):

| clips |
|---|
| 4711 |

Clean: 2881 clips, 4.10 h.

## Speakers

no speaker field in the corpus (DECISIONS "Corpus layout")

## Characters not in the model (top 20, rows containing)

| char | code | name | rows |
|---|---|---|---|
| । | U+0964 | DEVANAGARI DANDA | 6254 |
| ৰ | U+09F0 | BENGALI LETTER RA WITH MIDDLE DIAGONAL | 59 |
| ৭ | U+09ED | BENGALI DIGIT SEVEN | 50 |
| ৯ | U+09EF | BENGALI DIGIT NINE | 48 |
| " | U+0022 | QUOTATION MARK | 46 |
| ( | U+0028 | LEFT PARENTHESIS | 45 |
| ) | U+0029 | RIGHT PARENTHESIS | 44 |
| ' | U+0027 | APOSTROPHE | 41 |
| – | U+2013 | EN DASH | 41 |
| / | U+002F | SOLIDUS | 19 |
| : | U+003A | COLON | 7 |
| ৷ | U+09F7 | BENGALI CURRENCY NUMERATOR FOUR | 4 |
| “ | U+201C | LEFT DOUBLE QUOTATION MARK | 2 |
| * | U+002A | ASTERISK | 1 |
| ৱ | U+09F1 | BENGALI LETTER RA WITH LOWER DIAGONAL | 1 |
| … | U+2026 | HORIZONTAL ELLIPSIS | 1 |
=== END reports/audit_summary.md ===
PASS: summary written
CELL-OK C3.5
```

### C3.5.play — Listen: 20 random clean clips

```text
AUDIO 7387 [train] CTG: আঁই আনন্দে খাবি খাইবার নান গরি কইলাম , কঁততে  বিয়া ? | BN: আমি আনন্দে খাবি খাওয়ার মতো করে বললাম, কখন বিয়ে?:
<IPython.lib.display.Audio object>
AUDIO 4754 [train] CTG: ২০১২ সালর ১১ ফেব্রুয়ারি রাতিইয়া রাজধানীর পশ্চিম রাজাবাজারত মাছরাঙা টেলিভিশনর বার্তা সম্পাদক সাগর সারওয়ার এবং এটিএন বাংলার জ্যেষ্ঠ প্রতিবেদক রুনি নিজ ভাড়া বাসায় নির্মমভাবে খুন হোয়ই। | BN: ২০১২ সালের ১১ ফেব্রুয়ারি রাতে রাজধানীর পশ্চিম রাজাবাজারে মাছরাঙা টেলিভিশনের বার্তা সম্পাদক সাগর সারওয়ার এবং এটিএন বাংলার জ্যেষ্ঠ প্রতিবেদক রুনি নিজ ভাড়া বাসায় নির্মমভাবে খুন হন।:
<IPython.lib.display.Audio object>
AUDIO 3373 [train] CTG: আঁই কোইলাম, না, গোস্সা গোইজজুম কেয়া? | BN: আমি বললাম, না, রাগ করব কেন?:
<IPython.lib.display.Audio object>
AUDIO 554 [test] CTG: চাইলে আবার তেঝ্যা খতা  কয় | BN: চাইলে আবার তেজে কথা কয়:
<IPython.lib.display.Audio object>
AUDIO 110 [train] CTG: অনে কঅত্তে চিটাং আইস্সন? | BN: আপনি কখন চট্টগ্রামে এসেছেন?:
<IPython.lib.display.Audio object>
AUDIO 6876 [train] CTG: এক্কানা আগেই মুগ্ধ আঁর অঙ্গে  মানুষরে হাসপাতালত লইতসিলো! | BN: একটু আগেই মুগ্ধ আমার সাথে মানুষকে হাসপাতালে নিতেসিলো!:
<IPython.lib.display.Audio object>
AUDIO 4497 [train] CTG: সরফরাজ খাঁ চিন্তিত গলাত কোইয়ী, এত মানুষ থাইকতু মন্ত্রীর হালারে মুখ ভেঙ্গাই দিয়ুন কিল্লাই ? | BN: সরফরাজ খাঁ চিন্তিত গলায় বললেন, এত লোক থাকতে মন্ত্রীর শালাকে মুখ ভেংচি দিলেন কেন?:
<IPython.lib.display.Audio object>
AUDIO 4732 [val] CTG: বেলা সাড়ে ১১টার ইখ্যা চারগো ছাত্র নদীত গোসল গরিবাল্লাই  নামিয়েরে নিখোঁজ হৈয়ি। | BN: বেলা সাড়ে ১১টার দিকে চার ছাত্র নদীতে গোসল করতে নেমে নিখোঁজ হয়।:
<IPython.lib.display.Audio object>
AUDIO 4016 [train] CTG: এতাল্লাই  বিদেশী কোম্পানির ওপর নির্ভর ন গরি আরার তুন বাপেক্সেরে শক্তিশালী গরন পরিবু। | BN: এক্ষেত্রে বিদেশি কোম্পানির ওপর নির্ভরশীল না হয়ে আমাদের বাপেক্সকে শক্তিশালী করতে হবে।:
<IPython.lib.display.Audio object>
AUDIO 2402 [val] CTG: আরেকগো খতা , রেস্টুরেন্টর বয় বাবুইবার অঙ্গে রসিকতা ন গরিবি । | BN: আরেকটা কথা, রেস্টুরেন্টের বয় বাবুর্চির সঙ্গে রসিকতা করবি না।:
<IPython.lib.display.Audio object>
AUDIO 4617 [val] CTG: আঁরে  তুঁই  গরি  কন  কেয়া ? | BN: আমাকে তুমি করে বলছেন কেন?:
<IPython.lib.display.Audio object>
AUDIO 1643 [train] CTG: তারপরও যদি শর্ট পরে রীনা ভাবী নিশ্চয়ই দশ বারোওয়া পরোটা চট চট ভাজি দি দিবু। | BN: তারপরও যদি শর্ট পরে রীনা ভাবী নিশ্চয়ই দশ বারোটা পরোটা চট চট ভেজে দিয়ে দেবে।:
<IPython.lib.display.Audio object>
AUDIO 668 [train] CTG: তুঁই কেন আছো? | BN: তুমি কেমন আছো?:
<IPython.lib.display.Audio object>
AUDIO 1340 [train] CTG: ইবা সন্ত্রাস দমনে বাংলাদেশর লওয়া পদক্ষেপর প্রশংসা গইজ্জি। | BN: তিনি সন্ত্রাস দমনে বাংলাদেশের নেওয়া পদক্ষেপের প্রশংসা করেন।:
<IPython.lib.display.Audio object>
AUDIO 7322 [test] CTG: ইতারার  বেআগরে সামলো-রাখন পরে | BN: এদের সবাইকে সামলো-সুমলে রাখতে হয়।:
<IPython.lib.display.Audio object>
AUDIO 521 [train] CTG: বাইয়ুনত নাই বিচি | BN: বেগুনে নেই বিচি:
<IPython.lib.display.Audio object>
AUDIO 1798 [val] CTG: উগ্গোবার একখানা পাওয়ারর ঠেলা দি ? | BN: একবার একটু পাওয়ারের ধাক্কা দেই?:
<IPython.lib.display.Audio object>
AUDIO 584 [train] CTG: নিয়ন্ত্রন জঅনর শমত্ ইতার মা কইলজে, ও ফুত্ ওউগ্‌গোয়া ফইশা নে। | BN: নিমন্ত্রণে যাওয়ার সময় তার মা তাকে বলল, ও পুত্র, একটি পয়সা নে।:
<IPython.lib.display.Audio object>
AUDIO 6363 [train] CTG: আজিয়াকালিয়ার তরুণঅল ইংরেজি গালি দেয় ভালা  এ হিন্দিত  গালি দের কেয়া বুঝন  ন যায় | BN: আজকালকার তরুণরা ইংরেজি গালি দেয় ভালো এ হিন্দিতে গালি দিচ্ছে কেন বোঝা গেল না।:
<IPython.lib.display.Audio object>
AUDIO 2709 [train] CTG: মজার ইংরেজী জানন? | BN: মজার ইংরেজী জানেন?:
<IPython.lib.display.Audio object>
CELL-OK C3.5.play
```

### C3.6 — Pick dialect prompts and synthesise them

```text
CTG_1 clip 386: অ্যাক্কানা হুইত পাইজ্জম না, ছার? 
CTG_2 clip 29: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি।
CTG_3 clip 625: আগর ডইল্লা দেইন ন ফার
=== BEGIN configs/baseline_prompts.json ===
{
  "STD_1": "আজ সকালে আমি বাজারে গিয়ে কিছু তাজা সবজি কিনেছি।",
  "STD_2": "তুমি কি আগামীকাল বিকেলে আমাদের বাড়িতে চা খেতে আসবে?",
  "STD_3": "বৃষ্টির জন্য আজ শহরের অনেক রাস্তায় পানি জমে গেছে।",
  "CTG_1": "অ্যাক্কানা হুইত পাইজ্জম না, ছার? ",
  "CTG_2": "যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি।",
  "CTG_3": "আগর ডইল্লা দেইন ন ফার",
  "_sources": {
    "CTG_1": "386",
    "CTG_2": "29",
    "CTG_3": "625"
  }
}
=== END configs/baseline_prompts.json ===
PASS: picked 3 dialect prompts
female: model.pth 997.77 MB, config.json 0.01 MB
PASS: female cached (/content/drive/MyDrive/boinga-translator/pretrained/female)
WARNING:TTS.tts.utils.languages:Dataset `` has no language specified.
WARNING:TTS.tts.utils.languages:Could not identify language from config. Initializing with English for text processing.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
WARN: female STD_1: not in model charset: । (U+0964)
female STD_1 4.15 s -22.9 dBFS
female STD_2 4.41 s -20.9 dBFS
WARN: female STD_3: not in model charset: । (U+0964)
female STD_3 4.04 s -19.8 dBFS
female CTG_1 3.25 s -24.6 dBFS
WARN: female CTG_2: not in model charset: । (U+0964)
female CTG_2 5.81 s -22.7 dBFS
female CTG_3 2.30 s -23.2 dBFS
PASS: baseline female (6 wavs in /content/drive/MyDrive/boinga-translator/eval/baseline_smoke/female)
male: model.pth 997.77 MB, config.json 0.01 MB
PASS: male cached (/content/drive/MyDrive/boinga-translator/pretrained/male)
WARNING:TTS.tts.utils.languages:Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
WARN: male STD_1: not in model charset: । (U+0964)
male STD_1 5.13 s -20.2 dBFS
male STD_2 4.48 s -18.1 dBFS
WARN: male STD_3: not in model charset: । (U+0964)
male STD_3 5.46 s -19.4 dBFS
male CTG_1 3.08 s -21.4 dBFS
WARN: male CTG_2: not in model charset: । (U+0964)
male CTG_2 6.56 s -18.2 dBFS
male CTG_3 2.38 s -18.2 dBFS
PASS: baseline male (6 wavs in /content/drive/MyDrive/boinga-translator/eval/baseline_smoke/male)
CELL-OK C3.6
```

### C3.6.play — Listen: pretrained voices on dialect text

```text
AUDIO female CTG_1: অ্যাক্কানা হুইত পাইজ্জম না, ছার? :
<IPython.lib.display.Audio object>
AUDIO female CTG_2: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি।:
<IPython.lib.display.Audio object>
AUDIO female CTG_3: আগর ডইল্লা দেইন ন ফার:
<IPython.lib.display.Audio object>
AUDIO male CTG_1: অ্যাক্কানা হুইত পাইজ্জম না, ছার? :
<IPython.lib.display.Audio object>
AUDIO male CTG_2: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি।:
<IPython.lib.display.Audio object>
AUDIO male CTG_3: আগর ডইল্লা দেইন ন ফার:
<IPython.lib.display.Audio object>
CELL-OK C3.6.play
```

### C3.5-notes — Your observations — corpus

```text
=== BEGIN reports/notes/C3.5-notes.json ===
{
  "NOISE_LEVEL": "clean",
  "NOTES": "none",
  "SOUNDS_CHITTAGONIAN": "no, standard Bangla",
  "SPEAKERS_HEARD": "many",
  "TRANSCRIPTS_MATCH_AUDIO": "mostly"
}
=== END reports/notes/C3.5-notes.json ===
CELL-OK C3.5-notes
```

### C3.6-notes — Your observations — baseline on dialect text

```text
=== BEGIN reports/notes/C3.6-notes.json ===
{
  "HANDLES_DIALECT": "partly",
  "NOTES": "none"
}
=== END reports/notes/C3.6-notes.json ===
CELL-OK C3.6-notes
```
