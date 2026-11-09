# Colab run C12 run 1 — boinga-translator — end-to-end

- File: `boinga_translator.ipynb` · spec `notebooks/specs/final.py`
- Run: commit `f12cce7` · branch `chunk/C12-notebook` · GPU `Tesla_T4` · time `2026-11-09T03:31:57+00:00`
- Staleness: OK: the run used f12cce7, notebook built in 8bbe9ba
- Result: 31/31 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| S0 | Settings: RUN_TRAINING | PASS |  |
| S1 | Environment and GPU | PASS |  |
| S2 | Installed versions | PASS |  |
| S3 | Download the corpus | PASS |  |
| S4 | Index the corpus | PASS |  |
| S4.audit | Audit every clip | PASS |  |
| S4.play | Listen: 10 corpus clips | PASS |  |
| S5 | Audit summary | PASS |  |
| S6 | Prepare the training data | PASS |  |
| S6.parse | Coqui parses the manifests | PASS |  |
| S7 | Fetch and inspect the pretrained voices | PASS |  |
| S8 | Baseline synthesis | PASS |  |
| S9 | One-batch loader check | PASS |  |
| S10 | Fresh smoke run | PASS |  |
| S10.train | Smoke test and resume | PASS |  |
| S11 | Fine-tune run1 | PASS |  |
| S12 | Sample every checkpoint | PASS |  |
| S12.play | Listen: checkpoint samples | PASS |  |
| S13 | Rank the checkpoints | PASS |  |
| S14 | Speak held-out text with best/ | PASS |  |
| S14.play | Listen: held-out text | PASS |  |
| S15 | Evaluation audio | PASS |  |
| S15.play | Listen: baseline vs fine-tuned | PASS |  |
| S16 | Automated demo checks | PASS |  |
| S16.live | Live demo (stop the cell when done) | PASS |  |
| S17 | Reproducibility report | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
f12cce7 chore(notebook): build end-to-end notebook for the fresh-runtime test
PASS: /content/boinga-translator on chunk/C12-notebook
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
cpu_count          2
is_colab           True
torch              2.14.1+cu130
torch_cuda         13.0
torch_error        None
gpu_name           Tesla T4
gpu_mem_gb         14.6
coqui_tts          0.28.0
numpy              2.4.6
git_commit         f12cce7
disk_free_gb_root  3.5
disk_free_gb_fast  112.5
utc_time           2026-11-09T03:31:57+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 2,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 3.5,
  "git_commit": "f12cce7",
  "gpu_mem_gb": 14.6,
  "gpu_name": "Tesla T4",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-09T03:31:57+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C12 run=1 branch=chunk/C12-notebook commit=f12cce7 gpu=Tesla_T4 utc=2026-11-09T03:31:57+00:00
{'python': '3.13.16',
 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39',
 'cpu_count': 2,
 'is_colab': True,
 'torch': '2.14.1+cu130',
 'torch_cuda': '13.0',
 'torch_error': None,
 'gpu_name': 'Tesla T4',
 'gpu_mem_gb': 14.6,
 'coqui_tts': '0.28.0',
 'numpy': '2.4.6',
 'git_commit': 'f12cce7',
 'disk_free_gb_root': 3.5,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-09T03:31:57+00:00'}
```

### S0 — Settings: RUN_TRAINING

```text
RUN_TRAINING = False
CELL-OK S0
```

### S1 — Environment and GPU

```text
python             3.13.16
platform           Linux-6.6.159+-x86_64-with-glibc2.39
cpu_count          2
is_colab           True
torch              2.14.1+cu130
torch_cuda         13.0
torch_error        None
gpu_name           Tesla T4
gpu_mem_gb         14.6
coqui_tts          0.28.0
numpy              2.4.6
git_commit         f12cce7
disk_free_gb_root  3.5
disk_free_gb_fast  112.5
utc_time           2026-11-09T03:32:25+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 2,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 3.5,
  "git_commit": "f12cce7",
  "gpu_mem_gb": 14.6,
  "gpu_name": "Tesla T4",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-09T03:32:25+00:00"
}
=== END reports/env.json ===
PASS: env recorded
CELL-OK S1
```

### S2 — Installed versions

```text
coqui-tts 0.28.0
torch 2.14.1+cu130
numpy 2.4.6
gradio 6.31.0
boinga-translator 0.0.1
CELL-OK S2
```

### S3 — Download the corpus

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
copy to fast disk: 100% 8005/8005 [03:52<00:00, 34.47file/s]
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
dataset A Synchronized Chittagonian Speech Corpus for Dialect ASR and Speech-to-Speech Translation
doi 10.17632/n9xttk45df.1, licence CC-BY-4.0
CELL-OK S3
```

### S4 — Index the corpus

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
CELL-OK S4
```

### S4.audit — Audit every clip

```text
voice female
skipped 8009 rows already in /content/drive/MyDrive/boinga-translator/reports/audit.parts (female-a4543ed63b-151ac177e2)
wrote 0 rows to /content/drive/MyDrive/boinga-translator/reports/audit.parts
PASS: audited 8009 clips
CELL-OK S4.audit
```

### S4.play — Listen: 10 corpus clips

```text
AUDIO 3814 [train] CTG: আঁই কোইলাম, মাজেদা খালা চাবি ধরিলি গোইজজুম। | BN: আমি বললাম, মাজেদা খালা চেপে ধরলে করব।:
<IPython.lib.display.Audio object>
AUDIO 6502 [train] CTG: ঘনঘন নিঃশ্বাস ফেলার । | BN: ঘনঘন নিঃশ্বাস ফেলছেন।:
<IPython.lib.display.Audio object>
AUDIO 3205 [val] CTG: বাই হয় তত ফোয়াদ হয়। | BN: বাসি হয় তত স্বাদ হয়।:
<IPython.lib.display.Audio object>
AUDIO 3656 [train] CTG: আঁই এক্ষুনি আইর। | BN: আমি এক্ষুনি আসছি।:
<IPython.lib.display.Audio object>
AUDIO 4631 [train] CTG: ম্যানেজার সাব কোইয়ী , চলন, চশমা কিনি দি । | BN: ম্যানেজার সাহেব বললেন, চলুন, চশমা কিনে দেই।:
<IPython.lib.display.Audio object>
AUDIO 534 [train] CTG: লুসাই পাহাড়তুন লামিয়ারে যারগোই কর্ণফুলী | BN: লুসাই পাহাড় থেকে নেমে বয়ে চলেছে কর্ণফুলী:
<IPython.lib.display.Audio object>
AUDIO 1953 [test] CTG: মেছকাব্দর উগগো চোখে চাই আছে। | BN: মেছকাব্দর এক চোখে তাকিয়ে আছে।:
<IPython.lib.display.Audio object>
AUDIO 2909 [test] CTG: তামান্নার ছোট ভাইইবার নাম কি খালা? | BN: তামান্নার ছোট ভাইটার নাম কি খালা?:
<IPython.lib.display.Audio object>
AUDIO 5948 [test] CTG: অনরার কর্মকাণ্ড শুরু হইবার আগে আঁই কি উগগো  খতা  কইত  পারি? | BN: আপনাদের কর্মকাণ্ড শুরু হবার আগে আমি কি একটা কথা বলতে পারি?:
<IPython.lib.display.Audio object>
AUDIO 7416 [train] CTG: ভদ্রমহিলা কইলেন , তোঁয়ার নাম কি হিমালয়? | BN: ভদ্রমহিলা বললেন, তোমার নাম কি হিমালয়?:
<IPython.lib.display.Audio object>
CELL-OK S4.play
```

### S5 — Audit summary

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
CELL-OK S5
```

### S6 — Prepare the training data

```text
selection: 7506 clips, 0 excluded by char_map
restore: 100% 7512/7512 [00:10<00:00, 705.73file/s]
restored /content/boinga-fast/prepared from /content/drive/MyDrive/boinga-translator/data/prepared/prepared-fd93f4827c6d2b41.tar
transform: 0 clip(s) to do, 7506 already done
0it [00:00, ?it/s]
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
CELL-OK S6
```

### S6.parse — Coqui parses the manifests

```text
train: 6415 samples from metadata_train.csv
val: 338 samples from metadata_val.csv
test: 753 samples from metadata_test.csv
first train sample: text='চাটগাঁইয়া ডঅর চাটগাঁ অঞ্চলর আঞ্চলিক ভাষা.\n' audio_file=/content/boinga-fast/prepared/wavs/2.wav
PASS: coqui parsed 6415 train / 338 val samples (+ 753 test)
CELL-OK S6.parse
```

### S7 — Fetch and inspect the pretrained voices

````text
female: model.pth 997.77 MB, config.json 0.01 MB
PASS: female cached (/content/drive/MyDrive/boinga-translator/pretrained/female)
male: model.pth 997.77 MB, config.json 0.01 MB
PASS: male cached (/content/drive/MyDrive/boinga-translator/pretrained/male)
=== BEGIN reports/pretrained_charset_female.txt ===
ই	U+0987	BENGALI LETTER I
গ	U+0997	BENGALI LETTER GA
ং	U+0982	BENGALI SIGN ANUSVARA
়	U+09BC	BENGALI SIGN NUKTA
’	U+2019	RIGHT SINGLE QUOTATION MARK
ু	U+09C1	BENGALI VOWEL SIGN U
ঃ	U+0983	BENGALI SIGN VISARGA
ন	U+09A8	BENGALI LETTER NA
১	U+09E7	BENGALI DIGIT ONE
ঝ	U+099D	BENGALI LETTER JHA
ূ	U+09C2	BENGALI VOWEL SIGN UU
ও	U+0993	BENGALI LETTER O
‘	U+2018	LEFT SINGLE QUOTATION MARK
ঊ	U+098A	BENGALI LETTER UU
ো	U+09CB	BENGALI VOWEL SIGN O
ছ	U+099B	BENGALI LETTER CHA
প	U+09AA	BENGALI LETTER PA
ফ	U+09AB	BENGALI LETTER PHA
ৈ	U+09C8	BENGALI VOWEL SIGN AI
৮	U+09EE	BENGALI DIGIT EIGHT
ষ	U+09B7	BENGALI LETTER SSA
য	U+09AF	BENGALI LETTER YA
ৎ	U+09CE	BENGALI LETTER KHANDA TA
ঢ	U+09A2	BENGALI LETTER DDHA
ঈ	U+0988	BENGALI LETTER II
ক	U+0995	BENGALI LETTER KA
ঠ	U+09A0	BENGALI LETTER TTHA
ি	U+09BF	BENGALI VOWEL SIGN I
জ	U+099C	BENGALI LETTER JA
০	U+09E6	BENGALI DIGIT ZERO
৬	U+09EC	BENGALI DIGIT SIX
ী	U+09C0	BENGALI VOWEL SIGN II
ট	U+099F	BENGALI LETTER TTA
ড	U+09A1	BENGALI LETTER DDA
এ	U+098F	BENGALI LETTER E
অ	U+0985	BENGALI LETTER A
ঋ	U+098B	BENGALI LETTER VOCALIC R
ধ	U+09A7	BENGALI LETTER DHA
চ	U+099A	BENGALI LETTER CA
ে	U+09C7	BENGALI VOWEL SIGN E
২	U+09E8	BENGALI DIGIT TWO
৩	U+09E9	BENGALI DIGIT THREE
ণ	U+09A3	BENGALI LETTER NNA
উ	U+0989	BENGALI LETTER U
য়	U+09DF	BENGALI LETTER YYA
ঢ়	U+09DD	BENGALI LETTER RHA
খ	U+0996	BENGALI LETTER KHA
ল	U+09B2	BENGALI LETTER LA
ভ	U+09AD	BENGALI LETTER BHA
ৗ	U+09D7	BENGALI AU LENGTH MARK
স	U+09B8	BENGALI LETTER SA
হ	U+09B9	BENGALI LETTER HA
্	U+09CD	BENGALI SIGN VIRAMA
ড়	U+09DC	BENGALI LETTER RRA
দ	U+09A6	BENGALI LETTER DA
... [219 lines trimmed] ...
=== END reports/pretrained_charset_female.txt ===
WARN: female: config output_path is an absolute path: /home/ansary/Shabab/
WARN: female: config datasets.0.path is an absolute path: /home/ansary/Shabab/bn_tts/female/mono/
WARN: female: config datasets.0.meta_file_train is an absolute path: /home/ansary/Shabab/bn_tts/female/mono/metadata_female.txt
WARN: female: config restore_path is an absolute path: /home/ansary/Shabab/vits_4_nov/checkpoint_571000.pth
PASS: inspected female
=== BEGIN reports/pretrained_charset_male.txt ===
=== END reports/pretrained_charset_male.txt ===
WARN: male: config output_path is an absolute path: /home/ansary/Shabab/
WARN: male: config datasets.0.path is an absolute path: /home/ansary/Shabab/bn_tts/male/mono/
WARN: male: config datasets.0.meta_file_train is an absolute path: /home/ansary/Shabab/bn_tts/male/mono/metadata_male.txt
WARN: male: config restore_path is an absolute path: /home/ansary/Shabab/vits_20_october/checkpoint_626000.pth
PASS: inspected male
=== BEGIN configs/pretrained_facts.json ===
=== END configs/pretrained_facts.json ===
=== BEGIN reports/pretrained_inspect.md ===

```json
{
  "checkpoint": {
    "has_discriminator": true,
    "n_disc_keys": 111,
    "n_disc_params": 46747132,
    "n_params": 83050732,
    "step": 910000,
    "top_level_keys": [
      "config",
      "date",
      "epoch",
      "model",
      "model_loss",
      "optimizer",
      "scaler",
      "step"
    ]
  },
  "config": {
    "absolute_paths": {
      "datasets.0.meta_file_train": "/home/ansary/Shabab/bn_tts/male/mono/metadata_male.txt",
      "datasets.0.path": "/home/ansary/Shabab/bn_tts/male/mono/",
      "output_path": "/home/ansary/Shabab/",
      "restore_path": "/home/ansary/Shabab/vits_20_october/checkpoint_626000.pth"
    },
    "add_blank": true,
    "audio": {
      "fft_size": 1024,
      "hop_length": 256,
      "mel_fmax": null,
      "mel_fmin": 0,
      "num_mels": 80,
      "sample_rate": 22050,
      "win_length": 1024
    },
    "characters": {
      "blank": "<BLNK>",
      "bos": "<BOS>",
      "characters": "তট৫ভিঐঋখঊড়ইজমএেঘঙসীঢ়হঞ‘ঈকণ৬ঁৗশঢঠ‌১্২৮দৃঔগও—ছউংবৈঝাযফ‍চরষঅৌৎথড়৪ধ০ুূ৩আঃপয়’নলো",
      "characters_class": "TTS.tts.utils.text.characters.Graphemes",
      "eos": "।",
      "pad": "<PAD>",
      "phonemes": null,
      "punctuations": "-!,|.? "
    },
    "enable_eos_bos_chars": false,
    "model": "vits",
    "num_speakers": 0,
    "phoneme_language": "bn",
    "phonemizer": null,
    "text_cleaner": null,
    "use_phonemes": false,
    "use_speaker_embedding": false
  }
}
```
=== END reports/pretrained_inspect.md ===
CELL-OK S7
````

### S8 — Baseline synthesis

```text
female: model.pth 997.77 MB, config.json 0.01 MB
PASS: female cached (/content/drive/MyDrive/boinga-translator/pretrained/female)
Dataset `` has no language specified.
Could not identify language from config. Initializing with English for text processing.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
baseline female:   0% 0/6 [00:00<?, ?prompt/s]female STD_1 4.40 s -25.5 dBFS
baseline female:  17% 1/6 [00:03<00:19,  3.90s/prompt]female STD_2 4.89 s -17.7 dBFS
baseline female:  33% 2/6 [00:05<00:09,  2.33s/prompt]female STD_3 4.54 s -20.5 dBFS
baseline female:  50% 3/6 [00:06<00:05,  1.78s/prompt]female CTG_1 3.25 s -24.6 dBFS
baseline female:  67% 4/6 [00:07<00:02,  1.48s/prompt]female CTG_2 5.76 s -21.7 dBFS
baseline female:  83% 5/6 [00:08<00:01,  1.31s/prompt]female CTG_3 2.30 s -23.2 dBFS
baseline female: 100% 6/6 [00:09<00:00,  1.53s/prompt]
PASS: baseline female (6 wavs in /content/drive/MyDrive/boinga-translator/eval/baseline_smoke/female)
male: model.pth 997.77 MB, config.json 0.01 MB
PASS: male cached (/content/drive/MyDrive/boinga-translator/pretrained/male)
Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
baseline male:   0% 0/6 [00:00<?, ?prompt/s]male STD_1 4.73 s -19.2 dBFS
baseline male:  17% 1/6 [00:01<00:05,  1.14s/prompt]male STD_2 4.40 s -18.1 dBFS
baseline male:  33% 2/6 [00:02<00:04,  1.09s/prompt]male STD_3 5.00 s -19.6 dBFS
baseline male:  50% 3/6 [00:03<00:03,  1.12s/prompt]male CTG_1 3.08 s -21.4 dBFS
baseline male:  67% 4/6 [00:04<00:02,  1.05s/prompt]male CTG_2 6.76 s -16.9 dBFS
baseline male:  83% 5/6 [00:05<00:01,  1.06s/prompt]male CTG_3 2.38 s -18.2 dBFS
baseline male: 100% 6/6 [00:06<00:00,  1.04s/prompt]
PASS: baseline male (6 wavs in /content/drive/MyDrive/boinga-translator/eval/baseline_smoke/male)
CELL-OK S8
```

### S9 — One-batch loader check

```text
get_data_loader(config: coqpit.coqpit.Coqpit, *, is_eval: bool = False, samples: list[typing.Any] | None = None, verbose: bool = True) -> 'DataLoader'
samples: 6415 train / 338 eval
100% 6415/6415 [00:07<00:00, 862.62it/s]
/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:431: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:437: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
আইয়ী .

অইল্লে এরি.

Character '\n' not found in the vocabulary. Discarding it.
ইয়ান ঠিক.

স্টার্ট ন লর.

আঁই ন খাই .

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
সরবত খন, খন.

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
বোথ ভালা মাইয়া.

Character '\n' not found in the vocabulary. Discarding it.
পরটা ন.

Character '\n' not found in the vocabulary. Discarding it.
first batch (incl. worker startup + spec/mel, not a training speed) in 35.26 s
tokens: shape (16, 29) int64, ids 0..86 of vocab 87
token_lens: 13..29
token_rel_lens: shape (16,) float32, nan 0, inf 0
waveform: shape (16, 1, 21306) float32, nan 0, inf 0
waveform_lens: 17210..21306 samples, seconds: 0.78..0.97
waveform_rel_lens: shape (16,) float32, nan 0, inf 0
speaker_names: list of 16
language_names: list of 16
audio_files: list of 16
raw_text: list of 16
audio_unique_names: list of 16
spec: shape (16, 513, 83) float32, nan 0, inf 0
mel: shape (16, 80, 83) float32, nan 0, inf 0
spec_lens: shape (16,) int32
mel_lens: shape (16,) int32
PASS: loader ok
CELL-OK S9
```

### S10 — Fresh smoke run

```text
SKIPPED (RUN_TRAINING=False)
CELL-OK S10
```

### S10.train — Smoke test and resume

```text
SKIPPED (RUN_TRAINING=False)
CELL-OK S10.train
```

### S11 — Fine-tune run1

```text
SKIPPED (RUN_TRAINING=False)
CELL-OK S11
```

### S12 — Sample every checkpoint

```text
SKIPPED (RUN_TRAINING=False)
CELL-OK S12
```

### S12.play — Listen: checkpoint samples

```text
SKIPPED (RUN_TRAINING=False)
CELL-OK S12.play
```

### S13 — Rank the checkpoints

```text
SKIPPED (RUN_TRAINING=False)
CELL-OK S13
```

### S14 — Speak held-out text with best/

```text
reload check:   0% 0/3 [00:00<?, ?wav/s]synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
0_6.wav: 2.94 s, RMS -17.3 dBFS, ok: ইবা নিজর মতো লেকনর কন কুদরত নাই.
reload check:  33% 1/3 [00:38<01:16, 38.09s/wav]1_29.wav: 5.23 s, RMS -18.8 dBFS, ok: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি.
reload check:  67% 2/3 [00:39<00:16, 16.37s/wav]2_58.wav: 2.71 s, RMS -21.7 dBFS, ok: ইত্তিরি হাঁজুইন্না সাততো.
reload check: 100% 3/3 [00:40<00:00, 13.44s/wav]
PASS: reload check (3 wavs)
CELL-OK S14
```

### S14.play — Listen: held-out text

```text
AUDIO #0 6: ইবা নিজর মতো লেকনর কন কুদরত নাই.:
<IPython.lib.display.Audio object>
AUDIO #1 29: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি.:
<IPython.lib.display.Audio object>
AUDIO #2 58: ইত্তিরি হাঁজুইন্না সাততো.:
<IPython.lib.display.Audio object>
CELL-OK S14.play
```

### S15 — Evaluation audio

```text
eval audio:   0% 0/20 [00:00<?, ?prompt/s]eval audio: 100% 20/20 [00:00<00:00, 201649.23prompt/s]
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
CELL-OK S15
```

### S15.play — Listen: baseline vs fine-tuned

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
AUDIO P04 A_baseline: গত্ ত্থুন্ বাইর গরি দন পরে ।:
<IPython.lib.display.Audio object>
AUDIO P04 B_boinga: গত্ ত্থুন্ বাইর গরি দন পরে ।:
<IPython.lib.display.Audio object>
AUDIO P04 R_real: গত্ ত্থুন্ বাইর গরি দন পরে ।:
<IPython.lib.display.Audio object>
AUDIO P05 A_baseline: ইতা চাইদ্দি বোথ  শরম পার ।:
<IPython.lib.display.Audio object>
AUDIO P05 B_boinga: ইতা চাইদ্দি বোথ  শরম পার ।:
<IPython.lib.display.Audio object>
AUDIO P05 R_real: ইতা চাইদ্দি বোথ  শরম পার ।:
<IPython.lib.display.Audio object>
CELL-OK S15.play
```

### S16 — Automated demo checks

```text
demo selfcheck:   0% 0/6 [00:00<?, ?case/s]synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
CASE normal-boinga: ok 3.21 s
demo selfcheck:  17% 1/6 [00:12<01:00, 12.01s/case]Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
CASE normal-baseline: ok 3.08 s
demo selfcheck:  33% 2/6 [00:15<00:27,  6.98s/case]CASE empty: error Please enter some Bengali text.
CASE too-long: error Please keep it under 300 characters.
CASE latin-mixed: error The voice cannot say these characters, please remove them: h, e, l, o, w, r, d
CASE bengali-digits: ok 5.52 s
demo selfcheck: 100% 6/6 [00:15<00:00,  2.61s/case]
PASS: demo selfcheck (6 cases)
CELL-OK S16
```

### S16.live — Live demo (stop the cell when done)

```text
Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().
* Running on public URL: https://2400e85f06afead92e.gradio.live

This share link is temporary and will last for up to 1 week (best effort). For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
<IPython.core.display.HTML object>
Keyboard interruption in main thread... closing server.
Killing tunnel 127.0.0.1:7860 <> https://2400e85f06afead92e.gradio.live
CELL-OK S16.live
```

### S17 — Reproducibility report

```text
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 2,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 3.5,
  "git_commit": "f12cce7",
  "gpu_mem_gb": 14.6,
  "gpu_name": "Tesla T4",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-09T03:32:25+00:00"
}
=== END reports/env.json ===
CELL-OK S17
```
