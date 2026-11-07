# Colab run C2 run 1 — C2 — pretrained model and baseline

- File: `C2_pretrained_baseline_run1.ipynb` · spec `notebooks/specs/c2_pretrained_baseline.py`
- Run: commit `68ab521` · branch `chunk/C2-pretrained-baseline` · GPU `Tesla_T4` · time `2026-11-07T01:15:41+00:00`
- Staleness: OK: the run used 68ab521, notebook built in 68ab521
- Result: 10/10 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C2.1 | Fetch pretrained voices | PASS |  |
| C2.2 | Inspect pretrained voices | PASS |  |
| C2.4 | Baseline smoke synthesis | PASS |  |
| C2.4.play | Listen: baseline audio | PASS |  |
| C2.4-notes | Your observations — baseline | PASS |  |

## Form values

### C2.4-notes — Your observations — baseline

- `FEMALE_STD_INTELLIGIBLE` = "yes"
- `MALE_STD_INTELLIGIBLE` = "yes"
- `ARTEFACTS` = "none"
- `PREFERRED_VOICE_QUALITY` = "same"

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
68ab521 feat(baseline): synthesise smoke prompts and add a listening form
PASS: /content/boinga-translator on chunk/C2-pretrained-baseline
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
git_commit         68ab521
disk_free_gb_root  13.0
disk_free_gb_fast  112.5
utc_time           2026-11-07T01:15:41+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 13.0,
  "git_commit": "68ab521",
  "gpu_mem_gb": 14.6,
  "gpu_name": "Tesla T4",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T01:15:41+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C2 run=1 branch=chunk/C2-pretrained-baseline commit=68ab521 gpu=Tesla_T4 utc=2026-11-07T01:15:41+00:00
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
 'git_commit': '68ab521',
 'disk_free_gb_root': 13.0,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-07T01:15:41+00:00'}
```

### C2.1 — Fetch pretrained voices

```text
female: model.pth 997.77 MB, config.json 0.01 MB
PASS: female cached (/content/drive/MyDrive/boinga-translator/pretrained/female)
male: model.pth 997.77 MB, config.json 0.01 MB
PASS: male cached (/content/drive/MyDrive/boinga-translator/pretrained/male)
CELL-OK C2.1
```

### C2.2 — Inspect pretrained voices

````text
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
থ	U+09A5	BENGALI LETTER THA
ব	U+09AC	BENGALI LETTER BA
ঔ	U+0994	BENGALI LETTER AU
া	U+09BE	BENGALI VOWEL SIGN AA
... [207 lines trimmed] ...
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
CELL-OK C2.2
````

### C2.4 — Baseline smoke synthesis

```text
female: model.pth 997.77 MB, config.json 0.01 MB
PASS: female cached (/content/drive/MyDrive/boinga-translator/pretrained/female)
Dataset `` has no language specified.
Could not identify language from config. Initializing with English for text processing.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
WARN: female STD_1: not in model charset: । (U+0964)
female STD_1 4.15 s -22.9 dBFS
female STD_2 4.41 s -20.9 dBFS
WARN: female STD_3: not in model charset: । (U+0964)
female STD_3 4.04 s -19.8 dBFS
PASS: baseline female (3 wavs in /content/drive/MyDrive/boinga-translator/eval/baseline_smoke/female)
male: model.pth 997.77 MB, config.json 0.01 MB
PASS: male cached (/content/drive/MyDrive/boinga-translator/pretrained/male)
Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
WARN: male STD_1: not in model charset: । (U+0964)
male STD_1 5.13 s -20.2 dBFS
male STD_2 4.48 s -18.1 dBFS
WARN: male STD_3: not in model charset: । (U+0964)
male STD_3 5.46 s -19.4 dBFS
PASS: baseline male (3 wavs in /content/drive/MyDrive/boinga-translator/eval/baseline_smoke/male)
CELL-OK C2.4
```

### C2.4.play — Listen: baseline audio

```text
AUDIO female STD_1: আজ সকালে আমি বাজারে গিয়ে কিছু তাজা সবজি কিনেছি।:
<IPython.lib.display.Audio object>
AUDIO female STD_2: তুমি কি আগামীকাল বিকেলে আমাদের বাড়িতে চা খেতে আসবে?:
<IPython.lib.display.Audio object>
AUDIO female STD_3: বৃষ্টির জন্য আজ শহরের অনেক রাস্তায় পানি জমে গেছে।:
<IPython.lib.display.Audio object>
AUDIO male STD_1: আজ সকালে আমি বাজারে গিয়ে কিছু তাজা সবজি কিনেছি।:
<IPython.lib.display.Audio object>
AUDIO male STD_2: তুমি কি আগামীকাল বিকেলে আমাদের বাড়িতে চা খেতে আসবে?:
<IPython.lib.display.Audio object>
AUDIO male STD_3: বৃষ্টির জন্য আজ শহরের অনেক রাস্তায় পানি জমে গেছে।:
<IPython.lib.display.Audio object>
CELL-OK C2.4.play
```

### C2.4-notes — Your observations — baseline

```text
=== BEGIN reports/notes/C2.4-notes.json ===
{
  "ARTEFACTS": "none",
  "FEMALE_STD_INTELLIGIBLE": "yes",
  "MALE_STD_INTELLIGIBLE": "yes",
  "PREFERRED_VOICE_QUALITY": "same"
}
=== END reports/notes/C2.4-notes.json ===
CELL-OK C2.4-notes
```
