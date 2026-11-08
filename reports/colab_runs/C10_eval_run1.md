# Colab run C10 run 1 — prompt review

- File: `C10_eval_run1.ipynb` · spec `notebooks/specs/c10_eval_run1.py`
- Run: commit `b9d78c1` · branch `chunk/C10-eval` · GPU `none` · time `2026-11-08T20:42:02+00:00`
- Staleness: OK: the run used b9d78c1, notebook built in b9d78c1
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
| C10.1 | Pick evaluation prompts | PASS |  |
| C10.1.show | Show prompts | PASS |  |
| C10.1-notes | Prompt review (with a Chittagonian speaker) | PASS |  |

## Form values

### C10.1-notes — Prompt review (with a Chittagonian speaker)

- `REVIEWED_BY` = "Me"
- `IDS_TO_REPLACE` = "None"
- `NOTES` = "All okay"

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
b9d78c1 fix(eval): take all 20 prompts from our test split
PASS: /content/boinga-translator on chunk/C10-eval
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
PASS: 13 project folders ready under /content/drive/MyDrive/boinga-translator
```

### HDR — Run header

```text
python             3.13.18
platform           Linux-6.6.159+-x86_64-with-glibc2.39
cpu_count          2
is_colab           True
torch              2.14.1+cpu
torch_cuda         None
torch_error        None
gpu_name           none
gpu_mem_gb         None
coqui_tts          0.28.0
numpy              2.4.6
git_commit         b9d78c1
disk_free_gb_root  3.6
disk_free_gb_fast  107.5
utc_time           2026-11-08T20:42:02+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 2,
  "disk_free_gb_fast": 107.5,
  "disk_free_gb_root": 3.6,
  "git_commit": "b9d78c1",
  "gpu_mem_gb": null,
  "gpu_name": "none",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.18",
  "torch": "2.14.1+cpu",
  "torch_cuda": null,
  "torch_error": null,
  "utc_time": "2026-11-08T20:42:02+00:00"
}
=== END reports/env.json ===
PASS: env recorded
WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run needs one
RUN-HEADER chunk=C10 run=1 branch=chunk/C10-eval commit=b9d78c1 gpu=none utc=2026-11-08T20:42:02+00:00
{'python': '3.13.18',
 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39',
 'cpu_count': 2,
 'is_colab': True,
 'torch': '2.14.1+cpu',
 'torch_cuda': None,
 'torch_error': None,
 'gpu_name': 'none',
 'gpu_mem_gb': None,
 'coqui_tts': '0.28.0',
 'numpy': '2.4.6',
 'git_commit': 'b9d78c1',
 'disk_free_gb_root': 3.6,
 'disk_free_gb_fast': 107.5,
 'utc_time': '2026-11-08T20:42:02+00:00'}
```

### C10.pre — Restore data

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
copy to fast disk: 100% 8005/8005 [03:19<00:00, 40.07file/s] 
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
selection: 7506 clips, 0 excluded by char_map
restore: 100% 7512/7512 [00:11<00:00, 681.23file/s]
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
CELL-OK C10.pre
```

### C10.1 — Pick evaluation prompts

```text
P01 clip 386: অ্যাক্কানা হুইত পাইজ্জম না, ছার? 
P02 clip 1311: হেঁততে উগগো পোয়া ছুটি আইসসি।
P03 clip 5358: অনরে  ফেলাই  ধান পরিবু বুলি  ন ধাই ।
P04 clip 1874: গত্ ত্থুন্ বাইর গরি দন পরে ।
P05 clip 3312: ইতা চাইদ্দি বোথ  শরম পার ।
P06 clip 1729: ইতি আঁআর মিক্কা চাঁই রইয়্যে।
P07 clip 6151: অনে  অতি হাছা খতা  কইয়ুন।
P08 clip 29: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি।
P09 clip 3022: উগ্গোবার কোইলাম দেহা ন গোইজজুম –দেহা ন  গোইজজুম ।
P10 clip 6: ইবা নিজর মতো লেকনর কন কুদরত নাই।
P11 clip 2367: তুঁই থানাত কিল্লাই আইসসো হিবা ক।
P12 clip 1253: এক্কান জিনিস পুছার গরন আছিল।
P13 clip 625: আগর ডইল্লা দেইন ন ফার
P14 clip 5074: ইবা বোথ বিরক্ত ভাবে কোইয়ী , এহনও ন যস ?
P15 clip 4235: অনে নিজেই তো জন গৈ।
P16 clip 7420: ইবা  জীপ ত্থুন্  নাইম্মি ।
P17 clip 1071: হেডে আঁই পাশ গোইজজুম কেনে।
P18 clip 2985: আঁই এহন বাইর হোইর ।
P19 clip 6850: ইবারে চাই  অনে  চাবা  গলাত  কইয়ুন, এতদিন কডে  আছিল ?
P20 clip 5570: আঁই  কইলাম, তুঁই  কি আঁর খতা ইয়ান কইয়ুস?
=== BEGIN configs/eval_prompts.json ===
[
  {
    "id": "P01",
    "clip_id": "386",
    "ctg_text": "অ্যাক্কানা হুইত পাইজ্জম না, ছার? ",
    "bn_text": "একটু শুতে পারব, স্যার?",
    "has_reference": true
  },
  {
    "id": "P02",
    "clip_id": "1311",
    "ctg_text": "হেঁততে উগগো পোয়া ছুটি আইসসি।",
    "bn_text": "তখন একটা ছেলে ছুটে এল।",
    "has_reference": true
  },
  {
    "id": "P03",
    "clip_id": "5358",
    "ctg_text": "অনরে  ফেলাই  ধান পরিবু বুলি  ন ধাই ।",
    "bn_text": "আপনাকে ফেলে পালাতে হয় বলে পালাই নি।",
    "has_reference": true
  },
  {
    "id": "P04",
    "clip_id": "1874",
    "ctg_text": "গত্ ত্থুন্ বাইর গরি দন পরে ।",
    "bn_text": "শরীর থেকে বের করে দিতে হয়।",
    "has_reference": true
  },
  {
    "id": "P05",
    "clip_id": "3312",
    "ctg_text": "ইতা চাইদ্দি বোথ  শরম পার ।",
    "bn_text": "ও দেখি খুবই লজ্জা পাচ্ছে।",
    "has_reference": true
  },
  {
    "id": "P06",
    "clip_id": "1729",
... [46 lines trimmed] ...
  {
    "id": "P13",
    "clip_id": "625",
    "ctg_text": "আগর ডইল্লা দেইন ন ফার",
    "bn_text": "আগের মতো দেখতে পারে না",
    "has_reference": true
  },
  {
    "id": "P14",
    "clip_id": "5074",
    "ctg_text": "ইবা বোথ বিরক্ত ভাবে কোইয়ী , এহনও ন যস ?",
    "bn_text": "তিনি খুবই বিরক্ত ভঙ্গিতে বললেন, এখনো যাও নি?",
    "has_reference": true
  },
  {
    "id": "P15",
    "clip_id": "4235",
    "ctg_text": "অনে নিজেই তো জন গৈ।",
    "bn_text": "আপনি তো নিজেই যাচ্ছেন।",
    "has_reference": true
  },
  {
    "id": "P16",
    "clip_id": "7420",
    "ctg_text": "ইবা  জীপ ত্থুন্  নাইম্মি ।",
    "bn_text": "তিনি জীপ থেকে নামলেন।",
    "has_reference": true
  },
  {
    "id": "P17",
    "clip_id": "1071",
    "ctg_text": "হেডে আঁই পাশ গোইজজুম কেনে।",
    "bn_text": "সেখানে আমি পাশ করব কিভাবে।",
    "has_reference": true
  },
  {
    "id": "P18",
    "clip_id": "2985",
    "ctg_text": "আঁই এহন বাইর হোইর ।",
    "bn_text": "আমি এখন বেরোচ্ছি।",
    "has_reference": true
  },
  {
    "id": "P19",
    "clip_id": "6850",
    "ctg_text": "ইবারে চাই  অনে  চাবা  গলাত  কইয়ুন, এতদিন কডে  আছিল ?",
    "bn_text": "উনাকে দেখে আপনি চাপা গলায় বলবেন–এতদিন কোথায় ছিলে?",
    "has_reference": true
  },
  {
    "id": "P20",
    "clip_id": "5570",
    "ctg_text": "আঁই  কইলাম, তুঁই  কি আঁর খতা ইয়ান কইয়ুস?",
    "bn_text": "আমি বললাম, তুমি কি আমার কথাটা বলেছ?",
    "has_reference": true
  }
]
=== END configs/eval_prompts.json ===
PASS: 20 prompts
CELL-OK C10.1
```

### C10.1.show — Show prompts

```text
P01 (clip 386)
  ctg: অ্যাক্কানা হুইত পাইজ্জম না, ছার? 
  bn:  একটু শুতে পারব, স্যার?
P02 (clip 1311)
  ctg: হেঁততে উগগো পোয়া ছুটি আইসসি।
  bn:  তখন একটা ছেলে ছুটে এল।
P03 (clip 5358)
  ctg: অনরে  ফেলাই  ধান পরিবু বুলি  ন ধাই ।
  bn:  আপনাকে ফেলে পালাতে হয় বলে পালাই নি।
P04 (clip 1874)
  ctg: গত্ ত্থুন্ বাইর গরি দন পরে ।
  bn:  শরীর থেকে বের করে দিতে হয়।
P05 (clip 3312)
  ctg: ইতা চাইদ্দি বোথ  শরম পার ।
  bn:  ও দেখি খুবই লজ্জা পাচ্ছে।
P06 (clip 1729)
  ctg: ইতি আঁআর মিক্কা চাঁই রইয়্যে।
  bn:  সে তাকিয়ে আছে আমার দিকে।
P07 (clip 6151)
  ctg: অনে  অতি হাছা খতা  কইয়ুন।
  bn:  আপনি অতি সত্য কথা বলেছেন।
P08 (clip 29)
  ctg: যাঁরা বাঁআলা ন জানে, তারার বুজিবার সুবিদাল্লাই রোমান অক্কর বেবহার গঞ্জি।
  bn:  যাঁরা বাংলা জানেন না তাদের জন্য চাটগাঁইয়া পাঠের উচ্চারণ নির্দেশে ব্যবহৃত হয়েছে রোমান হরফ।
P09 (clip 3022)
  ctg: উগ্গোবার কোইলাম দেহা ন গোইজজুম –দেহা ন  গোইজজুম ।
  bn:  একবার বলেছি দেখা করব না –দেখা করব না।
P10 (clip 6)
  ctg: ইবা নিজর মতো লেকনর কন কুদরত নাই।
  bn:  এর নিজস্ব লিপি পদ্ধতি নেই।
P11 (clip 2367)
  ctg: তুঁই থানাত কিল্লাই আইসসো হিবা ক।
  bn:  তুমি থানায় এসেছ কেন সেটা বলো।
P12 (clip 1253)
  ctg: এক্কান জিনিস পুছার গরন আছিল।
  bn:  একটা ছোট্ট জিজ্ঞাসা ছিল।
P13 (clip 625)
  ctg: আগর ডইল্লা দেইন ন ফার
  bn:  আগের মতো দেখতে পারে না
P14 (clip 5074)
  ctg: ইবা বোথ বিরক্ত ভাবে কোইয়ী , এহনও ন যস ?
  bn:  তিনি খুবই বিরক্ত ভঙ্গিতে বললেন, এখনো যাও নি?
P15 (clip 4235)
  ctg: অনে নিজেই তো জন গৈ।
  bn:  আপনি তো নিজেই যাচ্ছেন।
P16 (clip 7420)
  ctg: ইবা  জীপ ত্থুন্  নাইম্মি ।
  bn:  তিনি জীপ থেকে নামলেন।
P17 (clip 1071)
  ctg: হেডে আঁই পাশ গোইজজুম কেনে।
  bn:  সেখানে আমি পাশ করব কিভাবে।
P18 (clip 2985)
  ctg: আঁই এহন বাইর হোইর ।
  bn:  আমি এখন বেরোচ্ছি।
P19 (clip 6850)
  ctg: ইবারে চাই  অনে  চাবা  গলাত  কইয়ুন, এতদিন কডে  আছিল ?
  bn:  উনাকে দেখে আপনি চাপা গলায় বলবেন–এতদিন কোথায় ছিলে?
P20 (clip 5570)
  ctg: আঁই  কইলাম, তুঁই  কি আঁর খতা ইয়ান কইয়ুস?
  bn:  আমি বললাম, তুমি কি আমার কথাটা বলেছ?
CELL-OK C10.1.show
```

### C10.1-notes — Prompt review (with a Chittagonian speaker)

```text
=== BEGIN reports/notes/C10.1-notes.json ===
{
  "IDS_TO_REPLACE": "None",
  "NOTES": "All okay",
  "REVIEWED_BY": "Me"
}
=== END reports/notes/C10.1-notes.json ===
CELL-OK C10.1-notes
```
