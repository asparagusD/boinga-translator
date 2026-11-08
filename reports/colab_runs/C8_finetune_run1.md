# Colab run C8 run 1 — C8 run 1 — fine-tune

- File: `C8_finetune_run1.ipynb` · spec `notebooks/specs/c8_finetune_run1.py`
- Run: commit `853ac3a` · branch `chunk/C8-finetune` · GPU `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` · time `2026-11-08T05:22:40+00:00`
- Staleness: OK: the run used 853ac3a, notebook built in 853ac3a
- Result: 15/15 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C8.pre | Restore prepared data | PASS |  |
| C8.tb | TensorBoard (optional) | PASS |  |
| C8.train | Train run1 to its budget (hours; resumable) | PASS |  |
| C8.thr | Throughput of run1 | PASS |  |
| C8.rec | Record the checkpoints | PASS |  |
| C8.flush | Check they really reached Drive | PASS |  |
| C8.sample | Sample every checkpoint | PASS |  |
| C8.cand | Rank checkpoints | PASS |  |
| C8.play | Listen: pretrained vs the checkpoints | PASS |  |
| C8-notes | Your observations — run 1 | PASS |  |

## Form values

### C8-notes — Your observations — run 1

- `BEST_STEP` = "36500"
- `VERDICT` = "good: promote BEST_STEP"
- `CLEARER_THAN_PRETRAINED` = "yes"
- `MORE_CHITTAGONIAN_THAN_PRETRAINED` = "yes"
- `ARTEFACTS` = "none"
- `NOTES` = "none"

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
853ac3a feat(eval): record run checkpoints on the VM and check them on Drive
PASS: /content/boinga-translator on chunk/C8-finetune
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
git_commit         853ac3a
disk_free_gb_root  10.1
disk_free_gb_fast  112.5
utc_time           2026-11-08T05:22:40+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 48,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 10.1,
  "git_commit": "853ac3a",
  "gpu_mem_gb": 95.0,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-08T05:22:40+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C8 run=1 branch=chunk/C8-finetune commit=853ac3a gpu=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition utc=2026-11-08T05:22:40+00:00
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
 'git_commit': '853ac3a',
 'disk_free_gb_root': 10.1,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-08T05:22:40+00:00'}
```

### C8.pre — Restore prepared data

```text
Dataset.zip: 100% 844M/844M [00:24<00:00, 34.9MB/s]
DONE download
DONE verify
unzip Dataset.zip: 100% 8010/8010 [00:28<00:00, 277.73file/s]
DONE unzip
copy to fast disk: 100% 8005/8005 [00:05<00:00, 1338.04file/s]
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
selection: 7506 clips, 0 excluded by char_map
restore: 100% 7512/7512 [00:02<00:00, 3087.82file/s]
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
CELL-OK C8.pre
```

### C8.tb — TensorBoard (optional)

```text
<IPython.core.display.Javascript object>
CELL-OK C8.tb
```

### C8.train — Train run1 to its budget (hours; resumable)

```text
Streaming output truncated to the last 5000 lines.
     | > step_time: 0.1504  (0.1425951690365361)
     | > loader_time: 0.0034  (0.002544482763244134)


   --> TIME: 2026-11-08 06:58:35 -- STEP: 396/400 -- GLOBAL_STEP: 32075
     | > current_lr_0: 1.814018552729449e-06  (1.859552237168974e-06)
     | > current_lr_1: 1.8185595568198018e-06  (1.8642072250148848e-06)
     | > loss_disc: 2.3192687034606934  (2.2770669661989102)
     | > loss_disc_real_0: 0.1739700436592102  (0.153103877686792)
     | > loss_disc_real_1: 0.1950177103281021  (0.18634674938941242)
     | > loss_disc_real_2: 0.22584223747253418  (0.2163780623614186)
     | > loss_disc_real_3: 0.22744540870189667  (0.21966583266703768)
     | > loss_disc_real_4: 0.2244511991739273  (0.213817770771607)
     | > loss_disc_real_5: 0.21379737555980682  (0.21859015388922257)
     | > loss_0: 2.3192687034606934  (2.2770669661989102)
     | > grad_norm_0: tensor(15.0733, device='cuda:0')  (tensor(15.3375, device='cuda:0'))
     | > loss_gen: 2.489933967590332  (2.5200221424150913)
     | > loss_kl: 2.2274041175842285  (2.2219028843171675)
     | > loss_feat: 13.737055778503418  (13.94014489530313)
     | > loss_mel: 20.385875701904297  (20.888734514063056)
     | > loss_duration: 1.7947206497192383  (1.8595370018120965)
     | > amp_scaler: 512.0  (512.0)
     | > loss_1: 40.634986877441406  (41.43034157608496)
     | > grad_norm_1: tensor(94.9868, device='cuda:0')  (tensor(123.0415, device='cuda:0'))
     | > step_time: 0.1737  (0.14338820931887378)
     | > loader_time: 0.0044  (0.0026227801737159174)


 > EVALUATION 

দেরি হন বরং ভালা.

Character '\n' not found in the vocabulary. Discarding it.
পইরতু ইচ্ছা ন গরের .

Character '\n' not found in the vocabulary. Discarding it.
লই তো আইসসি.

ওঁশিয়ের!

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.

  --> EVAL PERFORMANCE
     | > avg_loader_time: 0.0018350623902820406 (-2.8326397850399936e-05)
     | > avg_loss_disc: 2.328086960883367 (+0.02631495680127749)
     | > avg_loss_disc_real_0: 0.14906744144502138 (+0.00591178309349788)
     | > avg_loss_disc_real_1: 0.18499132458652767 (+0.004204818890208267)
     | > avg_loss_disc_real_2: 0.20797955422174363 (+0.006181270238899056)
     | > avg_loss_disc_real_3: 0.21061478519723528 (+0.00045944643872122826)
     | > avg_loss_disc_real_4: 0.19983733126095363 (-0.00457755085967837)
     | > avg_loss_disc_real_5: 0.21490390084329106 (+0.015715799161366062)
     | > avg_loss_0: 2.328086960883367 (+0.02631495680127749)
     | > avg_loss_gen: 2.3841426372528076 (+0.012895152682350197)
     | > avg_loss_kl: 2.7368484054292956 (-0.017107668377102758)
     | > avg_loss_feat: 13.696801446733021 (-0.310718797502064)
     | > avg_loss_mel: 20.410486584617974 (-0.4507358641851553)
     | > avg_loss_duration: 1.9048585579508828 (-7.882004692438827e-05)
     | > avg_loss_1: 41.133137475876595 (-0.7657464345296177)
... [4881 lines trimmed] ...
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
দেরি হন বরং ভালা.

Character '\n' not found in the vocabulary. Discarding it.
পইরতু ইচ্ছা ন গরের .

Character '\n' not found in the vocabulary. Discarding it.

  --> EVAL PERFORMANCE
     | > avg_loader_time: 0.0019677309762863885 (+4.406202407110306e-05)
     | > avg_loss_disc: 2.3388339905511764 (+0.030199533417111013)
     | > avg_loss_disc_real_0: 0.14666974615483055 (+0.006011280807710767)
     | > avg_loss_disc_real_1: 0.18380713285434813 (+0.0035017356276512424)
     | > avg_loss_disc_real_2: 0.20587363165049327 (+0.0008904270472980813)
     | > avg_loss_disc_real_3: 0.2112175804518518 (+0.002502549617063421)
     | > avg_loss_disc_real_4: 0.19854305258819036 (+0.008261733111881081)
     | > avg_loss_disc_real_5: 0.21392134860867545 (+0.008033709866659966)
     | > avg_loss_0: 2.3388339905511764 (+0.030199533417111013)
     | > avg_loss_gen: 2.357088429587228 (-0.005856224468775206)
     | > avg_loss_kl: 2.6800207495689388 (+0.01158393848509931)
     | > avg_loss_feat: 14.017990521022252 (-0.16696171533493676)
     | > avg_loss_mel: 20.599488394601003 (-0.02442273639497472)
     | > avg_loss_duration: 1.8984536500204177 (-0.00964299837748217)
     | > avg_loss_1: 41.553041730608264 (-0.19530005682080542)


 > EPOCH: 91/999
 --> /content/drive/MyDrive/boinga-translator/runs/run1/run1-November-08-2026_05+25AM-853ac3a

 > TRAINING (2026-11-08 07:11:00) 
আঁই ন খাই .

ইয়ান ঠিক.

স্টার্ট ন লর.

পরটা ন.

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
বোথ ভালা মাইয়া.

সরবত খন, খন.

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
আইয়ী .

অইল্লে এরি.

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
STOP at step 36500

 > CHECKPOINT : /content/drive/MyDrive/boinga-translator/runs/run1/run1-November-08-2026_05+25AM-853ac3a/checkpoint_36500.pth
PASS: run run1 at step 36500 (peak GPU memory 82.333 GB)
CELL-OK C8.train
```

### C8.thr — Throughput of run1

```text
=== BEGIN reports/throughput.json ===
{
  "run": "run1",
  "folder": "/content/drive/MyDrive/boinga-translator/runs/run1/run1-November-08-2026_05+25AM-853ac3a",
  "steps_done": 36500,
  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "batch_size": 16,
  "mixed_precision": true,
  "sec_per_step_mean": 0.1495,
  "n_step_samples": 1459,
  "warmup_only": false,
  "steps_per_hour": 24078.5,
  "sec_per_step_coqui_avg": 0.142929560178286,
  "any_nan_loss": false,
  "last_losses": {
    "loss_disc": 2.370466709136963,
    "loss_disc_real_0": 0.14292851090431213,
    "loss_disc_real_1": 0.20584696531295776,
    "loss_disc_real_2": 0.21867868304252625,
    "loss_disc_real_3": 0.2000235915184021,
    "loss_disc_real_4": 0.22488945722579956,
    "loss_disc_real_5": 0.2583351731300354,
    "loss_0": 2.370466709136963,
    "loss_gen": 2.465363025665283,
    "loss_kl": 2.4050445556640625,
    "loss_feat": 12.664129257202148,
    "loss_mel": 21.07125473022461,
    "loss_duration": 1.858324408531189,
    "loss_1": 40.464115142822266
  },
  "last_logged_step": 36475,
  "peak_mem_gb": 82.333
}
=== END reports/throughput.json ===
PASS: throughput recorded: 0.1495 s/step (1459 samples), peak 82.333 GB, losses from logged step 36475 of 36500
CELL-OK C8.thr
```

### C8.rec — Record the checkpoints

```text
md5:   0% 0/5 [00:00<?, ?file/s]checkpoint_10000.pth: 997.8 MB
md5:  20% 1/5 [00:01<00:05,  1.38s/file]checkpoint_20000.pth: 997.8 MB
md5:  40% 2/5 [00:02<00:04,  1.37s/file]checkpoint_30000.pth: 997.8 MB
md5:  60% 3/5 [00:04<00:02,  1.37s/file]checkpoint_36500.pth: 997.8 MB
md5:  80% 4/5 [00:05<00:01,  1.40s/file]config.json: 0.0 MB
md5: 100% 5/5 [00:05<00:00,  1.11s/file]
PASS: recorded 5 files of run run1 (run1-November-08-2026_05+25AM-853ac3a)
CELL-OK C8.rec
```

### C8.flush — Check they really reached Drive

```text
Mounted at /content/drive
WARNING:google_auth_httplib2:httplib2 transport does not support per-request timeout. Set the timeout when constructing the httplib2.Http instance.
WARNING:google_auth_httplib2:httplib2 transport does not support per-request timeout. Set the timeout when constructing the httplib2.Http instance.
Drive API as nufel1130@gmail.com: must own the mounted Drive
checkpoint_10000.pth: ok
checkpoint_20000.pth: ok
checkpoint_30000.pth: ok
checkpoint_36500.pth: ok
config.json: ok
PASS: 5 files of run run1 on Drive (Drive API)
CELL-OK C8.flush
```

### C8.sample — Sample every checkpoint

```text
Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 10000: mean 2.59 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 20000: mean 2.51 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 30000: mean 2.29 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 36500: mean 2.14 s
PASS: sampled 4 checkpoints (4 in run1-November-08-2026_05+25AM-853ac3a)
CELL-OK C8.sample
```

### C8.cand — Rank checkpoints

```text
=== BEGIN reports/candidates_run1.md ===
# Candidates for run run1

Eval losses: nearest eval block (eval_step = last logged train step before it).

| step | file | eval_step | loss_mel | loss_total | best_model |
|---|---|---|---|---|---|
| 30000 | checkpoint_30000.pth | 30050 | 20.598 | 43.908 | no |
| 36500 | checkpoint_36500.pth | 36475 | 20.599 | 43.892 | no |
| 20000 | checkpoint_20000.pth | 20025 | 20.909 | 44.002 | no |
| 10000 | checkpoint_10000.pth | 10000 | 21.117 | 44.201 | no |
=== END reports/candidates_run1.md ===
PASS: candidates listed (4 checkpoints)
CELL-OK C8.cand
```

### C8.play — Listen: pretrained vs the checkpoints

```text
AUDIO pretrained male #0: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO pretrained male #1: অনে কন ডইল্যা বেয়ারি?:
<IPython.lib.display.Audio object>
AUDIO pretrained male #2: ইত্তিরি বেআন.:
<IPython.lib.display.Audio object>
AUDIO pretrained male #3: দশ মিনিট লাইব.:
<IPython.lib.display.Audio object>
AUDIO pretrained male #4: তেই দশশর আগদি আইব.:
<IPython.lib.display.Audio object>
AUDIO step 30000 #0: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO step 30000 #1: অনে কন ডইল্যা বেয়ারি?:
<IPython.lib.display.Audio object>
AUDIO step 30000 #2: ইত্তিরি বেআন.:
<IPython.lib.display.Audio object>
AUDIO step 30000 #3: দশ মিনিট লাইব.:
<IPython.lib.display.Audio object>
AUDIO step 30000 #4: তেই দশশর আগদি আইব.:
<IPython.lib.display.Audio object>
AUDIO step 36500 #0: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO step 36500 #1: অনে কন ডইল্যা বেয়ারি?:
<IPython.lib.display.Audio object>
AUDIO step 36500 #2: ইত্তিরি বেআন.:
<IPython.lib.display.Audio object>
AUDIO step 36500 #3: দশ মিনিট লাইব.:
<IPython.lib.display.Audio object>
AUDIO step 36500 #4: তেই দশশর আগদি আইব.:
<IPython.lib.display.Audio object>
AUDIO step 20000 #0: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO step 20000 #1: অনে কন ডইল্যা বেয়ারি?:
<IPython.lib.display.Audio object>
AUDIO step 20000 #2: ইত্তিরি বেআন.:
<IPython.lib.display.Audio object>
AUDIO step 20000 #3: দশ মিনিট লাইব.:
<IPython.lib.display.Audio object>
AUDIO step 20000 #4: তেই দশশর আগদি আইব.:
<IPython.lib.display.Audio object>
AUDIO step 10000 #0: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO step 10000 #1: অনে কন ডইল্যা বেয়ারি?:
<IPython.lib.display.Audio object>
AUDIO step 10000 #2: ইত্তিরি বেআন.:
<IPython.lib.display.Audio object>
AUDIO step 10000 #3: দশ মিনিট লাইব.:
<IPython.lib.display.Audio object>
AUDIO step 10000 #4: তেই দশশর আগদি আইব.:
<IPython.lib.display.Audio object>
CELL-OK C8.play
```

### C8-notes — Your observations — run 1

```text
=== BEGIN reports/notes/C8-notes.json ===
{
  "ARTEFACTS": "none",
  "BEST_STEP": "36500",
  "CLEARER_THAN_PRETRAINED": "yes",
  "MORE_CHITTAGONIAN_THAN_PRETRAINED": "yes",
  "NOTES": "none",
  "VERDICT": "good: promote BEST_STEP"
}
=== END reports/notes/C8-notes.json ===
CELL-OK C8-notes
```
