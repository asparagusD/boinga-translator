# Colab run C8 run 1 — C8 run 1 — fine-tune

- File: `C8_finetune_run1.ipynb` · spec `notebooks/specs/c8_finetune_run1.py`
- Run: commit `8005e5d` · branch `chunk/C8-finetune` · GPU `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` · time `2026-11-07T22:00:48+00:00`
- Staleness: OK: the run used 8005e5d, notebook built in adc2f48
- Result: 13/13 cells PASS

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
| C8.sample | Sample every checkpoint | PASS |  |
| C8.cand | Rank checkpoints | PASS |  |
| C8.play | Listen: pretrained vs best candidates vs last | PASS |  |
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
8005e5d chore(notebook): build C8 run 1 notebook
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
PASS: 12 project folders ready under /content/drive/MyDrive/boinga-translator
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
git_commit         8005e5d
disk_free_gb_root  6.4
disk_free_gb_fast  112.5
utc_time           2026-11-07T22:00:48+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 48,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 6.4,
  "git_commit": "8005e5d",
  "gpu_mem_gb": 95.0,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T22:00:48+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C8 run=1 branch=chunk/C8-finetune commit=8005e5d gpu=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition utc=2026-11-07T22:00:48+00:00
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
 'git_commit': '8005e5d',
 'disk_free_gb_root': 6.4,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-07T22:00:48+00:00'}
```

### C8.pre — Restore prepared data

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
copy to fast disk: 100% 8005/8005 [02:11<00:00, 60.83file/s] 
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
selection: 7506 clips, 0 excluded by char_map
restore: 100% 7512/7512 [00:02<00:00, 2826.63file/s]
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
The tensorboard extension is already loaded. To reload it, use:
  %reload_ext tensorboard
Reusing TensorBoard on port 6006 (pid 13366), started 2:45:08 ago. (Use '!kill 13366' to kill it.)
<IPython.core.display.Javascript object>
CELL-OK C8.tb
```

### C8.train — Train run1 to its budget (hours; resumable)

```text
Streaming output truncated to the last 5000 lines.
     | > loss_gen: 2.555069923400879  (2.5583861933814123)
     | > loss_kl: 2.2333428859710693  (2.225482310610591)
     | > loss_feat: 14.053730964660645  (14.028305807498972)
     | > loss_mel: 20.3809757232666  (20.938619276489874)
     | > loss_duration: 1.801131248474121  (1.8598812246563459)
     | > amp_scaler: 256.0  (256.0)
     | > loss_1: 41.02425003051758  (41.610674771395615)
     | > grad_norm_1: tensor(95.4490, device='cuda:0')  (tensor(131.8595, device='cuda:0'))
     | > step_time: 0.1686  (0.14421914382414372)
     | > loader_time: 0.0042  (0.0026650880322311877)


 > EVALUATION 

লই তো আইসসি.

Character '\n' not found in the vocabulary. Discarding it.
দেরি হন বরং ভালা.

Character '\n' not found in the vocabulary. Discarding it.
পইরতু ইচ্ছা ন গরের .

Character '\n' not found in the vocabulary. Discarding it.
ওঁশিয়ের!

Character '\n' not found in the vocabulary. Discarding it.

  --> EVAL PERFORMANCE
     | > avg_loader_time: 0.002059595925467356 (-3.224327450706676e-05)
     | > avg_loss_disc: 2.2779814657710853 (+0.0029729604721073777)
     | > avg_loss_disc_real_0: 0.14455722751361982 (+0.0077681683358691855)
     | > avg_loss_disc_real_1: 0.18020306527614594 (-0.002030689091909482)
     | > avg_loss_disc_real_2: 0.20879673248245603 (+0.008665477945691058)
     | > avg_loss_disc_real_3: 0.20475204288959503 (-0.005003327769892568)
     | > avg_loss_disc_real_4: 0.19950829836584272 (-0.00019208058005287798)
     | > avg_loss_disc_real_5: 0.2135747034280073 (+0.015117429551624123)
     | > avg_loss_0: 2.2779814657710853 (+0.0029729604721073777)
     | > avg_loss_gen: 2.4444995153517945 (+0.0343033586229593)
     | > avg_loss_kl: 2.7425851651600435 (-0.011861676261537557)
     | > avg_loss_feat: 13.786935442969913 (-0.4284297398158472)
     | > avg_loss_mel: 20.421770913260325 (-0.4259206226893859)
     | > avg_loss_duration: 1.9028326471646626 (+0.0024546640259879293)
     | > avg_loss_1: 41.298623857044035 (-0.8294539678664492)


 > EPOCH: 80/999
 --> /content/drive/MyDrive/boinga-translator/runs/run1/run1-November-07-2026_10+05PM-8005e5d

 > TRAINING (2026-11-08 00:10:12) 
সরবত খন, খন.

Character '\n' not found in the vocabulary. Discarding it.
বোথ ভালা মাইয়া.

Character '\n' not found in the vocabulary. Discarding it.
আইয়ী .

Character '\n' not found in the vocabulary. Discarding it.
অইল্লে এরি.
... [4881 lines trimmed] ...
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
দেরি হন বরং ভালা.

Character '\n' not found in the vocabulary. Discarding it.
পইরতু ইচ্ছা ন গরের .

Character '\n' not found in the vocabulary. Discarding it.

  --> EVAL PERFORMANCE
     | > avg_loader_time: 0.002036026545933315 (-3.2192184811546606e-05)
     | > avg_loss_disc: 2.299704114596049 (+0.011058830079577309)
     | > avg_loss_disc_real_0: 0.14670178595752942 (+0.00480801221870239)
     | > avg_loss_disc_real_1: 0.18088406785613015 (+0.0011772733359109722)
     | > avg_loss_disc_real_2: 0.20720360250700087 (+0.00189880742913201)
     | > avg_loss_disc_real_3: 0.2136911398598126 (+0.0028736669392812886)
     | > avg_loss_disc_real_4: 0.20140394532964342 (+0.0053608105296180375)
     | > avg_loss_disc_real_5: 0.21291145859729677 (+0.00863847952513469)
     | > avg_loss_0: 2.299704114596049 (+0.011058830079577309)
     | > avg_loss_gen: 2.4230688015619912 (-0.025180629321507553)
     | > avg_loss_kl: 2.6807159838222328 (+0.01663047642934945)
     | > avg_loss_feat: 14.12932791028704 (-0.17600821313403792)
     | > avg_loss_mel: 20.602658317202614 (-0.03626065027146197)
     | > avg_loss_duration: 1.9010488986968994 (-0.005044057255699563)
     | > avg_loss_1: 41.736819403512136 (-0.22586332048688718)


 > EPOCH: 91/999
 --> /content/drive/MyDrive/boinga-translator/runs/run1/run1-November-07-2026_10+05PM-8005e5d

 > TRAINING (2026-11-08 00:25:48) 
অইল্লে এরি.

আইয়ী .

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
স্টার্ট ন লর.

ইয়ান ঠিক.

পরটা ন.

বোথ ভালা মাইয়া.

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
আঁই ন খাই .

Character '\n' not found in the vocabulary. Discarding it.
সরবত খন, খন.

Character '\n' not found in the vocabulary. Discarding it.
STOP at step 36500

 > CHECKPOINT : /content/drive/MyDrive/boinga-translator/runs/run1/run1-November-07-2026_10+05PM-8005e5d/checkpoint_36500.pth
PASS: run run1 at step 36500 (peak GPU memory 82.335 GB)
CELL-OK C8.train
```

### C8.thr — Throughput of run1

```text
=== BEGIN reports/throughput.json ===
{
  "run": "run1",
  "folder": "/content/drive/MyDrive/boinga-translator/runs/run1/run1-November-07-2026_10+05PM-8005e5d",
  "steps_done": 36500,
  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "batch_size": 16,
  "mixed_precision": true,
  "sec_per_step_mean": 0.1501,
  "n_step_samples": 1459,
  "warmup_only": false,
  "steps_per_hour": 23983.9,
  "sec_per_step_coqui_avg": 0.14412398524098582,
  "any_nan_loss": false,
  "last_losses": {
    "loss_disc": 2.3039069175720215,
    "loss_disc_real_0": 0.13423344492912292,
    "loss_disc_real_1": 0.19613981246948242,
    "loss_disc_real_2": 0.20231404900550842,
    "loss_disc_real_3": 0.19527901709079742,
    "loss_disc_real_4": 0.2328655868768692,
    "loss_disc_real_5": 0.24907612800598145,
    "loss_0": 2.3039069175720215,
    "loss_gen": 2.4944379329681396,
    "loss_kl": 2.4287309646606445,
    "loss_feat": 12.772073745727539,
    "loss_mel": 21.291357040405273,
    "loss_duration": 1.8541450500488281,
    "loss_1": 40.84074401855469
  },
  "last_logged_step": 36475,
  "peak_mem_gb": 82.335
}
=== END reports/throughput.json ===
PASS: throughput recorded: 0.1501 s/step (1459 samples), peak 82.335 GB, losses from logged step 36475 of 36500
CELL-OK C8.thr
```

### C8.sample — Sample every checkpoint

```text
Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 1203: mean 2.48 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 1604: mean 2.58 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 2005: mean 2.57 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 2406: mean 2.51 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 2807: mean 2.57 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 3609: mean 2.60 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 5614: mean 2.77 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 6015: mean 2.83 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 8421: mean 2.74 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 10025: mean 2.85 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 12030: mean 2.81 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 12431: mean 2.72 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 13634: mean 2.84 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 14436: mean 2.82 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 20451: mean 2.70 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 36000: mean 2.01 s
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
step 36500: mean 2.07 s
PASS: sampled 17 checkpoints (17 in run1-November-07-2026_10+05PM-8005e5d)
CELL-OK C8.sample
```

### C8.cand — Rank checkpoints

```text
=== BEGIN reports/candidates_run1.md ===
# Candidates for run run1

Eval losses: nearest eval block (eval_step = last logged train step before it).

| step | file | eval_step | loss_mel | loss_total | best_model |
|---|---|---|---|---|---|
| 20451 | best_model_20451.pth | 20450 | 20.486 | 43.133 | yes |
| 36500 | checkpoint_36500.pth | 36475 | 20.603 | 44.037 | no |
| 36000 | checkpoint_36000.pth | 36075 | 20.639 | 44.251 | no |
| 14436 | best_model_14436.pth | 14425 | 20.763 | 43.448 | no |
| 12431 | best_model_12431.pth | 12425 | 20.873 | 43.928 | no |
| 13634 | best_model_13634.pth | 13625 | 20.9 | 43.925 | no |
| 12030 | best_model_12030.pth | 12025 | 20.984 | 44.08 | no |
| 10025 | best_model_10025.pth | 10000 | 21.038 | 44.105 | no |
| 8421 | best_model_8421.pth | 8400 | 21.246 | 44.317 | no |
| 6015 | best_model_6015.pth | 6000 | 21.376 | 44.636 | no |
| 5614 | best_model_5614.pth | 5600 | 21.563 | 44.658 | no |
| 3609 | best_model_3609.pth | 3600 | 21.984 | 45.101 | no |
| 2807 | best_model_2807.pth | 2800 | 22.443 | 46.446 | no |
| 2406 | best_model_2406.pth | 2400 | 22.789 | 46.845 | no |
| 2005 | best_model_2005.pth | 2000 | 22.913 | 47.464 | no |
| 1604 | best_model_1604.pth | 1600 | 23.245 | 47.786 | no |
| 1203 | best_model_1203.pth | 1200 | 23.635 | 48.995 | no |
=== END reports/candidates_run1.md ===
PASS: candidates listed (17 checkpoints)
CELL-OK C8.cand
```

### C8.play — Listen: pretrained vs best candidates vs last

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
AUDIO step 20451 #0: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO step 20451 #1: অনে কন ডইল্যা বেয়ারি?:
<IPython.lib.display.Audio object>
AUDIO step 20451 #2: ইত্তিরি বেআন.:
<IPython.lib.display.Audio object>
AUDIO step 20451 #3: দশ মিনিট লাইব.:
<IPython.lib.display.Audio object>
AUDIO step 20451 #4: তেই দশশর আগদি আইব.:
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
AUDIO step 36000 #0: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO step 36000 #1: অনে কন ডইল্যা বেয়ারি?:
<IPython.lib.display.Audio object>
AUDIO step 36000 #2: ইত্তিরি বেআন.:
<IPython.lib.display.Audio object>
AUDIO step 36000 #3: দশ মিনিট লাইব.:
<IPython.lib.display.Audio object>
AUDIO step 36000 #4: তেই দশশর আগদি আইব.:
<IPython.lib.display.Audio object>
AUDIO step 14436 #0: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO step 14436 #1: অনে কন ডইল্যা বেয়ারি?:
<IPython.lib.display.Audio object>
AUDIO step 14436 #2: ইত্তিরি বেআন.:
<IPython.lib.display.Audio object>
AUDIO step 14436 #3: দশ মিনিট লাইব.:
<IPython.lib.display.Audio object>
AUDIO step 14436 #4: তেই দশশর আগদি আইব.:
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
