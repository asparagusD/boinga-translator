# Colab run C7 run 1 — C7 — training runner and smoke test

- File: `C7_train_runner_run1.ipynb` · spec `notebooks/specs/c7_train_runner.py`
- Run: commit `87c0470` · branch `chunk/C7-train-runner` · GPU `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` · time `2026-11-07T20:15:12+00:00`
- Staleness: OK: the run used 87c0470, notebook built in b0ec918
- Result: 11/12 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C7.pre | Restore prepared data on this VM | PASS |  |
| C7.1 | Build and validate config | FAIL | expected /PASS: coqui accepts config/ not found; forbidden /^FAIL:/ found |
| C7.2 | One-batch loader check | PASS |  |
| C7.4.a | Smoke test: 50 steps | PASS |  |
| C7.4.b | Resume test: to 60 steps | PASS |  |
| C7.4.c | Throughput | PASS |  |
| C7.4.d | Listen: smoke checkpoint vs pretrained | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
87c0470 chore(notebook): build C7 run 1 notebook
PASS: /content/boinga-translator on chunk/C7-train-runner
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
gpu_name           NVIDIA RTX PRO 6000 Blackwell Server Edition
gpu_mem_gb         95.0
coqui_tts          0.28.0
numpy              2.4.6
git_commit         87c0470
disk_free_gb_root  8.2
disk_free_gb_fast  112.5
utc_time           2026-11-07T20:15:12+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 8.2,
  "git_commit": "87c0470",
  "gpu_mem_gb": 95.0,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T20:15:12+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C7 run=1 branch=chunk/C7-train-runner commit=87c0470 gpu=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition utc=2026-11-07T20:15:12+00:00
{'python': '3.13.16',
 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39',
 'is_colab': True,
 'torch': '2.14.1+cu130',
 'torch_cuda': '13.0',
 'torch_error': None,
 'gpu_name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
 'gpu_mem_gb': 95.0,
 'coqui_tts': '0.28.0',
 'numpy': '2.4.6',
 'git_commit': '87c0470',
 'disk_free_gb_root': 8.2,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-07T20:15:12+00:00'}
```

### C7.pre — Restore prepared data on this VM

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
copy to fast disk: 100% 8005/8005 [03:01<00:00, 44.12file/s]
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
selection: 7506 clips, 0 excluded by char_map
restore: 100% 7512/7512 [00:05<00:00, 1316.55file/s]
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
CELL-OK C7.pre
```

### C7.1 — Build and validate config

```text
=== BEGIN configs/vits_ctg_finetune.json ===
{
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
  "batch_group_size": 0,
  "batch_size": 16,
  "characters": {
    "blank": "<BLNK>",
    "bos": "<BOS>",
    "characters": "তট৫ভিঐঋখঊড়ইজমএেঘঙসীঢ়হঞ‘ঈকণ৬ঁৗশঢঠ‌১্২৮দৃঔগও—ছউংবৈঝাযফ‍চরষঅৌৎথড়৪ধ০ুূ৩আঃপয়’নলো",
    "characters_class": "TTS.tts.utils.text.characters.Graphemes",
    "eos": "।",
    "is_sorted": true,
    "is_unique": true,
    "pad": "<PAD>",
    "phonemes": null,
    "punctuations": "-!,|.? ",
    "vocab_dict": null
  },
  "compute_f0": false,
  "compute_input_seq_cache": true,
  "compute_linear_spec": true,
  "cudnn_benchmark": true,
  "cudnn_deterministic": false,
  "cudnn_enable": true,
  "d_vector_dim": 0,
  "d_vector_file": null,
  "dashboard_logger": "tensorboard",
  "datasets": [
    {
      "dataset_name": "",
      "formatter": "ljspeech",
      "ignored_speakers": null,
      "language": "bn",
      "meta_file_attn_mask": "",
      "meta_file_train": "metadata_train.csv",
      "meta_file_val": "metadata_val.csv",
      "path": "/content/boinga-fast/prepared",
      "phonemizer": ""
    }
  ],
  "disc_loss_alpha": 1.0,
  "distributed_backend": "nccl",
  "distributed_url": "tcp://localhost:54321",
  "dur_loss_alpha": 1.0,
  "enable_eos_bos_chars": false,
  "epochs": 1000,
  "eval_batch_size": 8,
  "eval_split_max_size": null,
  "eval_split_size": 0.01,
  "feat_loss_alpha": 1.0,
  "gen_loss_alpha": 1.0,
... [182 lines trimmed] ...
    [
      "দশ মিনিট লাইব."
    ],
    [
      "তেই দশশর আগদি আইব."
    ]
  ],
  "test_sentences_file": "",
  "text_cleaner": null,
  "training_seed": 54321,
  "use_d_vector_file": false,
  "use_grad_scaler": false,
  "use_language_embedding": false,
  "use_language_weighted_sampler": false,
  "use_length_weighted_sampler": false,
  "use_noise_augment": false,
  "use_phonemes": false,
  "use_speaker_embedding": false,
  "use_speaker_weighted_sampler": false,
  "use_weighted_sampler": false,
  "wandb_entity": null,
  "weighted_sampler_attrs": null,
  "weighted_sampler_multipliers": null
}
=== END configs/vits_ctg_finetune.json ===
=== BEGIN reports/config_overrides.md ===
# Fine-tune config overrides

Base: `/content/drive/MyDrive/boinga-translator/pretrained/male/config.json`, run `smoke`.

| key | pretrained | fine-tune | reason |
|---|---|---|---|
| `batch_size` | 104 | 16 | fits a T4 (16 GB) with fp16 and 15 s clips; smoke test measures |
| `datasets` | [{"name": "", "path": "/home/ansary/Shabab/bn_tts/male/mo... | [{"formatter": "ljspeech", "dataset_name": "", "path": "/... | our prepared ljspeech manifests on the fast disk replace the author's paths |
| `epochs` | 5000 | 1000 | upper bound only; runs stop by a step budget (RUN_BUDGETS, C8) |
| `eval_batch_size` | 48 | 8 | eval runs beside training memory; half the train batch |
| `lr_disc` | 0.0002 | 0.0001 | 0.5 x pretrained, like lr_gen, so G and D stay balanced |
| `lr_gen` | 0.0002 | 0.0001 | 0.5 x pretrained: fine-tuning should not undo the pretrained weights |
| `max_audio_len` | Infinity | 330750 | 15 s x sample_rate: longer clips risk T4 out-of-memory |
| `min_audio_len` | 1 | 11025 | 0.5 s x sample_rate, on trimmed clips (C6 follow-up) |
| `mixed_precision` | false | true | fp16 halves memory, speeds up a T4; NaN -> fp32 (skill) |
| `num_eval_loader_workers` | 4 | 1 | eval is small; leave the other CPU to training |
| `num_loader_workers` | 4 | 2 | Colab VMs have 2 vCPUs |
| `output_path` | "/home/ansary/Shabab/" | "/content/drive/MyDrive/boinga-translator/runs" | runs on Drive (paths.runs()) survive a Colab disconnect |
| `print_step` | 50 | 5 | smoke run only: ~10 timed steps in 60 instead of 1 |
| `restore_path` | "/home/ansary/Shabab/vits_20_october/checkpoint_626000.pth" | null | the author's absolute path; the runner passes it (C7.3) |
| `run_eval` | true | true | val loss picks the best checkpoint, with listening (vits-finetune) |
| `run_name` | "vits_20_october" | "smoke" | given per run (boinga build-config --run NAME) |
| `save_best_after` | 500 | 1000 | no best checkpoint from the noisy first steps |
| `save_checkpoints` | true | true | resume after a Colab disconnect needs them |
| `save_n_checkpoints` | 5 | 2 | keep the last 2 (+ best): ~1 GB each, Drive budget |
| `save_step` | 500 | 500 | a checkpoint every ~500 steps to Drive (vits-finetune) |
| `seed` | (missing) | 1234 | the project seed (settings.SEED) for reproducible runs |
| `test_delay_epochs` | -1 | 0 | test sentences from the first epoch: early sanity listening |
| `test_sentences` | [["ইয়াহুদ,নয়,ব্যয়,ইয়াহুদী,নেওয়া,উভয়ে,যায়,হয়েছিল,প্রয়োজন।"]] | [["এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন."], ["অনে কন ... | first 5 val manifest texts: prepare_text output as trained, not re-normalised (Gate G4) |
=== END reports/config_overrides.md ===
PASS: config written to /content/boinga-translator/configs/vits_ctg_finetune.json (base /content/drive/MyDrive/boinga-translator/pretrained/male/config.json, 23 overrides)
sample_rate 22050, batch_size 16, lr_gen 0.0001, lr_disc 0.0001, mixed_precision True
FAIL: coqui loaded the config but not these overrides: ['seed']
CELL-OK C7.1
```

### C7.2 — One-batch loader check

```text
get_data_loader(config: coqpit.coqpit.Coqpit, *, is_eval: bool = False, samples: list[typing.Any] | None = None, verbose: bool = True) -> 'DataLoader'
samples: 6415 train / 338 eval
100% 6415/6415 [00:00<00:00, 39924.77it/s]
অইল্লে এরি.

আইয়ী .

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
first batch (incl. worker startup + spec/mel, not a training speed) in 3.74 s
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
CELL-OK C7.2
```

### C7.4.a — Smoke test: 50 steps

```text
=== BEGIN configs/vits_ctg_finetune.json ===
{
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
  "batch_group_size": 0,
  "batch_size": 16,
  "characters": {
    "blank": "<BLNK>",
    "bos": "<BOS>",
    "characters": "তট৫ভিঐঋখঊড়ইজমএেঘঙসীঢ়হঞ‘ঈকণ৬ঁৗশঢঠ‌১্২৮দৃঔগও—ছউংবৈঝাযফ‍চরষঅৌৎথড়৪ধ০ুূ৩আঃপয়’নলো",
    "characters_class": "TTS.tts.utils.text.characters.Graphemes",
    "eos": "।",
    "is_sorted": true,
    "is_unique": true,
    "pad": "<PAD>",
    "phonemes": null,
    "punctuations": "-!,|.? ",
    "vocab_dict": null
  },
  "compute_f0": false,
  "compute_input_seq_cache": true,
  "compute_linear_spec": true,
  "cudnn_benchmark": true,
  "cudnn_deterministic": false,
  "cudnn_enable": true,
  "d_vector_dim": 0,
  "d_vector_file": null,
  "dashboard_logger": "tensorboard",
  "datasets": [
    {
      "dataset_name": "",
      "formatter": "ljspeech",
      "ignored_speakers": null,
      "language": "bn",
      "meta_file_attn_mask": "",
      "meta_file_train": "metadata_train.csv",
      "meta_file_val": "metadata_val.csv",
      "path": "/content/boinga-fast/prepared",
      "phonemizer": ""
    }
  ],
  "disc_loss_alpha": 1.0,
  "distributed_backend": "nccl",
  "distributed_url": "tcp://localhost:54321",
  "dur_loss_alpha": 1.0,
  "enable_eos_bos_chars": false,
  "epochs": 1000,
  "eval_batch_size": 8,
  "eval_split_max_size": null,
  "eval_split_size": 0.01,
  "feat_loss_alpha": 1.0,
  "gen_loss_alpha": 1.0,
... [458 lines trimmed] ...
=== END configs/vits_ctg_finetune.json ===
=== BEGIN reports/config_overrides.md ===
=== END reports/config_overrides.md ===
PASS: config written to /content/boinga-translator/configs/vits_ctg_finetune.json (base /content/drive/MyDrive/boinga-translator/pretrained/male/config.json, 23 overrides)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
     | > loss_duration: 2.0508363246917725  (2.598838485990252)
     | > amp_scaler: 512.0  (1375.0857142857142)
     | > loss_1: 75.00283813476562  (86.98986903599331)
     | > grad_norm_1: tensor(126.2277, device='cuda:0')  (tensor(185.8538, device='cuda:0'))
     | > step_time: 0.5797  (0.41103175027029853)
     | > loader_time: 0.0027  (0.003684289114815848)


   --> TIME: 2026-11-07 20:21:21 -- STEP: 40/400 -- GLOBAL_STEP: 40
     | > current_lr_0: 9.950121682254156e-05  (9.974416591222455e-05)
     | > current_lr_1: 9.95883239353732e-05  (9.982429606278683e-05)
     | > loss_disc: 3.5871920585632324  (8.76216989159584)
     | > loss_disc_real_0: 0.8240166902542114  (2.429082728922367)
     | > loss_disc_real_1: 0.2842716872692108  (0.5487079307436943)
     | > loss_disc_real_2: 0.3894321024417877  (0.5294397667050361)
     | > loss_disc_real_3: 0.39165642857551575  (0.6380358688533306)
     | > loss_disc_real_4: 0.2567608058452606  (0.6144590444862841)
     | > loss_disc_real_5: 0.40240558981895447  (2.576976654306054)
     | > loss_0: 3.5871920585632324  (8.76216989159584)
     | > grad_norm_0: tensor(35.3861, device='cuda:0')  (tensor(70.9791, device='cuda:0'))
     | > loss_gen: 3.3145029544830322  (3.675729602575302)
     | > loss_kl: 6.810514450073242  (14.534237551689149)
     | > loss_feat: 19.87938690185547  (27.398195886611937)
     | > loss_mel: 31.4284725189209  (36.41949372291565)
     | > loss_duration: 1.9318485260009766  (2.513711762428284)
     | > amp_scaler: 512.0  (1267.1999999999998)
     | > loss_1: 63.364723205566406  (84.54136867523195)
     | > grad_norm_1: tensor(112.8646, device='cuda:0')  (tensor(176.2226, device='cuda:0'))
     | > step_time: 0.29  (0.39553089141845704)
     | > loader_time: 0.0025  (0.003535187244415283)


   --> TIME: 2026-11-07 20:21:23 -- STEP: 45/400 -- GLOBAL_STEP: 45
     | > current_lr_0: 9.943904410714931e-05  (9.971302637580806e-05)
     | > current_lr_1: 9.952609679164422e-05  (9.979392811554209e-05)
     | > loss_disc: 3.8996033668518066  (8.210978094736733)
     | > loss_disc_real_0: 1.3710339069366455  (2.282459416654375)
     | > loss_disc_real_1: 0.23850160837173462  (0.5151219281885359)
     | > loss_disc_real_2: 0.365928590297699  (0.5098509867986043)
     | > loss_disc_real_3: 0.2165547013282776  (0.5979039599498112)
     | > loss_disc_real_4: 0.268356591463089  (0.5777360118097727)
     | > loss_disc_real_5: 0.3466905951499939  (2.3401642060942116)
     | > loss_0: 3.8996033668518066  (8.210978094736733)
     | > grad_norm_0: tensor(53.2444, device='cuda:0')  (tensor(69.1335, device='cuda:0'))
     | > loss_gen: 3.5641396045684814  (3.6477260006798637)
     | > loss_kl: 7.190377712249756  (13.61322347852919)
     | > loss_feat: 22.08795928955078  (26.813634660508896)
     | > loss_mel: 32.814056396484375  (35.95539991590712)
     | > loss_duration: 1.8436484336853027  (2.4462359878751965)
     | > amp_scaler: 512.0  (1183.2888888888886)
     | > loss_1: 67.50018310546875  (82.47622010972766)
     | > grad_norm_1: tensor(136.8300, device='cuda:0')  (tensor(173.4253, device='cuda:0'))
     | > step_time: 0.1496  (0.3871864000956217)
     | > loader_time: 0.0021  (0.0034127818213568794)

STOP at step 50

 > CHECKPOINT : /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+20PM-87c0470/checkpoint_50.pth
PASS: run smoke at step 50 (peak GPU memory 3.385 GB)
CELL-OK C7.4.a
```

### C7.4.b — Resume test: to 60 steps

```text
RESUME from step 50 in /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+20PM-87c0470
 > Training Environment:
 | > Backend: Torch
 | > Mixed precision: True
 | > Precision: fp16
 | > Current device: 0
 | > Num. of GPUs: 1
 | > Num. of CPUs: 48
 | > Num. of Torch Threads: 24
 | > Torch seed: 54321
 | > Torch CUDNN: True
 | > Torch CUDNN deterministic: False
 | > Torch CUDNN benchmark: True
 | > Torch TF32 MatMul: False
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1794082898.572278   15431 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1794082898.603568   15431 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1794082899.613137   15431 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1794082899.613393   15431 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
 > Start Tensorboard: tensorboard --logdir=/content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+20PM-87c0470
 > Continuing from checkpoint_50.pth ...
 > Restoring Model...
 > Restoring Optimizer...
 > Restoring Scheduler...
 > Restoring Scaler...
 > Model restored from step 50

 > Model has 83050732 parameters
 > Restoring best loss from checkpoint_50.pth ...
 > Starting with loaded last best loss {'train_loss': 44.41926211483626, 'eval_loss': None}

 > EPOCH: 0/999
 --> /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+20PM-87c0470
100% 6415/6415 [00:00<00:00, 41922.06it/s]

 > TRAINING (2026-11-07 20:21:42) 
আইয়ী .

Character '\n' not found in the vocabulary. Discarding it.
অইল্লে এরি.

Character '\n' not found in the vocabulary. Discarding it.

   --> TIME: 2026-11-07 20:21:49 -- STEP: 4/400 -- GLOBAL_STEP: 55
     | > current_lr_0: 9.932723110067987e-05  (9.934585883720885e-05)
     | > current_lr_1: 9.941418589985758e-05  (9.943282994380871e-05)
     | > loss_disc: 4.577139377593994  (3.8922691345214844)
     | > loss_disc_real_0: 1.046539545059204  (0.8813933432102203)
     | > loss_disc_real_1: 0.2953500747680664  (0.2593305930495262)
     | > loss_disc_real_2: 0.4394058585166931  (0.3375304415822029)
     | > loss_disc_real_3: 0.4475542902946472  (0.31171995028853416)
     | > loss_disc_real_4: 0.5626880526542664  (0.34921281784772873)
     | > loss_disc_real_5: 0.552116334438324  (0.5286746025085449)
     | > loss_0: 4.577139377593994  (3.8922691345214844)
     | > grad_norm_0: tensor(54.2939, device='cuda:0')  (tensor(40.0319, device='cuda:0'))
     | > loss_gen: 3.858236789703369  (3.533273756504059)
     | > loss_kl: 5.628435134887695  (5.730025768280029)
     | > loss_feat: 30.47313117980957  (28.011201858520508)
     | > loss_mel: 30.91847801208496  (30.48617458343506)
     | > loss_duration: 1.9640508890151978  (2.0354686975479126)
     | > amp_scaler: 512.0  (512.0)
     | > loss_1: 72.84233093261719  (69.79614543914795)
     | > grad_norm_1: tensor(226.2462, device='cuda:0')  (tensor(157.7954, device='cuda:0'))
     | > step_time: 0.5731  (0.8302930593490601)
     | > loader_time: 0.0024  (0.0024257302284240723)

STOP at step 60

 > CHECKPOINT : /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+20PM-87c0470/checkpoint_60.pth
PASS: run smoke at step 60 (peak GPU memory 3.262 GB)
CELL-OK C7.4.b
```

### C7.4.c — Throughput

```text
=== BEGIN reports/throughput.json ===
{
  "run": "smoke",
  "folder": "/content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+20PM-87c0470",
  "steps_done": 60,
  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "batch_size": 16,
  "mixed_precision": true,
  "sec_per_step_mean": 0.3406,
  "n_step_samples": 10,
  "warmup_only": false,
  "steps_per_hour": 10570.2,
  "sec_per_step_coqui_avg": 0.8302930593490601,
  "any_nan_loss": false,
  "last_losses": {
    "loss_disc": 4.577139377593994,
    "loss_disc_real_0": 1.046539545059204,
    "loss_disc_real_1": 0.2953500747680664,
    "loss_disc_real_2": 0.4394058585166931,
    "loss_disc_real_3": 0.4475542902946472,
    "loss_disc_real_4": 0.5626880526542664,
    "loss_disc_real_5": 0.552116334438324,
    "loss_0": 4.577139377593994,
    "loss_gen": 3.858236789703369,
    "loss_kl": 5.628435134887695,
    "loss_feat": 30.47313117980957,
    "loss_mel": 30.91847801208496,
    "loss_duration": 1.9640508890151978,
    "loss_1": 72.84233093261719
  },
  "last_logged_step": 55,
  "peak_mem_gb": 3.385
}
=== END reports/throughput.json ===
PASS: throughput recorded: 0.3406 s/step (10 samples), peak 3.385 GB, losses from logged step 55 of 60
CELL-OK C7.4.c
```

### C7.4.d — Listen: smoke checkpoint vs pretrained

```text
WARNING:TTS.tts.utils.languages:Dataset `` has no language specified.
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
synth: use_cuda=True; set inference_noise_scale=0.667, inference_noise_scale_dp=0.8, length_scale=1.0
AUDIO pretrained male: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
AUDIO smoke step 60: এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন.:
<IPython.lib.display.Audio object>
CELL-OK C7.4.d
```
