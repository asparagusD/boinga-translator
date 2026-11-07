# Colab run C7 run 1 — C7 — training runner and smoke test

- File: `C7_train_runner_run1.ipynb` · spec `notebooks/specs/c7_train_runner.py`
- Run: commit `f681627` · branch `chunk/C7-train-runner` · GPU `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` · time `2026-11-07T20:40:43+00:00`
- Staleness: OK: the run used f681627, notebook built in 90c4504
- Result: 13/13 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C7.pre | Restore prepared data on this VM | PASS |  |
| C7.1 | Build and validate config | PASS |  |
| C7.2 | One-batch loader check | PASS |  |
| C7.4.0 | Fresh smoke run | PASS |  |
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
f681627 docs(decisions): record gate G5 verdict before full fine-tune
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
git_commit         f681627
disk_free_gb_root  6.4
disk_free_gb_fast  112.5
utc_time           2026-11-07T20:40:43+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 6.4,
  "git_commit": "f681627",
  "gpu_mem_gb": 95.0,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T20:40:43+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C7 run=1 branch=chunk/C7-train-runner commit=f681627 gpu=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition utc=2026-11-07T20:40:43+00:00
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
 'git_commit': 'f681627',
 'disk_free_gb_root': 6.4,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-07T20:40:43+00:00'}
```

### C7.pre — Restore prepared data on this VM

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
copy to fast disk: 100% 8005/8005 [02:58<00:00, 44.82file/s]
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
selection: 7506 clips, 0 excluded by char_map
restore: 100% 7512/7512 [00:09<00:00, 751.34file/s]
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
... [181 lines trimmed] ...
    [
      "দশ মিনিট লাইব."
    ],
    [
      "তেই দশশর আগদি আইব."
    ]
  ],
  "test_sentences_file": "",
  "text_cleaner": null,
  "training_seed": 1234,
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
| `test_delay_epochs` | -1 | 0 | test sentences from the first epoch: early sanity listening |
| `test_sentences` | [["ইয়াহুদ,নয়,ব্যয়,ইয়াহুদী,নেওয়া,উভয়ে,যায়,হয়েছিল,প্রয়োজন।"]] | [["এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন."], ["অনে কন ... | first 5 val manifest texts: prepare_text output as trained, not re-normalised (Gate G4) |
| `training_seed` | 54321 | 1234 | the project seed (settings.SEED) for reproducible runs |
=== END reports/config_overrides.md ===
PASS: config written to /content/boinga-translator/configs/vits_ctg_finetune.json (base /content/drive/MyDrive/boinga-translator/pretrained/male/config.json, 23 overrides)
sample_rate 22050, batch_size 16, lr_gen 0.0001, lr_disc 0.0001, mixed_precision True
PASS: coqui accepts config
CELL-OK C7.1
```

### C7.2 — One-batch loader check

```text
get_data_loader(config: coqpit.coqpit.Coqpit, *, is_eval: bool = False, samples: list[typing.Any] | None = None, verbose: bool = True) -> 'DataLoader'
samples: 6415 train / 338 eval
100% 6415/6415 [00:00<00:00, 40676.03it/s]
আইয়ী .

অইল্লে এরি.

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
first batch (incl. worker startup + spec/mel, not a training speed) in 3.55 s
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

### C7.4.0 — Fresh smoke run

```text
PASS: cleaned run smoke: removed /content/drive/MyDrive/boinga-translator/runs/smoke (2.00 GB; Drive keeps deleted files in its trash until emptied)
CELL-OK C7.4.0
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
... [457 lines trimmed] ...
=== END configs/vits_ctg_finetune.json ===
=== BEGIN reports/config_overrides.md ===
=== END reports/config_overrides.md ===
PASS: config written to /content/boinga-translator/configs/vits_ctg_finetune.json (base /content/drive/MyDrive/boinga-translator/pretrained/male/config.json, 23 overrides)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
     | > loss_duration: 2.1192009449005127  (2.6903229713439942)
     | > amp_scaler: 512.0  (1345.8285714285714)
     | > loss_1: 74.22608184814453  (87.96364430018834)
     | > grad_norm_1: tensor(109.3967, device='cuda:0')  (tensor(150.3474, device='cuda:0'))
     | > step_time: 0.5801  (0.41274990354265484)
     | > loader_time: 0.0024  (0.003660651615687779)


   --> TIME: 2026-11-07 20:47:13 -- STEP: 40/400 -- GLOBAL_STEP: 40
     | > current_lr_0: 9.950121682254156e-05  (9.974416591222455e-05)
     | > current_lr_1: 9.95883239353732e-05  (9.982492102372435e-05)
     | > loss_disc: 3.7293248176574707  (8.544663816690447)
     | > loss_disc_real_0: 0.8916399478912354  (2.3343529120087623)
     | > loss_disc_real_1: 0.25575873255729675  (0.5761487621814012)
     | > loss_disc_real_2: 0.40154704451560974  (0.5675577346235514)
     | > loss_disc_real_3: 0.3599744737148285  (0.6307933833450079)
     | > loss_disc_real_4: 0.3881739377975464  (0.5440529253333807)
     | > loss_disc_real_5: 0.5578170418739319  (2.442156211659312)
     | > loss_0: 3.7293248176574707  (8.544663816690447)
     | > grad_norm_0: tensor(45.1611, device='cuda:0')  (tensor(82.6008, device='cuda:0'))
     | > loss_gen: 3.516704559326172  (3.6687462985515595)
     | > loss_kl: 6.808310508728027  (15.049708867073056)
     | > loss_feat: 20.39444351196289  (27.50361270904541)
     | > loss_mel: 31.369155883789062  (36.6533618927002)
     | > loss_duration: 1.8995816707611084  (2.596161955595016)
     | > amp_scaler: 512.0  (1241.6)
     | > loss_1: 63.988197326660156  (85.471591091156)
     | > grad_norm_1: tensor(130.1550, device='cuda:0')  (tensor(146.1565, device='cuda:0'))
     | > step_time: 0.2904  (0.3970996856689454)
     | > loader_time: 0.0025  (0.003507184982299805)


   --> TIME: 2026-11-07 20:47:15 -- STEP: 45/400 -- GLOBAL_STEP: 45
     | > current_lr_0: 9.943904410714931e-05  (9.971302637580806e-05)
     | > current_lr_1: 9.952609679164422e-05  (9.979448363637543e-05)
     | > loss_disc: 5.47482967376709  (8.050723621580337)
     | > loss_disc_real_0: 2.657137155532837  (2.2284311850865683)
     | > loss_disc_real_1: 0.4006015658378601  (0.5517539319064881)
     | > loss_disc_real_2: 0.4186107814311981  (0.5443456878264746)
     | > loss_disc_real_3: 0.2522156238555908  (0.5911402530140347)
     | > loss_disc_real_4: 0.273187518119812  (0.5164227505524954)
     | > loss_disc_real_5: 0.4202556014060974  (2.2168142408132554)
     | > loss_0: 5.47482967376709  (8.050723621580337)
     | > grad_norm_0: tensor(108.1200, device='cuda:0')  (tensor(80.0525, device='cuda:0'))
     | > loss_gen: 3.52711820602417  (3.667906501558092)
     | > loss_kl: 7.227449417114258  (14.080929491255016)
     | > loss_feat: 24.52824592590332  (27.186069107055665)
     | > loss_mel: 35.1555061340332  (36.26935128106011)
     | > loss_duration: 1.8877040147781372  (2.5217951748106207)
     | > amp_scaler: 512.0  (1160.5333333333333)
     | > loss_1: 72.32601928710938  (83.726050652398)
     | > grad_norm_1: tensor(137.9020, device='cuda:0')  (tensor(143.7392, device='cuda:0'))
     | > step_time: 0.1483  (0.38856787681579597)
     | > loader_time: 0.0021  (0.0033863915337456595)

STOP at step 50

 > CHECKPOINT : /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+46PM-f681627/checkpoint_50.pth
PASS: run smoke at step 50 (peak GPU memory 3.39 GB)
CELL-OK C7.4.a
```

### C7.4.b — Resume test: to 60 steps

```text
RESUME from step 50 in /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+46PM-f681627
 > Training Environment:
 | > Backend: Torch
 | > Mixed precision: True
 | > Precision: fp16
 | > Current device: 0
 | > Num. of GPUs: 1
 | > Num. of CPUs: 48
 | > Num. of Torch Threads: 24
 | > Torch seed: 1234
 | > Torch CUDNN: True
 | > Torch CUDNN deterministic: False
 | > Torch CUDNN benchmark: True
 | > Torch TF32 MatMul: False
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1794084450.810610   18214 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1794084450.841724   18214 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1794084451.842525   18214 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1794084451.842786   18214 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
 > Start Tensorboard: tensorboard --logdir=/content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+46PM-f681627
 > Continuing from checkpoint_50.pth ...
 > Restoring Model...
 > Restoring Optimizer...
 > Restoring Scheduler...
 > Restoring Scaler...
 > Model restored from step 50

 > Model has 83050732 parameters
 > Restoring best loss from checkpoint_50.pth ...
 > Starting with loaded last best loss {'train_loss': 44.848531883590084, 'eval_loss': None}

 > EPOCH: 0/999
 --> /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+46PM-f681627
100% 6415/6415 [00:00<00:00, 42005.57it/s]

 > TRAINING (2026-11-07 20:47:34) 
আইয়ী .

Character '\n' not found in the vocabulary. Discarding it.
অইল্লে এরি.

Character '\n' not found in the vocabulary. Discarding it.

   --> TIME: 2026-11-07 20:47:41 -- STEP: 4/400 -- GLOBAL_STEP: 55
     | > current_lr_0: 9.932723110067987e-05  (9.934585883720885e-05)
     | > current_lr_1: 9.941418589985758e-05  (9.943282994380871e-05)
     | > loss_disc: 4.2839179039001465  (4.176746368408203)
     | > loss_disc_real_0: 0.7954037189483643  (1.05277119576931)
     | > loss_disc_real_1: 0.42728447914123535  (0.29751788452267647)
     | > loss_disc_real_2: 0.4044608473777771  (0.3502687066793442)
     | > loss_disc_real_3: 0.2580528259277344  (0.28932855278253555)
     | > loss_disc_real_4: 0.35927993059158325  (0.34266381710767746)
     | > loss_disc_real_5: 0.5298196077346802  (0.5258325710892677)
     | > loss_0: 4.2839179039001465  (4.176746368408203)
     | > grad_norm_0: tensor(62.7535, device='cuda:0')  (tensor(55.6619, device='cuda:0'))
     | > loss_gen: 3.6788997650146484  (3.4509655237197876)
     | > loss_kl: 6.001951694488525  (5.766870021820068)
     | > loss_feat: 30.11858558654785  (27.938390731811523)
     | > loss_mel: 30.71438217163086  (30.663147926330566)
     | > loss_duration: 1.9997127056121826  (2.0685315132141113)
     | > amp_scaler: 512.0  (512.0)
     | > loss_1: 72.5135269165039  (69.88790321350098)
     | > grad_norm_1: tensor(121.4775, device='cuda:0')  (tensor(160.8688, device='cuda:0'))
     | > step_time: 0.5767  (0.8402160406112671)
     | > loader_time: 0.0026  (0.0025694966316223145)

STOP at step 60

 > CHECKPOINT : /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+46PM-f681627/checkpoint_60.pth
PASS: run smoke at step 60 (peak GPU memory 3.262 GB)
CELL-OK C7.4.b
```

### C7.4.c — Throughput

```text
=== BEGIN reports/throughput.json ===
{
  "run": "smoke",
  "folder": "/content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_08+46PM-f681627",
  "steps_done": 60,
  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "batch_size": 16,
  "mixed_precision": true,
  "sec_per_step_mean": 0.3418,
  "n_step_samples": 10,
  "warmup_only": false,
  "steps_per_hour": 10533.4,
  "sec_per_step_coqui_avg": 0.8402160406112671,
  "any_nan_loss": false,
  "last_losses": {
    "loss_disc": 4.2839179039001465,
    "loss_disc_real_0": 0.7954037189483643,
    "loss_disc_real_1": 0.42728447914123535,
    "loss_disc_real_2": 0.4044608473777771,
    "loss_disc_real_3": 0.2580528259277344,
    "loss_disc_real_4": 0.35927993059158325,
    "loss_disc_real_5": 0.5298196077346802,
    "loss_0": 4.2839179039001465,
    "loss_gen": 3.6788997650146484,
    "loss_kl": 6.001951694488525,
    "loss_feat": 30.11858558654785,
    "loss_mel": 30.71438217163086,
    "loss_duration": 1.9997127056121826,
    "loss_1": 72.5135269165039
  },
  "last_logged_step": 55,
  "peak_mem_gb": 3.39
}
=== END reports/throughput.json ===
PASS: throughput recorded: 0.3418 s/step (10 samples), peak 3.39 GB, losses from logged step 55 of 60
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
