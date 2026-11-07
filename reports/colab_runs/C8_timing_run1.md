# Colab run C8 run 1 — C8 — timing at batch 32 (RTX PRO 6000)

- File: `C8_timing_run1.ipynb` · spec `notebooks/specs/c8_timing.py`
- Run: commit `946319e` · branch `chunk/C8-finetune` · GPU `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition` · time `2026-11-07T21:26:07+00:00`
- Staleness: OK: the run used 946319e, notebook built in 946319e
- Result: 12/12 cells PASS

## Cells

| Cell | Title | Status | Reason |
|---|---|---|---|
| BOOT.1 | mount Google Drive | PASS |  |
| BOOT.2 | clone or update the private repo with the GH_TOKEN secret | PASS |  |
| BOOT.3 | install pinned dependencies (coqui-tts==0.28.0 via the [train] extra) | PASS |  |
| BOOT.4 | create project folders (the HDR cell records the environment) | PASS |  |
| HDR | Run header | PASS |  |
| C8T.pre | Restore prepared data on this VM | PASS |  |
| C8T.1 | Build and validate config | PASS |  |
| C8T.2 | One-batch loader check | PASS |  |
| C8T.3 | Fresh smoke run | PASS |  |
| C8T.4 | Smoke test: 50 steps | PASS |  |
| C8T.5 | Resume test: to 60 steps | PASS |  |
| C8T.6 | Throughput | PASS |  |

## Output

### BOOT.1 — mount Google Drive

```text
Mounted at /content/drive
PASS: Drive mounted
```

### BOOT.2 — clone or update the private repo with the GH_TOKEN secret

```text
946319e train(notebook): add the batch-32 timing notebook for the RTX PRO 6000
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
git_commit         946319e
disk_free_gb_root  4.5
disk_free_gb_fast  112.5
utc_time           2026-11-07T21:26:07+00:00
=== BEGIN reports/env.json ===
{
  "coqui_tts": "0.28.0",
  "cpu_count": 48,
  "disk_free_gb_fast": 112.5,
  "disk_free_gb_root": 4.5,
  "git_commit": "946319e",
  "gpu_mem_gb": 95.0,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "is_colab": true,
  "numpy": "2.4.6",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "torch": "2.14.1+cu130",
  "torch_cuda": "13.0",
  "torch_error": null,
  "utc_time": "2026-11-07T21:26:07+00:00"
}
=== END reports/env.json ===
PASS: env recorded
RUN-HEADER chunk=C8 run=1 branch=chunk/C8-finetune commit=946319e gpu=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition utc=2026-11-07T21:26:07+00:00
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
 'git_commit': '946319e',
 'disk_free_gb_root': 4.5,
 'disk_free_gb_fast': 112.5,
 'utc_time': '2026-11-07T21:26:07+00:00'}
```

### C8T.pre — Restore prepared data on this VM

```text
SKIP download (done)
SKIP verify (done)
SKIP unzip (done)
copy to fast disk: 100% 8005/8005 [02:43<00:00, 49.09file/s]
DONE copy
PASS: dataset ready at /content/boinga-fast/data, 8005 files
selection: 7506 clips, 0 excluded by char_map
restore: 100% 7512/7512 [00:04<00:00, 1527.45file/s]
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
CELL-OK C8T.pre
```

### C8T.1 — Build and validate config

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
  "batch_size": 32,
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
  "eval_batch_size": 16,
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
| `batch_size` | 104 | 32 | RTX PRO 6000 (95 GB): batch 16 used 3.39 GB; 2x it, lr kept |
| `datasets` | [{"name": "", "path": "/home/ansary/Shabab/bn_tts/male/mo... | [{"formatter": "ljspeech", "dataset_name": "", "path": "/... | our prepared ljspeech manifests on the fast disk replace the author's paths |
| `epochs` | 5000 | 1000 | upper bound only; runs stop by a step budget (RUN_BUDGETS, C8) |
| `eval_batch_size` | 48 | 16 | half the train batch, as before |
| `lr_disc` | 0.0002 | 0.0001 | 0.5 x pretrained, like lr_gen, so G and D stay balanced |
| `lr_gen` | 0.0002 | 0.0001 | 0.5 x pretrained: fine-tuning should not undo the pretrained weights |
| `max_audio_len` | Infinity | 330750 | 15 s x sample_rate: kept from C7; a batch pads to its longest clip |
| `min_audio_len` | 1 | 11025 | 0.5 s x sample_rate, on trimmed clips (C6 follow-up) |
| `mixed_precision` | false | true | fp16: no NaN in the smoke test, less memory; NaN -> fp32 |
| `num_eval_loader_workers` | 4 | 4 | eval is small; half the train workers |
| `num_loader_workers` | 4 | 8 | fixed 8 for the RTX PRO 6000 VM; check env cpu_count |
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
sample_rate 22050, batch_size 32, lr_gen 0.0001, lr_disc 0.0001, mixed_precision True
PASS: coqui accepts config
CELL-OK C8T.1
```

### C8T.2 — One-batch loader check

```text
get_data_loader(config: coqpit.coqpit.Coqpit, *, is_eval: bool = False, samples: list[typing.Any] | None = None, verbose: bool = True) -> 'DataLoader'
samples: 6415 train / 338 eval
100% 6415/6415 [00:00<00:00, 39548.78it/s]
দে চোয়ার.

আঁই ন খাই .

ইতা কি?

ঐ দিন পূর্ণিমা?

বোথ ভালা মাইয়া.

ইয়ান ঠিক.

অনে কি হুইননুন ?

আইয়ী .

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
first batch (incl. worker startup + spec/mel, not a training speed) in 4.04 s
tokens: shape (32, 29) int64, ids 0..86 of vocab 87
token_lens: 13..29
token_rel_lens: shape (32,) float32, nan 0, inf 0
waveform: shape (32, 1, 22402) float32, nan 0, inf 0
waveform_lens: 17210..22402 samples, seconds: 0.78..1.02
waveform_rel_lens: shape (32,) float32, nan 0, inf 0
speaker_names: list of 32
language_names: list of 32
audio_files: list of 32
raw_text: list of 32
audio_unique_names: list of 32
spec: shape (32, 513, 87) float32, nan 0, inf 0
mel: shape (32, 80, 87) float32, nan 0, inf 0
spec_lens: shape (32,) int32
mel_lens: shape (32,) int32
PASS: loader ok
CELL-OK C8T.2
```

### C8T.3 — Fresh smoke run

```text
PASS: cleaned run smoke: removed /content/drive/MyDrive/boinga-translator/runs/smoke (2.00 GB; Drive keeps deleted files in its trash until emptied)
CELL-OK C8T.3
```

### C8T.4 — Smoke test: 50 steps

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
  "batch_size": 32,
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
  "eval_batch_size": 16,
  "eval_split_max_size": null,
  "eval_split_size": 0.01,
  "feat_loss_alpha": 1.0,
  "gen_loss_alpha": 1.0,
... [475 lines trimmed] ...
=== END configs/vits_ctg_finetune.json ===
=== BEGIN reports/config_overrides.md ===
=== END reports/config_overrides.md ===
PASS: config written to /content/boinga-translator/configs/vits_ctg_finetune.json (base /content/drive/MyDrive/boinga-translator/pretrained/male/config.json, 23 overrides)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
     | > loss_duration: 1.959580421447754  (2.5844697884150913)
     | > amp_scaler: 512.0  (1360.4571428571428)
     | > loss_1: 63.05112075805664  (84.04218760899136)
     | > grad_norm_1: tensor(109.9516, device='cuda:0')  (tensor(127.9866, device='cuda:0'))
     | > step_time: 0.5454  (0.6713272367204939)
     | > loader_time: 0.003  (0.002877249036516462)


   --> TIME: 2026-11-07 21:32:05 -- STEP: 40/200 -- GLOBAL_STEP: 40
     | > current_lr_0: 9.950121682254156e-05  (9.974416591222455e-05)
     | > current_lr_1: 9.95883239353732e-05  (9.982460852372438e-05)
     | > loss_disc: 4.239755153656006  (7.096866464614868)
     | > loss_disc_real_0: 1.501986026763916  (1.9195430763065822)
     | > loss_disc_real_1: 0.2053098976612091  (0.5336962714791297)
     | > loss_disc_real_2: 0.31579649448394775  (0.4860171303153038)
     | > loss_disc_real_3: 0.3149688243865967  (0.5332403507083654)
     | > loss_disc_real_4: 0.30725470185279846  (0.4353620857000351)
     | > loss_disc_real_5: 0.42210668325424194  (1.8554310999810695)
     | > loss_0: 4.239755153656006  (7.096866464614868)
     | > grad_norm_0: tensor(74.7578, device='cuda:0')  (tensor(54.2874, device='cuda:0'))
     | > loss_gen: 3.316446304321289  (3.593953639268875)
     | > loss_kl: 6.080410003662109  (13.987549185752863)
     | > loss_feat: 22.902284622192383  (25.517984247207643)
     | > loss_mel: 31.055503845214844  (35.88334331512451)
     | > loss_duration: 1.8894914388656616  (2.4974357962608336)
     | > amp_scaler: 512.0  (1254.4)
     | > loss_1: 65.24413299560547  (81.48026552200318)
     | > grad_norm_1: tensor(80.0766, device='cuda:0')  (tensor(123.7773, device='cuda:0'))
     | > step_time: 0.8478  (0.6701283752918242)
     | > loader_time: 0.003  (0.0028982579708099363)


   --> TIME: 2026-11-07 21:32:08 -- STEP: 45/200 -- GLOBAL_STEP: 45
     | > current_lr_0: 9.943904410714931e-05  (9.971302637580806e-05)
     | > current_lr_1: 9.952609679164422e-05  (9.979420585859768e-05)
     | > loss_disc: 3.216707706451416  (6.682822974522908)
     | > loss_disc_real_0: 0.6026946306228638  (1.788396123382781)
     | > loss_disc_real_1: 0.2576993703842163  (0.5065510378943547)
     | > loss_disc_real_2: 0.24336078763008118  (0.4682638121975793)
     | > loss_disc_real_3: 0.26502060890197754  (0.5069873170720206)
     | > loss_disc_real_4: 0.2639666795730591  (0.4158682495355606)
     | > loss_disc_real_5: 0.273800790309906  (1.6890298432774014)
     | > loss_0: 3.216707706451416  (6.682822974522908)
     | > grad_norm_0: tensor(30.0244, device='cuda:0')  (tensor(51.6747, device='cuda:0'))
     | > loss_gen: 3.211082935333252  (3.5612246407402885)
     | > loss_kl: 5.754144668579102  (13.080645158555768)
     | > loss_feat: 21.24453353881836  (25.061795255872937)
     | > loss_mel: 29.439838409423828  (35.29502042134603)
     | > loss_duration: 1.8906601667404175  (2.4343218326568605)
     | > amp_scaler: 512.0  (1171.911111111111)
     | > loss_1: 61.54025650024414  (79.43300662570529)
     | > grad_norm_1: tensor(117.4498, device='cuda:0')  (tensor(121.7190, device='cuda:0'))
     | > step_time: 0.5459  (0.6566953712039522)
     | > loader_time: 0.0034  (0.0029330200619167753)

STOP at step 50

 > CHECKPOINT : /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_09+31PM-946319e/checkpoint_50.pth
PASS: run smoke at step 50 (peak GPU memory 5.642 GB)
CELL-OK C8T.4
```

### C8T.5 — Resume test: to 60 steps

```text
RESUME from step 50 in /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_09+31PM-946319e
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
I0000 00:00:1794087151.401440   35122 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1794087151.433194   35122 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1794087152.462469   35122 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1794087152.462737   35122 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
 > Start Tensorboard: tensorboard --logdir=/content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_09+31PM-946319e
 > Continuing from checkpoint_50.pth ...
 > Restoring Model...
 > Restoring Optimizer...
 > Restoring Scheduler...
 > Restoring Scaler...
 > Model restored from step 50

 > Model has 83050732 parameters
 > Restoring best loss from checkpoint_50.pth ...
 > Starting with loaded last best loss {'train_loss': 42.19845173796829, 'eval_loss': None}

 > EPOCH: 0/999
 --> /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_09+31PM-946319e
100% 6415/6415 [00:00<00:00, 41264.60it/s]

 > TRAINING (2026-11-07 21:32:35) 
ইয়ান ঠিক.

আঁই ন খাই .

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
বোথ ভালা মাইয়া.

আইয়ী .

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
অনে কি হুইননুন ?

ইতা কি?

ঐ দিন পূর্ণিমা?

দে চোয়ার.

Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.
Character '\n' not found in the vocabulary. Discarding it.

   --> TIME: 2026-11-07 21:32:42 -- STEP: 4/200 -- GLOBAL_STEP: 55
     | > current_lr_0: 9.932723110067987e-05  (9.934585883720885e-05)
     | > current_lr_1: 9.941418589985758e-05  (9.943282994380871e-05)
     | > loss_disc: 3.5701558589935303  (3.525189459323883)
     | > loss_disc_real_0: 0.7754753232002258  (0.8105775862932205)
     | > loss_disc_real_1: 0.26068827509880066  (0.2328721545636654)
     | > loss_disc_real_2: 0.27828001976013184  (0.3432232439517975)
     | > loss_disc_real_3: 0.27729639410972595  (0.31399966776371)
     | > loss_disc_real_4: 0.20120126008987427  (0.2609447240829468)
     | > loss_disc_real_5: 0.4550367593765259  (0.45200228691101074)
     | > loss_0: 3.5701558589935303  (3.525189459323883)
     | > grad_norm_0: tensor(35.4904, device='cuda:0')  (tensor(32.3544, device='cuda:0'))
     | > loss_gen: 3.225921630859375  (3.5651991963386536)
     | > loss_kl: 5.757418632507324  (5.693405270576477)
     | > loss_feat: 27.36067771911621  (27.79255437850952)
     | > loss_mel: 29.80141830444336  (30.795164585113525)
     | > loss_duration: 1.9451580047607422  (2.0184335112571716)
     | > amp_scaler: 512.0  (512.0)
     | > loss_1: 68.09059143066406  (69.86475563049316)
     | > grad_norm_1: tensor(124.9009, device='cuda:0')  (tensor(98.7847, device='cuda:0'))
     | > step_time: 0.8389  (0.760942816734314)
     | > loader_time: 0.0018  (0.0033981800079345703)

STOP at step 60

 > CHECKPOINT : /content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_09+31PM-946319e/checkpoint_60.pth
PASS: run smoke at step 60 (peak GPU memory 5.249 GB)
CELL-OK C8T.5
```

### C8T.6 — Throughput

```text
=== BEGIN reports/throughput.json ===
{
  "run": "smoke",
  "folder": "/content/drive/MyDrive/boinga-translator/runs/smoke/smoke-November-07-2026_09+31PM-946319e",
  "steps_done": 60,
  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "batch_size": 32,
  "mixed_precision": true,
  "sec_per_step_mean": 0.7262,
  "n_step_samples": 10,
  "warmup_only": false,
  "steps_per_hour": 4957.3,
  "sec_per_step_coqui_avg": 0.760942816734314,
  "any_nan_loss": false,
  "last_losses": {
    "loss_disc": 3.5701558589935303,
    "loss_disc_real_0": 0.7754753232002258,
    "loss_disc_real_1": 0.26068827509880066,
    "loss_disc_real_2": 0.27828001976013184,
    "loss_disc_real_3": 0.27729639410972595,
    "loss_disc_real_4": 0.20120126008987427,
    "loss_disc_real_5": 0.4550367593765259,
    "loss_0": 3.5701558589935303,
    "loss_gen": 3.225921630859375,
    "loss_kl": 5.757418632507324,
    "loss_feat": 27.36067771911621,
    "loss_mel": 29.80141830444336,
    "loss_duration": 1.9451580047607422,
    "loss_1": 68.09059143066406
  },
  "last_logged_step": 55,
  "peak_mem_gb": 5.642
}
=== END reports/throughput.json ===
PASS: throughput recorded: 0.7262 s/step (10 samples), peak 5.642 GB, losses from logged step 55 of 60
CELL-OK C8T.6
```
