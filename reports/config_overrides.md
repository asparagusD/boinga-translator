# Fine-tune config overrides

Base: `/content/drive/MyDrive/boinga-translator/pretrained/male/config.json`, run `run1`.

| key | pretrained | fine-tune | reason |
|---|---|---|---|
| `batch_size` | 104 | 16 | measured best on the RTX PRO 6000: 46.8 clips/s vs 44.1 at batch 32 |
| `datasets` | [{"name": "", "path": "/home/ansary/Shabab/bn_tts/male/mo... | [{"formatter": "ljspeech", "dataset_name": "", "path": "/... | our prepared ljspeech manifests on the fast disk replace the author's paths |
| `epochs` | 5000 | 1000 | upper bound only; runs stop by a step budget (RUN_BUDGETS, C8) |
| `eval_batch_size` | 48 | 8 | half the train batch |
| `lr_disc` | 0.0002 | 0.0001 | 0.5 x pretrained, like lr_gen, so G and D stay balanced |
| `lr_gen` | 0.0002 | 0.0001 | 0.5 x pretrained: fine-tuning should not undo the pretrained weights |
| `max_audio_len` | Infinity | 330750 | 15 s x sample_rate: kept from C7; a batch pads to its longest clip |
| `min_audio_len` | 1 | 11025 | 0.5 s x sample_rate, on trimmed clips (C6 follow-up) |
| `mixed_precision` | false | true | fp16: no NaN in the smoke test, less memory; NaN -> fp32 |
| `num_eval_loader_workers` | 4 | 4 | eval is small; half the train workers |
| `num_loader_workers` | 4 | 8 | RTX PRO 6000 VM has 48 CPUs (env.json); no speed change seen |
| `output_path` | "/home/ansary/Shabab/" | "/content/drive/MyDrive/boinga-translator/runs" | runs on Drive (paths.runs()) survive a Colab disconnect |
| `print_step` | 50 | 25 | step logs often enough to read sec/step in the smoke test |
| `restore_path` | "/home/ansary/Shabab/vits_20_october/checkpoint_626000.pth" | null | the author's absolute path; the runner passes it (C7.3) |
| `run_eval` | true | true | val loss picks the best checkpoint, with listening (vits-finetune) |
| `run_name` | "vits_20_october" | "run1" | given per run (boinga build-config --run NAME) |
| `save_best_after` | 500 | 1000 | no best checkpoint from the noisy first steps |
| `save_checkpoints` | true | true | resume after a Colab disconnect needs them |
| `save_n_checkpoints` | 5 | 2 | keep the last 2 (+ best): ~1 GB each, Drive budget |
| `save_step` | 500 | 500 | a checkpoint every ~500 steps to Drive (vits-finetune) |
| `test_delay_epochs` | -1 | 0 | test sentences from the first epoch: early sanity listening |
| `test_sentences` | [["ইয়াহুদ,নয়,ব্যয়,ইয়াহুদী,নেওয়া,উভয়ে,যায়,হয়েছিল,প্রয়োজন।"]] | [["এ কাজল্লাই খরচ দিয়েদে এ কে খান ফাউন্ডেশন."], ["অনে কন ... | first 5 val manifest texts: prepare_text output as trained, not re-normalised (Gate G4) |
| `training_seed` | 54321 | 1234 | the project seed (settings.SEED) for reproducible runs |
