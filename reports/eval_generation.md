# Evaluation audio

20 prompts (configs/eval_prompts.json), audio on Drive eval/final/. Voices use the fixed seed and inference scales; R_real is the corpus clip through prepare.transform_audio.

| set | wavs | mean duration s | min s | max s | mean RMS dBFS |
|---|---|---|---|---|---|
| A_baseline | 20 | 3.59 | 2.38 | 6.76 | -24.1 |
| B_boinga | 20 | 2.83 | 1.61 | 5.22 | -17.7 |
| R_real | 20 | 2.90 | 1.87 | 4.63 | -16.6 |
