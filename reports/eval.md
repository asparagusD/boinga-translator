# Blind listening test

1 raters (R1), 20 prompts, 40 ratings. A_baseline = pretrained Bangla voice, B_boinga = fine-tuned voice (Drive best/). Scores 1-5; mean ± sd over all ratings.

| criterion | A_baseline | B_boinga | B − A |
|---|---|---|---|
| intelligibility | 1.40 ± 0.50 | 1.90 ± 0.79 | +0.50 |
| naturalness | 1.10 ± 0.31 | 1.80 ± 0.70 | +0.70 |
| dialect_authenticity | 1.00 ± 0.00 | 1.60 ± 0.68 | +0.60 |
| dialect_words | 1.00 ± 0.00 | 1.60 ± 0.75 | +0.60 |
| overall | 1.12 ± 0.17 | 1.73 ± 0.60 | +0.60 |

## Per prompt

Mean of the 4 criteria; B win rate = share of raters scoring B above A (tie = 0.5).

| prompt | A_baseline | B_boinga | B win rate |
|---|---|---|---|
| P01 | 1.50 | 1.00 | 0.00 |
| P02 | 1.50 | 1.00 | 0.00 |
| P03 | 1.00 | 1.25 | 1.00 |
| P04 | 1.25 | 1.50 | 1.00 |
| P05 | 1.25 | 1.00 | 0.00 |
| P06 | 1.00 | 2.50 | 1.00 |
| P07 | 1.00 | 1.50 | 1.00 |
| P08 | 1.25 | 2.50 | 1.00 |
| P09 | 1.00 | 2.75 | 1.00 |
| P10 | 1.00 | 1.75 | 1.00 |
| P11 | 1.00 | 2.25 | 1.00 |
| P12 | 1.25 | 1.50 | 1.00 |
| P13 | 1.25 | 2.00 | 1.00 |
| P14 | 1.25 | 1.75 | 1.00 |
| P15 | 1.00 | 1.50 | 1.00 |
| P16 | 1.00 | 1.00 | 0.50 |
| P17 | 1.00 | 1.50 | 1.00 |
| P18 | 1.00 | 1.25 | 1.00 |
| P19 | 1.00 | 2.25 | 1.00 |
| P20 | 1.00 | 2.75 | 1.00 |

B_boinga win rate over prompts: 0.82

## Comments

### A_baseline

(none)

### B_boinga

(none)

## Limitations

- Informal panel of 1 rater(s), not a formal MOS test; no significance testing, so small
  differences between A and B may be noise.
- One target voice (fine-tuned on the corpus's mixed clean clips): this measures unseen
  sentences in a known voice, not new speakers.
- The base voice (tts_models/bn/custom/vits-male) was trained on Indian Bengali data; its
  accent can colour both A and B.
- The model reads graphemes (Bengali script, NFC + char map), not phonemes, so spelling
  variation in the dialect text reaches the model directly.
- ASR word error rate (WER) is not used: standard-Bangla ASR penalises correct Chittagonian forms.
