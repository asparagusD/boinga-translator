# Corpus audit summary

Charset: female-a4543ed63b-151ac177e2. Rows: 8009; decodable clips: 8004, 9.04 h.

## Per split

| split | clips | hours | clean clips | clean hours |
|---|---|---|---|---|
| (none) | 5 | 0.00 | 0 | 0.00 |
| test | 800 | 0.89 | 755 | 0.81 |
| train | 6403 | 7.26 | 6010 | 6.56 |
| val | 801 | 0.89 | 748 | 0.81 |

## Audio formats

| sample_rate | channels | subtype | clips |
|---|---|---|---|
| 16000 | 1 | PCM_16 | 8004 |

## Exclusions (a clip can have several reasons)

| reason | clips | hours |
|---|---|---|
| undecodable (or no wav) | 5 | 0.00 |
| too short (< 1 s) | 8 | 0.00 |
| too long (> 15 s) | 29 | 0.14 |
| clipped (clip_ratio > 0.001) | 7 | 0.01 |
| quiet (rms_db < -45) | 0 | 0.00 |
| empty text | 0 | 0.00 |
| excluded character (char_map: / : * digits) | 279 | 0.62 |
| unknown to the model after char_map | 114 | 0.24 |
| speech rate (chars/s outside 1st-99th percentile, or no speech) | 167 | 0.13 |
| duplicate text | 28 | 0.02 |

Clean: 7513 clips, 8.18 h.

## Speakers

no speaker field in the corpus (DECISIONS "Corpus layout")

## Characters not in the model after char_map (top 20, rows containing)

| char | code | name | rows |
|---|---|---|---|
| ৭ | U+09ED | BENGALI DIGIT SEVEN | 50 |
| ৯ | U+09EF | BENGALI DIGIT NINE | 48 |
| / | U+002F | SOLIDUS | 19 |
| : | U+003A | COLON | 7 |
| * | U+002A | ASTERISK | 1 |
