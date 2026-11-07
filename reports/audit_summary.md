# Corpus audit summary

Charset: female-a4543ed63b. Rows: 8009; decodable clips: 8004, 9.04 h.

## Per split

| split | clips | hours | clean clips | clean hours |
|---|---|---|---|---|
| (none) | 5 | 0.00 | 0 | 0.00 |
| test | 800 | 0.89 | 299 | 0.42 |
| train | 6403 | 7.26 | 2324 | 3.30 |
| val | 801 | 0.89 | 258 | 0.37 |

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
| oov (oov_ratio > 0.02) | 5015 | 4.74 |
| speech rate (chars/s outside 1st-99th percentile, or no speech) | 167 | 0.13 |
| duplicate text | 28 | 0.02 |

oov only because of the danda । (the model's eos symbol):

| clips |
|---|
| 4711 |

Clean: 2881 clips, 4.10 h.

## Speakers

no speaker field in the corpus (DECISIONS "Corpus layout")

## Characters not in the model (top 20, rows containing)

| char | code | name | rows |
|---|---|---|---|
| । | U+0964 | DEVANAGARI DANDA | 6254 |
| ৰ | U+09F0 | BENGALI LETTER RA WITH MIDDLE DIAGONAL | 59 |
| ৭ | U+09ED | BENGALI DIGIT SEVEN | 50 |
| ৯ | U+09EF | BENGALI DIGIT NINE | 48 |
| " | U+0022 | QUOTATION MARK | 46 |
| ( | U+0028 | LEFT PARENTHESIS | 45 |
| ) | U+0029 | RIGHT PARENTHESIS | 44 |
| ' | U+0027 | APOSTROPHE | 41 |
| – | U+2013 | EN DASH | 41 |
| / | U+002F | SOLIDUS | 19 |
| : | U+003A | COLON | 7 |
| ৷ | U+09F7 | BENGALI CURRENCY NUMERATOR FOUR | 4 |
| “ | U+201C | LEFT DOUBLE QUOTATION MARK | 2 |
| * | U+002A | ASTERISK | 1 |
| ৱ | U+09F1 | BENGALI LETTER RA WITH LOWER DIAGONAL | 1 |
| … | U+2026 | HORIZONTAL ELLIPSIS | 1 |
