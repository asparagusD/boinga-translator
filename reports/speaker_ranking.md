# Speaker clusters

Encoder: resemblyzer-0.1.4. Clean clips with an embedding: 7513. Agglomerative clustering, cosine distance, average linkage. Chosen threshold 0.50: best silhouette among thresholds whose largest cluster has >= 30 clean minutes, else the one with the most.

| threshold | clusters | largest cluster min | silhouette | chosen |
|---|---|---|---|---|
| 0.20 | 1464 | 51.64 | 0.042 |  |
| 0.25 | 568 | 65.70 | 0.049 |  |
| 0.30 | 187 | 72.08 | 0.125 |  |
| 0.35 | 71 | 128.35 | 0.108 |  |
| 0.40 | 20 | 395.17 | 0.011 |  |
| 0.45 | 8 | 452.77 | 0.109 |  |
| 0.50 | 3 | 490.73 | 0.196 | yes |

## Clusters ranked by clean minutes (top 20)

| rank | clips | clean min | train min | val min | test min | clip ids |
|---|---|---|---|---|---|---|
| 1 | 7506 | 490.73 | 393.72 | 48.25 | 48.76 | 4351, 5466, 6654, 7478, 941 |
| 2 | 5 | 0.18 | 0.12 | 0.00 | 0.06 | 5048, 5101, 5604, 6659, 6671 |
| 3 | 2 | 0.11 | 0.00 | 0.11 | 0.00 | 6490, 6504 |
