# Iowa Results Included in This Release

These values come from the saved research outputs, not the synthetic example.

| Component | Saved result | Source |
|---|---|---|
| Candidate histories | 14,379 arrays of shape 7 x 60 x 5 | Step 5 preparation metadata |
| Grouping fit | 2,043 nonoverlapping histories | Step 5 preparation metadata |
| PCA | 45 retained components, approximately 95.03% variance | Step 5 PCA and summary files |
| K-means | Seven selected groups | Step 5 K diagnostics and summary |
| Model-development nowcasts | STRK MAE 0.812 degrees C over 10,213 targets | Step 7 overall CV metrics |
| Protected-event evaluation | STRK MAE 0.752 degrees C over 15,906 targets in 289 events | Step 8 overall metrics |
| Exact fixed-camera test hours represented in the histories | STRK MAE 0.667 degrees C over 440 targets in eight events | Step 8 overall metrics |
| Parent selection | Ten near-centroid histories per group; 70 parents | Step 9 selection summary |
| SGS | Ten realizations per parent; 700 total | Step 10 manifest |
| Support network | 122,329 support points | Step 10 manifest |
| Conditioning | 420 completed station-hour RST values per parent | Step 10 manifest |

The three MAE entries describe different evaluation designs and target sets. They should not be treated as repeated estimates of a single performance quantity. In particular, they are not the later operational simulation's group-calibrated station MAE or networkwide prediction MAE.

The exported model tables preserve the seven fitted regression designs, covariance-family comparisons, feature scaling, coefficient uncertainty, VIF, and whether parameters reached fitting bounds. The SGS statistics summarize the generated fields; they do not validate unobserved real-world road temperatures.
