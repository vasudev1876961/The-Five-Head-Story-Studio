# Phase 5 Reproducibility Verification Report

- **Run ID:** `run_b42_s1042_n200`
- **Overall Status:** **REPRODUCED_EXACT_AND_EQUIVALENT**
- **Deterministic Base Seed:** `42` (for `E1, E2, E3, E4, E5, E7`)
- **Stochastic Comparison Seed:** `1042` (for `E6, E8`)
- **Stories per Condition:** `200` | **Total Generated:** `1600` | **Failures:** `0`
- **Wall Time:** `251.93 s` (`6.35 stories/sec`)
- **Canonical E6 Decoder Verified:** `top_k(k=5)`

## 1. Environment & Configuration Hashes

- **Python:** `3.12.8` (`Windows-11-10.0.22621-SP0`)
- **NumPy / Pandas:** `numpy 1.26.4`, `pandas 2.3.3`
- **Canonical Config Hash:** `CE63533CA8C37379910642B087B117F2100307B3BB5B9B853C73C8645B9617E0`
- **Reproduced Registry Hash:** `CE63533CA8C37379910642B087B117F2100307B3BB5B9B853C73C8645B9617E0`

## 2. Configuration Verdicts

| Config | Type | Seed Offset Rule | Verdict | Details |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | Deterministic | `base + 0*10000 + i` | **EXACT_REPRODUCTION** | 200/200 exact token matches |
| **E2** | Deterministic | `base + 1*10000 + i` | **EXACT_REPRODUCTION** | 200/200 exact token matches |
| **E3** | Deterministic | `base + 2*10000 + i` | **EXACT_REPRODUCTION** | 200/200 exact token matches |
| **E4** | Deterministic | `base + 3*10000 + i` | **EXACT_REPRODUCTION** | 200/200 exact token matches |
| **E5** | Deterministic | `base + 4*10000 + i` | **EXACT_REPRODUCTION** | 200/200 exact token matches |
| **E6** | Stochastic | `stoch + 5*10000 + i` | **STATISTICAL_EQUIVALENCE** | All 6 pre-registered metrics passed Holm-adjusted TOST equivalence. |
| **E7** | Deterministic | `base + 6*10000 + i` | **EXACT_REPRODUCTION** | 200/200 exact token matches |
| **E8** | Stochastic | `stoch + 7*10000 + i` | **STATISTICAL_EQUIVALENCE** | All 6 pre-registered metrics passed Holm-adjusted TOST equivalence. |

## 3. Stochastic Statistical Equivalence Analysis (TOST + Holm-Bonferroni)

Family-wise error rate $\alpha = 0.05$ across $m = 12$ pre-registered tests.

| Config | Metric | Canon Mean (SD) | Repro Mean (SD) | Diff | Hedges' g | Margin +/- | Holm TOST p | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| E6 | `length_total_tokens` | 94.6300 (11.1027) | 93.6650 (10.5236) | -0.9650 | -0.089 | +/-4.0 | 0.01053 | **STATISTICAL_EQUIVALENCE** |
| E6 | `repetition_repeated_bigram_rate` | 0.1917 (0.0477) | 0.1919 (0.0428) | +0.0002 | +0.003 | +/-0.025 | 3.082e-07 | **STATISTICAL_EQUIVALENCE** |
| E6 | `diversity_ttr` | 0.2888 (0.0442) | 0.2864 (0.0377) | -0.0024 | -0.060 | +/-0.02 | 8.467e-05 | **STATISTICAL_EQUIVALENCE** |
| E6 | `setting_setting_consistency_ratio` | 0.7327 (0.2021) | 0.7245 (0.2012) | -0.0082 | -0.041 | +/-0.06 | 0.01584 | **STATISTICAL_EQUIVALENCE** |
| E6 | `action_dynamic_action_ratio` | 0.9831 (0.0213) | 0.9814 (0.0198) | -0.0016 | -0.079 | +/-0.04 | 2.268e-62 | **STATISTICAL_EQUIVALENCE** |
| E6 | `character_character_continuity_ratio` | 0.3321 (0.3116) | 0.3531 (0.3192) | +0.0210 | +0.067 | +/-0.09 | 0.02938 | **STATISTICAL_EQUIVALENCE** |
| E8 | `length_total_tokens` | 104.2650 (11.2435) | 102.4700 (12.4767) | -1.7950 | -0.151 | +/-4.0 | 0.03205 | **STATISTICAL_EQUIVALENCE** |
| E8 | `repetition_repeated_bigram_rate` | 0.0061 (0.0087) | 0.0063 (0.0086) | +0.0002 | +0.023 | +/-0.025 | 3.516e-98 | **STATISTICAL_EQUIVALENCE** |
| E8 | `diversity_ttr` | 0.6780 (0.0431) | 0.6829 (0.0446) | +0.0049 | +0.112 | +/-0.02 | 0.001952 | **STATISTICAL_EQUIVALENCE** |
| E8 | `setting_setting_consistency_ratio` | 0.2095 (0.0541) | 0.2150 (0.0620) | +0.0055 | +0.094 | +/-0.06 | 2.686e-18 | **STATISTICAL_EQUIVALENCE** |
| E8 | `action_dynamic_action_ratio` | 0.6596 (0.1032) | 0.6549 (0.1023) | -0.0047 | -0.046 | +/-0.04 | 0.001952 | **STATISTICAL_EQUIVALENCE** |
| E8 | `character_character_continuity_ratio` | 0.0710 (0.0806) | 0.0735 (0.0822) | +0.0025 | +0.031 | +/-0.09 | 3.826e-23 | **STATISTICAL_EQUIVALENCE** |
