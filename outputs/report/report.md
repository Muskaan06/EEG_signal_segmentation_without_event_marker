# EEG Reading vs. Non-Reading Classification — Report

## Dataset

- Task: NR
- Subjects: 3 ['ZAB', 'ZDN', 'ZJM']
- Sentences: 900
- Word (reading) events recovered: 12945
- Sampling rate: 500.0 Hz
- Channels (raw): 128

## Preprocessing

- **ZAB_NR1**: line freq=50.0Hz, 32 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZAB_NR2**: line freq=50.0Hz, 33 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZAB_NR3**: line freq=50.0Hz, 24 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZAB_NR4**: line freq=50.0Hz, 15 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZAB_NR5**: line freq=50.0Hz, 23 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZAB_NR6**: line freq=50.0Hz, 17 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZDN_NR1**: line freq=50.0Hz, 17 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZDN_NR2**: line freq=50.0Hz, 17 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZDN_NR3**: line freq=50.0Hz, 5 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZDN_NR4**: line freq=50.0Hz, 2 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZDN_NR5**: line freq=50.0Hz, 30 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZDN_NR6**: line freq=50.0Hz, 14 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZJM_NR1**: line freq=50.0Hz, 1 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZJM_NR2**: line freq=50.0Hz, 50 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZJM_NR3**: line freq=50.0Hz, 1 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZJM_NR4**: line freq=50.0Hz, 50 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZJM_NR5**: line freq=50.0Hz, 51 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components
- **ZJM_NR6**: line freq=50.0Hz, 50 bad channels interpolated, ICA excluded 0 EOG + 0 muscle components

## ROIs (scalp-location approximations)

- **occipital**: ['E70', 'E74', 'E75', 'E82', 'E83']
- **vwfa**: ['E50', 'E58', 'E64', 'E69']
- **pstg**: ['E44', 'E45', 'E57']
- **wernicke**: ['E39', 'E40']
- **broca**: ['E26', 'E33', 'E38']
- **parietal**: ['E41', 'E42', 'E47', 'E52', 'E53']

## Most discriminative features (READING vs NON-READING)

Ranked by |Cohen's d|, with paired t-test (across subjects) and label-permutation p-values:

| feature                           |   cohens_d |   paired_t |   paired_p |   perm_observed_diff |      perm_p |   abs_cohens_d |
|:----------------------------------|-----------:|-----------:|-----------:|---------------------:|------------:|---------------:|
| pstg_wernicke_beta_plv            |   0.381921 |   2.15685  |  0.163735  |          0.057306    | 0.000999001 |       0.381921 |
| parietal_pstg_beta_plv            |   0.354835 |   1.45936  |  0.281872  |          0.0779898   | 0.000999001 |       0.354835 |
| pstg_wernicke_beta_coherence      |   0.275114 |   1.17627  |  0.360534  |          0.0332013   | 0.000999001 |       0.275114 |
| occipital_P1_auc                  |   0.272081 |   1.96816  |  0.187906  |         50.2099      | 0.000999001 |       0.272081 |
| occipital_P1_mean                 |   0.270302 |   1.97005  |  0.187641  |          1.24016     | 0.000999001 |       0.270302 |
| occipital_beta_abs_power          |   0.266495 |  -0.800427 |  0.507435  |          9.70022e-13 | 0.000999001 |       0.266495 |
| parietal_pstg_gamma_plv           |   0.265478 |   0.772071 |  0.520822  |          0.0632482   | 0.000999001 |       0.265478 |
| occipital_P1_peak                 |   0.265323 |   2.58516  |  0.122694  |          2.09882     | 0.000999001 |       0.265323 |
| parietal_wernicke_beta_plv        |   0.255271 |  -0.193174 |  0.864662  |          0.0589728   | 0.000999001 |       0.255271 |
| parietal_wernicke_delta_coherence |   0.248796 |   3.60125  |  0.069199  |          0.0596238   | 0.000999001 |       0.248796 |
| parietal_pstg_beta_coherence      |   0.248621 |   0.385535 |  0.736984  |          0.0356697   | 0.000999001 |       0.248621 |
| parietal_wernicke_beta_coherence  |   0.238182 |   0.303555 |  0.790134  |          0.0377838   | 0.000999001 |       0.238182 |
| occipital_gamma_abs_power         |   0.233217 |   1.43684  |  0.287303  |          6.67101e-13 | 0.000999001 |       0.233217 |
| parietal_wernicke_theta_coherence |   0.230033 |   3.17864  |  0.0863469 |          0.0521504   | 0.000999001 |       0.230033 |
| vwfa_P1_peak                      |   0.221879 |   4.63314  |  0.0435637 |          1.5948      | 0.000999001 |       0.221879 |

## Model comparison

| protocol            | model              |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:--------------------|:-------------------|-----------:|------------:|---------:|---------:|----------:|
| subject_dependent   | LogisticRegression |   0.611285 |    0.625907 | 0.615059 | 0.618003 |  0.620875 |
| subject_dependent   | MLP                |   0.629672 |    0.626206 | 0.655323 | 0.638081 |  0.630149 |
| subject_dependent   | RandomForest       |   0.678355 |    0.663344 | 0.709379 | 0.631149 |  0.648182 |
| subject_dependent   | SVM                |   0.666116 |    0.657046 | 0.698165 | 0.608336 |  0.661581 |
| subject_dependent   | XGBoost            |   0.655292 |    0.638676 | 0.669682 | 0.639697 |  0.636906 |
| subject_independent | LogisticRegression |   0.515496 |    0.621201 | 0.376072 | 0.450087 |  0.574714 |
| subject_independent | MLP                |   0.523929 |    0.615637 | 0.396195 | 0.465816 |  0.570626 |
| subject_independent | RandomForest       |   0.42213  |    0.625625 | 0.40119  | 0.331469 |  0.591414 |
| subject_independent | SVM                |   0.456701 |    0.583418 | 0.514616 | 0.451043 |  0.594661 |
| subject_independent | XGBoost            |   0.472579 |    0.595408 | 0.377015 | 0.391518 |  0.577443 |

## Deep learning models (raw epochs, subject-independent)

| model     |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:----------|-----------:|------------:|---------:|---------:|----------:|
| CNNLSTM   |   0.496604 |    0.571454 | 0.532865 | 0.523943 |  0.550229 |
| EEGNet    |   0.505358 |    0.647368 | 0.506403 | 0.497256 |  0.613684 |
| SimpleCNN |   0.552364 |    0.642016 | 0.575575 | 0.572812 |  0.631635 |

## Interpretation

- Most informative ROIs (by frequency in top-15 discriminative features): {'parietal': np.int64(7), 'occipital': np.int64(5), 'pstg': np.int64(2)}
- ERP-component features in the top 15: ['occipital_P1_auc', 'occipital_P1_mean', 'occipital_P1_peak', 'vwfa_P1_peak']
- Frequency-band features in the top 15: ['pstg_wernicke_beta_plv', 'parietal_pstg_beta_plv', 'pstg_wernicke_beta_coherence', 'occipital_beta_abs_power', 'parietal_pstg_gamma_plv', 'parietal_wernicke_beta_plv', 'parietal_wernicke_delta_coherence', 'parietal_pstg_beta_coherence', 'parietal_wernicke_beta_coherence', 'occipital_gamma_abs_power', 'parietal_wernicke_theta_coherence']

**Can EEG reliably tell READING from NON-READING without decoding word identity?** On this validation subset, the best subject-independent (Leave-One-Subject-Out) model was **SVM** with accuracy=0.46, F1=0.45, ROC-AUC=0.59. 
This is a binary reading-state classification (not word-identity decoding): labels only distinguish whether a word was being fixated/read vs. genuinely non-reading intervals (pre-block, inter-trial, inter-word gaps) from the same recordings, so any above-chance performance reflects general reading-state signatures (e.g. posterior/temporal ERP negativity, alpha/theta power changes) rather than which word was read.