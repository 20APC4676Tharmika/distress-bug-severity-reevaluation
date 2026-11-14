# Developer distress and bug severity: leakage-controlled re-evaluation

Review repository for the article *Does Developer Distress Help Predict Bug Severity? A Leakage-Controlled
Re-evaluation Across Stack Overflow, GitHub, and Bugzilla*. It holds the scripts, configurations, split records,
saved predictions and result tables behind every table and figure (see Table XV, the run record, in the article).

## Layout

```
audit/        records behind the conference-paper account (Section I-B, Table I)
  Severity_Bugzilla.ipynb      original notebook: fixing-time labels, resample(replace=True) on the full corpus, 80:20 split
  Severity_GitHub.ipynb        original notebook: keyword-rule labels, resampling before the split, 80:20 split
data/         raw-data sources and the cleaned splits used in the article
  clean_bugzilla_{train,val,test}.csv   MD5 prefixes 52b63da134, cd246f452d, 043d879265
  clean_github_{train,val,test}.csv     MD5 prefixes 01861445e1, a6ee84872f, f8aa878fe3
  emo_probs_bz_*.npy, emo_probs_bzmask_*.npy, emo_probs_gh_*.npy   frozen emotion-classifier outputs
notebooks/    the Colab notebooks that produced the main results tables
scripts/      revision scripts
  rerun_masked_emotion.py      Bugzilla rerun with severity words masked before all three feature groups
  github_resplits_thresholds.py  GitHub resplits with thresholds computed inside each training partition,
                                 per-class metrics, saved predictions, and the audit counts
results/      result tables (CSV) exactly as produced by the notebooks and scripts
figures/      the article figures (methodology diagram also as draw.io source)
predictions/  test-set predictions and split indices
  bugzilla_main_split_test_predictions.csv, github_main_split_test_predictions.csv
  gh_resplit{0..9}_test_predictions.csv, gh_resplits_indices_thresholds.json, gh_resplits_results.csv
```

## Fixed settings

| Item | Setting |
|---|---|
| Main split | 70/15/15, seed 42; Bugzilla stratified by severity; GitHub split at random, then tertile thresholds from the training resolution times |
| Resplits | ten 70/15/15 resplits, seeds 0-9; GitHub thresholds recomputed inside each training partition |
| TF-IDF | unigrams + bigrams, 20,000 features, min_df 2, sublinear tf, fitted on training data |
| Emotion scores | `j-hartmann/emotion-english-distilroberta-base`, first 256 tokens, frozen |
| Keyword counts | six-category lexicon (Table V of the article) plus total |
| Bugzilla masking | blocker(s), critical(s), major(s), minor(s), trivial, normal, severity, sev, enhancement, regression removed before all three feature groups |
| XGBoost | 300 trees, depth 6, lr 0.1, subsample 0.8, colsample 0.8, balanced sample weights, seeds 42-46 |
| Transformers | 4 epochs, AdamW 2e-5, wd 0.01, 6% warm-up, max length 192, batch 32, seeds 42-44 |

## Data provenance

The Stack Overflow questions were collected by API tag (counts in the article). The other two files come from
public datasets:

- GitHub: the Visual Studio Code subset of GitBugs (CC BY 4.0), https://github.com/av9ash/gitbugs (32,829 issues).
- Bugzilla: the Kaggle dataset "Bugzilla Bug Reports" by Q. Liu (version 1, 2021),
  https://www.kaggle.com/datasets/qicongliu/bugzilla-bug-reports. Its `sev.csv` and `fix.csv` files were saved
  locally as `Bugzilla_sev.csv` and `Bugzilla_fix.csv`. Each holds 32,428 Mozilla reports. The dataset page gives no
  license and no collection method.

No API-selection filter is documented for either file. The text is the report as it stood when the
files were exported and may include later edits; the analysis is retrospective.

## Reproducing

Run the scripts in Google Colab with the files above in `MyDrive/Colab Notebooks/`. Each script prints the numbers
reported in the article and writes its records to `MyDrive/Colab Notebooks/repro/`.
