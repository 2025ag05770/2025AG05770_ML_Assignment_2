# Dry Bean Classifier

A small ML project that classifies dry beans into 7 types from 16 shape measurements, using the UCI Dry Bean Dataset. Five classifiers are trained and compared, and a Streamlit app lets you score any CSV, look at the confusion matrix and compare the models.

## a. Problem statement

Given the 16 numeric shape features of a bean (area, perimeter, axis lengths, shape factors and so on), predict which of 7 bean types it is. This is a multi-class classification problem with imbalanced classes.

## b. Dataset description

UCI Dry Bean Dataset. The training script downloads it from UCI every time it runs, so the raw file is not stored in this repo.

| Item | Value |
|---|---|
| Rows x columns | 13,611 x 17 |
| Features | 16, all numeric |
| Target column | `Class` (7 values) |
| Missing values | 0 |
| Duplicate rows | 68 (kept, not removed) |

Features: `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`. The column names are spelled exactly like this in the source file, including `AspectRation`.

Class balance:

| Class | Rows | Share |
|---|---|---|
| DERMASON | 3,546 | 26.1% |
| SIRA | 2,636 | 19.4% |
| SEKER | 2,027 | 14.9% |
| HOROZ | 1,928 | 14.2% |
| CALI | 1,630 | 12.0% |
| BARBUNYA | 1,322 | 9.7% |
| BOMBAY | 522 | 3.8% |

Data preparation:

- 80/20 stratified split with `random_state=42`: 10,888 training rows and 2,723 test rows. The test split has 104 BOMBAY rows.
- Class names are changed to numbers 0 to 6 with a `LabelEncoder` (saved in `model/`).
- Features are on very different scales (`Area` is in the tens of thousands, `Solidity` is near 1), so every model is a pipeline of `StandardScaler` followed by the classifier. The scaler is fitted on the training split only.
- No other cleaning, no feature selection and no hyper-parameter search.

## c. GitHub repository link

https://github.com/2025ag05770/2025AG05770_ML_Assignment_2

Live Streamlit app: https://2025ag05770mlassignment2-tunfwld5btas3hqzbvmhkg.streamlit.app/

## d. Models used

Settings of the five models (each one is `StandardScaler` followed by the classifier):

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=8`, `min_samples_leaf=5` |
| KNN | `k=7`, `weights="distance"` |
| Naive Bayes (Gaussian) | defaults |
| Support Vector Machine | RBF kernel, `C=10`, `probability=True` (shown as "SVM (RBF)" in the app) |

Metrics, all computed on the 2,723 held-out test rows:

- **Accuracy**: share of correct predictions.
- **AUC**: one-vs-rest, macro average. It is `NaN` if a class is missing from the data being scored.
- **Precision, Recall, F1**: macro averages, so each of the 7 classes counts equally. `zero_division=0`.
- **MCC**: Matthews correlation coefficient, which stays honest on imbalanced classes.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.9023 | 0.9843 | 0.9166 | 0.9130 | 0.9147 | 0.8818 |
| KNN | 0.9166 | 0.9860 | 0.9318 | 0.9271 | 0.9292 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | 0.9243 | 0.9946 | 0.9372 | 0.9350 | 0.9361 | 0.9085 |

I retrained all five models from the same code and got exactly the same numbers as `model/metrics.csv`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very close second: F1 0.9335 against 0.9361 for the best model, and the best AUC (0.9948, with SVM at 0.9946). It trains in 0.29 s. That a plain linear model gets this far suggests the classes are fairly well separated after scaling (I did not test this further). Weakest class is SIRA (recall 0.8805). |
| Decision Tree | F1 0.9147 and the lowest AUC (0.9843). Its SIRA recall of 0.8273 is the lowest SIRA value of the five models. Trains in 0.14 s. |
| KNN | Third on F1 (0.9292) and accuracy (0.9166). Training is almost instant (0.01 s) because it only stores the data, so its saved file is the largest, 1.48 MB (it keeps all 10,888 training rows). Weakest class is SIRA (recall 0.8691). |
| Naive Bayes | Lowest accuracy (0.8979) and F1 (0.9091), but its AUC (0.9916) is third best. Its weak spots are BARBUNYA (recall 0.7962, the lowest value in the whole recall table) and DERMASON (0.8604, also the lowest DERMASON value). It is the only model whose weakest class is not SIRA. Training takes 0.01 s. |
| Support Vector Machine | Best on accuracy (0.9243), precision (0.9372), recall (0.9350), F1 (0.9361) and MCC (0.9085). It is also the slowest to train: 2.92 s on my machine. Weakest class is SIRA (recall 0.8634). |

Training times are single runs on my machine.

Patterns across all models (per-class recall table below):

- **BOMBAY** has recall 1.0000 for all five models, even though it is only 3.8% of the data. It is the easiest class to separate.
- **Hardest pair: DERMASON and SIRA.** For every model the largest single confusion is between these two, in one direction or the other (for example 78 DERMASON rows predicted as SIRA by Naive Bayes, 62 SIRA rows predicted as DERMASON by the Decision Tree). The second hardest pair is BARBUNYA predicted as CALI (16 to 40 rows depending on the model).
- SIRA is the weakest class by recall for four of the five models. Naive Bayes is the exception, where BARBUNYA is weakest.

Per-class recall (test set):

| ML Model Name | BARBUNYA | BOMBAY | CALI | DERMASON | HOROZ | SEKER | SIRA |
|---|---|---|---|---|---|---|---|
| Test rows | 265 | 104 | 326 | 709 | 386 | 406 | 527 |
| Logistic Regression | 0.8906 | 1.0000 | 0.9417 | 0.9083 | 0.9508 | 0.9532 | 0.8805 |
| Decision Tree | 0.8604 | 1.0000 | 0.9080 | 0.9140 | 0.9430 | 0.9384 | 0.8273 |
| KNN | 0.8717 | 1.0000 | 0.9417 | 0.9083 | 0.9534 | 0.9458 | 0.8691 |
| Naive Bayes | 0.7962 | 1.0000 | 0.9233 | 0.8604 | 0.9534 | 0.9433 | 0.8880 |
| Support Vector Machine | 0.9094 | 1.0000 | 0.9417 | 0.9238 | 0.9560 | 0.9507 | 0.8634 |

### Overall winner for this dataset

**Support Vector Machine (RBF)** wins on F1 (0.9361), accuracy (0.9243) and MCC (0.9085). Its lead over Logistic Regression is small (0.0026 F1 on 2,723 test rows) and comes from one split, so I would not call the two clearly different. If training speed matters, Logistic Regression is nearly as good and about ten times faster to train in my run.

## Streamlit app features

- Use the bundled `test_data.csv` (ticked by default) or upload your own CSV.
- The CSV needs the 16 feature columns and a `Class` column. Missing columns or unknown class names show an error and stop before any prediction.
- The app loads the saved models and only predicts. It never retrains on the uploaded data.
- Choose one of the 5 models and see the six metrics (AUC shows `n/a` if a class is missing from the file).
- 10-row data preview.
- Confusion matrix with a Counts / Row-normalised toggle.
- Per-class classification report and a note naming the weakest class by recall.
- Download the predictions as a CSV.
- Model comparison: all five models on the same data, a best-by-F1 banner, a table with the best value in each column highlighted, a bar chart and a heatmap.

## How to run

```
pip install -r requirements.txt
python test_metrics.py
python train_models.py
python -m streamlit run app.py
```

- `requirements.txt` pins the exact versions used (Python 3.12.0): streamlit 1.66.1, scikit-learn 1.9.0, numpy 2.5.2, pandas 3.0.5, matplotlib 3.11.1, seaborn 0.13.2, joblib 1.5.3, openpyxl 3.1.5, certifi 2025.6.15.
- `python train_models.py` needs internet to download the dataset. It writes the models to `model/` and the test split to `test_data.csv`. The saved files are already in the repo, so you can skip it and start the app directly.
- Start the app with `streamlit run`, not `python app.py`. Plain `python app.py` has no Streamlit session, so no page is served.

## Project layout

```
app.py             Streamlit app
train_models.py    split, 5 pipelines, training, saving
metrics_utils.py   compute_metrics() for the six metrics
test_metrics.py    sanity checks for compute_metrics()
explore_data.py    loads the dataset from UCI and prints a summary
requirements.txt   packages
test_data.csv      2,723-row hold-out test set (765 KB)
model/             5 pipelines (.joblib), label_encoder.joblib, metrics.csv
```

Saved model sizes: Logistic Regression 3.2 KB, Decision Tree 34.9 KB, KNN 1.48 MB, Naive Bayes 4.0 KB, SVM 443 KB.

## Reproducibility check

I installed the pinned `requirements.txt` into a fresh virtual environment, ran `python test_metrics.py` (all tests passed) and `python train_models.py`, and started the app. The retrained metrics were identical to the comparison table above, the regenerated `test_data.csv` was byte-identical to the one in the repo, and the app started with no exceptions.

## How I built it

This project was built stage by stage:

1. Project scaffold, requirements and README placeholder.
2. Data exploration script (shape, dtypes, missing values, duplicates, class balance).
3. Multi-class metrics helper (`compute_metrics`) with sanity tests.
4. Training script: stratified split and five scaler + classifier pipelines.
5. Trained and saved the models, label encoder, `metrics.csv` and the hold-out `test_data.csv`.
6. Streamlit app, first version: CSV upload, input validation, model selector, six metrics.
7. Confusion matrix, classification report and predictions download.
8. Model comparison: table, bar chart and heatmap.
9. Full README with results, per-class recall and observations.
10. Final checks: pinned versions, clean-environment rerun and this build log.

## Known limitations

- **One split only.** No cross-validation, so small gaps between models (such as SVM against Logistic Regression) could change with another split.
- **No tuning.** The hyper-parameters are fixed choices, not searched.
- **Duplicates kept.** The 68 duplicate rows were not removed, so a duplicate could appear in both train and test.
- **Macro metrics on partial files.** If you upload a file with only some of the classes, AUC is `n/a`. Macro F1, precision and recall also drop, because classes that a model predicts but that are absent from the file count as zero. For example KNN on a file with only SIRA, SEKER and BOMBAY rows has accuracy 0.9122 but macro F1 0.4807. Compare models on a file that has all 7 classes.
- **SVC warning.** scikit-learn 1.9.0 prints a `FutureWarning` that `SVC(probability=True)` is deprecated and will be removed in 1.11. It still works in 1.9.0, but training would fail in 1.11 or later.
- **Saved models depend on versions.** The `.joblib` files were made with scikit-learn 1.9.0 and may not load on very different versions.
- **Internet needed for training.** `train_models.py` downloads the data from UCI each run.


