# Data Challenges: Classification and Regression

Two supervised learning projects on public tabular data, built with the same pipeline: leakage-safe preprocessing, classical and boosted baselines, a PyTorch MLP, and pretrained tabular foundation models used through in-context learning.

| Project | Task | Metric | Target |
|---|---|---|---|
| [Mortgage Applications in the United States](#1_MAUS) | Binary classification | AUC ROC | **accepted** |
| [Powerlifting Performances](#2_PP) | Regression | RMSE | **TotalKg** |

## Repository Structure

```
.
├── Classification.ipynb 
├── class_utils.py
├── Regression.ipynb
├── regress_utils.py
└── README.md
```

Each notebook imports its helper module (**class_utils** or **regress_utils**).

## Data Setup

The datasets are not included in this repository. Place the files anywhere and set their locations in the imports cell (section 1) of each notebook:

```python
TRAIN_PATH, TEST_PATH = "data_clas/train.csv", "data_clas/test.csv"   # classification
TRAIN_PATH, TEST_PATH = "data_reg/train.csv", "data_reg/test.csv"     # regression
```

Expected files:

| File | Required columns |
|---|---|
| **train.csv** | *id*, the explanatory variables, and the target |
| **test.csv** | *id* and the explanatory variables |

## Environment

```bash
pip install numpy pandas scikit-learn torch tqdm lightgbm xgboost tabpfn tabicl structured-data-models
```
* **GPU:** *DEVICE* is selected automatically (*cuda* if available, otherwise *cpu*). A GPU is strongly recommended for the MLP and the foundation models.

* **Notebook flags:** *MODEL_SIZE* sets the foundation-model size where available (*"small"* by default). With *RAISE_ERRORS = False*, a model that fails (for example, a missing package or token) is skipped and the rest of the notebook still runs.

## General Methodology

Both notebooks follow the same workflow:

1. **Split.** 80/20 train/validation split with a fixed seed (*SEED = 3407*). The classification split is stratified by target; the regression split is random.
2. **Preprocessing.** Cleaning, feature engineering, imputation, scaling and encoding are fitted on the training rows only and then applied to validation and test data. Two feature views are produced:
   * **scl:** median-imputed, standardized numerics and one-hot categoricals, with categories below 0.5 % frequency grouped. Used by linear models, trees, boosting and the MLP.
   * **ord:** numerics with missing values preserved plus ordinal-coded categoricals. Used by the foundation models.
3. **Models.** Every model is fitted on the training set and scored on the validation set.
4. **Selection.** The best validation model is refit on training plus validation data and used to predict the test set.

### Models

| Model | Description |
|---|---|
| Baseline | Training class prior (classification) or training mean (regression). |
| Logistic / Ridge Regression | Linear model with L2 regularization. |
| Random Forest | Bagged decision trees on bootstrap samples; predictions averaged. |
| Extra Trees | Random split thresholds, no bootstrap. |
| Histogram Gradient Boosting | scikit-learn histogram boosting, averaged over 3 seeds, with early stopping on an internal 10 % split. |
| XGBoost | Leaf-wise histogram boosting with early stopping on an internal 10 % split. |
| LightGBM | Leaf-wise histogram boosting with early stopping on an internal 10 % split. |
| MLP (PyTorch) | Linear-BatchNorm-Sigmoid-Dropout blocks trained with Adam and early stopping; best epoch restored. |
| TabPFN-3.5 | Transformer pretrained on synthetic tabular data; in-context learning, no training on the target dataset. |
| TabICLv2 | Tabular in-context learning transformer pretrained on synthetic data; ensemble of 4 estimators. |
| Kumo-Tabular | NVIDIA tabular foundation model with in-context learning. |

Foundation models receive labeled training rows as context (`max_context` caps the context size) and score queries in batches, so no gradient updates are performed.

---

## 1. Mortgage Applications in the United States

**Notebook:** `Classification.ipynb` · **Helpers:** `class_utils.py`

This project addresses a binary classification challenge based on public HMDA mortgage application data released by the CFPB/FFIEC. Each record describes a U.S. home-purchase loan application, combining loan terms, property and neighborhood characteristics, financial information, and self-reported demographic attributes.

The goal is to estimate whether an application is accepted (`accepted = 1`) or denied (`accepted = 0`) from 56 explanatory variables. The training data contains roughly 105,000 labeled rows and the test set about 45,000.

Because most applications are accepted, the target is imbalanced. Performance is measured with **AUC ROC**, which evaluates the relative ranking of predicted probabilities and does not require a decision threshold.

### Pipeline

- **Cleaning:** clipping of loan amount and property value at the training 99.9th percentile, bounds on loan-to-value ratio, income and loan term, numeric conversion of debt-to-income ranges, and grouping of unknown age values.
- **Engineered features:** calculated LTV and LTV gap, loan-to-income, income relative to the MSA median, monthly payment proxy, payment-to-income, tract housing ratios, and a missing-value count.
- **Variables:** 34 selected variables (`SELECTED_VARS` in `class_utils.py`).
- **Evaluation:** AUC, PR-AUC for the rejected class, MCC, macro F1, balanced accuracy and accuracy. Thresholded metrics use each model's MCC-optimal validation threshold.
- **Selection:** highest validation AUC, excluding the baseline.

### Output

**class_pred.csv** with columns *id* and *accepted*, where *accepted* is the predicted acceptance probability.

---

## 2. Powerlifting Performances

**Notebook:** `Regression.ipynb` · **Helpers:** `regress_utils.py`

This project addresses a regression challenge based on public OpenPowerlifting data, the largest collection of powerlifting competition results, updated daily and in the public domain. Each record describes an athlete's performance at a full meet along with athlete, federation, and competition characteristics.

The goal is to predict the achieved total **(TotalKg)** from 16 explanatory variables: three numerical and thirteen categorical. The training set contains roughly 105,000 labeled rows and the test set about 45,000.

Performance is measured with **RMSE**, so predictions are evaluated by how closely they match the achieved total in kilograms.

### Pipeline

- **Engineered features:** sex indicator, numeric weight class and bodyweight-minus-class gap, birth year, junior/master/tested flags, country and state indicators, log transforms, bodyweight-sex interaction, and category frequency counts learned on the training rows.
- **Evaluation:** RMSE, MAE, median absolute error, R² and MAPE.
- **Selection:** lowest validation RMSE.

### Output

**regress_pred.csv** with columns *id* and *TotalKg*.

---

## Helper Modules

Both modules share the same structure.

| Component | Purpose |
|---|---|
| **Cleaner**| Outlier handling and engineered features; statistics learned on training data only. |
| **Preprocessor** | Builds the **scl** and **ord** views (**transform_views**) and exposes **ord_cat_idx**, the categorical column positions in **ord**. |
| **SeedBag** | Seed-averaged histogram gradient boosting. |
| **EarlyStopped** | Early-stopping wrapper for XGBoost and LightGBM. |
| **TorchMLP** | PyTorch MLP with early stopping. |
| **InContextModel** | Base class for foundation models, with **TabPFNModel**, **TabICLModel** and **KumoTabularModel**. |
| **fit_predict**, **run_model**, **refit_predict** | Training and prediction helpers. |
| **report**, **results_table** | Validation metrics and ranking table. |