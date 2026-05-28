# Steam Games Hit Prediction

A binary classification project predicting whether a Steam game will achieve **"hit" status** based on review counts, playtime, categories, genres, and other game metadata.

**Dataset**: 122,611 Steam games · [fronkongames/steam-games-dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) on Kaggle

---

## The Problem

### What makes a game a "hit"?

A game is labelled `is_hit = 1` if it meets **both** criteria:
- At least **1,000 positive reviews**
- At least **80% positive review ratio** (`Positive / (Positive + Negative) ≥ 0.8`)

Only **4,717 out of 122,611 games** (~4%) qualify as hits — this is a severely imbalanced binary classification problem.

### Why is this dataset hard?

| Challenge | Details |
|---|---|
| **Class imbalance** | Only ~4% positive class. Standard accuracy is meaningless — a model predicting "not a hit" for every game would still reach 96% accuracy. |
| **Heavily right-skewed distributions** | Most games have 0 playtime, 0 reviews, 0 recommendations. A small number of popular titles create extreme long tails. Standard histograms and boxplots are dominated by the zero mass. |
| **High missing values** | Several columns have 50–100% missing data: `Movies` (100%), `Score rank` (99.97%), `Metacritic url` (96.5%), `Reviews` (90.2%). |
| **High-cardinality features** | `Tags`, `Developers`, `Publishers` have tens of thousands of unique values, making direct encoding infeasible. |
| **Multi-value string columns** | `Categories`, `Genres`, `Supported languages` store comma-separated lists that require one-hot expansion. |
| **Data leakage risk** | `Positive` and `Negative` are used to construct the target — they must be dropped before any model training. |

---

## Dataset Overview

| Property | Value |
|---|---|
| Total games | 122,611 |
| Original columns | 39 (14 int, 3 float, 19 string, 3 bool) |
| Columns after cleaning | 24 |
| Columns after encoding | 129 |
| Hit games (`is_hit = 1`) | 4,717 (~3.85%) |
| Non-hit games (`is_hit = 0`) | 117,894 (~96.15%) |

Key raw columns:

| Column | Type | Description |
|---|---|---|
| `Positive` / `Negative` | int | Number of positive / negative reviews |
| `Average/Median playtime forever` | int | Playtime in minutes (all-time) |
| `Estimated owners` | int | Steam owner range midpoint |
| `Recommendations` | int | Number of Steam recommendations |
| `Achievements` | int | Number of in-game achievements |
| `Price` | int | Game price in cents |
| `Release date` | string | Release date → converted to days since earliest |
| `Categories` / `Genres` | string | Comma-separated Steam tags |
| `Supported languages` | string | Comma-separated language list |
| `Developers` / `Publishers` | string | Company names |
| `Mac` / `Linux` | bool | Platform availability flags |

---

## Analysis Approach

### Step 1 — EDA & Data Cleaning (`eda_workflow.ipynb`)

1. **Column report** — compute null rates, cardinality, and outlier counts for all 39 columns
2. **Drop uninformative columns** using three criteria:
   - `> 50%` missing values → 8 columns removed
   - `> 99%` single-value dominance → 2 columns removed (`Windows`, `User score`)
   - `> 90%` unique values (too high cardinality) → 5 columns removed (`AppID`, `About the game`, `Tags`, etc.)
3. **Visualise distributions** — standard histograms fail (dominated by zeros), so ECDF and boxenplots with `log(1+x)` transform are used instead
4. **Encode categorical features**:
   - Release date → days since the earliest date in the dataset
   - Owner ranges like `"1000–2000"` → interval midpoint
   - Boolean columns → 0 / 1
   - Supported languages → 11 one-hot columns (top 10 + "other")
   - Categories / Genres → 58 / 33 one-hot columns
   - Developers / Publishers → frequency-encoded value + top-50 binary flag
5. **Statistical analysis** — Spearman correlation and Mann-Whitney U tests to assess feature–target relationships before modelling

> **Note on `Tags`**: this column was explored but excluded. One-hot encoding it would have tripled the dataset size, making training computationally prohibitive within the project's time constraints.

### Step 2 — Feature Engineering (`explore.py`)

Five aggregate features derived from existing columns:

| Feature | Formula | Rationale |
|---|---|---|
| `n_languages` | sum of supported language flags | More languages → broader potential audience |
| `n_categories` | sum of Steam category flags | Richer feature set within Steam |
| `n_genres` | sum of genre flags | Genre breadth |
| `n_tags` | sum of tag flags | User-defined taxonomy depth |
| `engagement_ratio` | `Median playtime / (Avg playtime + 1)` | Values near 1 suggest consistent engagement; values near 0 suggest sporadic play |

### Step 3 — Modelling (`modeling_workflow.ipynb`)

**Target definition**: `is_hit = 1` if `Positive ≥ 1000` AND `Positive / (Positive + Negative) ≥ 0.8`. The `Positive` and `Negative` columns are immediately dropped after to prevent data leakage.

**Train / test split**: 80% train / 20% test, stratified by `is_hit`.

**Cross-validation**: 5-fold `StratifiedKFold` on the training set. `StandardScaler` is fit inside each fold on training data only — no leakage from scaling.

Three model families compared:

| Model | Configuration | Imbalance handling |
|---|---|---|
| **Logistic Regression** | ElasticNet (`l1_ratio=0.5`), `C=1.0`, `max_iter=5000` | `class_weight='balanced'` |
| **LightGBM** | 400 trees, `max_depth=6`, `num_leaves=63` | `is_unbalance=True` |
| **MLP** | 256 → 128 → 64, ReLU, Adam, L2=0.001, 50 epochs | `sample_weight='balanced'` |

**Primary metric**: **PR AUC** (Precision-Recall AUC). With only ~4% positive examples, ROC AUC can be misleading — a model can score high ROC AUC while performing poorly on the minority class. PR AUC directly reflects how well the model identifies the rare positive class.

---

## Results

Cross-validation results (mean ± std across 5 folds, evaluated on the held-out test set):

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | **PR AUC** |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.908 ± 0.003 | 0.283 ± 0.008 | 0.909 ± 0.004 | 0.431 ± 0.009 | 0.9685 ± 0.0017 | 0.5973 ± 0.0103 |
| **LightGBM** | **0.980 ± 0.000** | **0.663 ± 0.003** | **0.954 ± 0.004** | **0.782 ± 0.003** | **0.9941 ± 0.0002** | **0.8795 ± 0.0035** |
| MLP | 0.970 ± 0.006 | 0.588 ± 0.060 | 0.803 ± 0.042 | 0.675 ± 0.032 | 0.9777 ± 0.0022 | 0.7055 ± 0.0204 |

**LightGBM** is the best-performing model on every metric, with a PR AUC of **0.88** and extremely low fold-to-fold variance — indicating a stable, well-generalising model.

### Feature Importance — SHAP & Stability Selection

Per-fold SHAP beeswarm plots show that playtime metrics (`Median playtime forever`, `Average playtime forever`) and the engineered `engagement_ratio` are consistently the most predictive features.

SHAP stability selection (30 bootstrap iterations, 10k samples each, threshold ≥ 70%) identifies a stable core feature set that reliably appears across different data subsets, independent of any particular train/test split.

---

## Project Structure

```
.
├── eda_workflow.ipynb            # EDA, cleaning, encoding, statistical analysis (English)
├── modeling_workflow.ipynb       # Model training, CV, SHAP, stability selection (English)
├── column_report.ipynb           # Original EDA notebook (Polish)
├── classification.ipynb          # Original modelling notebook (Polish)
├── classification.py             # ML pipeline: models, CV, visualisations
├── clean_data.py                 # Raw data cleaning utilities
├── explore.py                    # EDA plots and feature engineering
├── statistic.py                  # Statistical tests (Spearman, Mann-Whitney U)
├── transform_categorical.py      # Categorical feature encoding
├── csv_files/
│   ├── cleaned_steam_games.csv       # Data after cleaning
│   ├── preprocessed_data.csv         # Data after full preprocessing
│   └── wyniki.csv                    # Model cross-validation results
├── wykresy/                          # Generated plots (PNG)
│   ├── NaN_and_outliners/
│   ├── visualization_num/
│   ├── visualization_num_log/
│   ├── visualization_cat/
│   ├── post_transform/
│   ├── statystyka/
│   └── klasyfikacja/
├── column_report.json            # Column statistics (JSON)
└── requirements.txt
```

---

## Setup

### Requirements

Python 3.10+

```bash
pip install -r requirements.txt
```

### Running the notebooks

Execute in order:

1. **`eda_workflow.ipynb`** — downloads the dataset from Kaggle, performs EDA, cleans and encodes all features, saves to `csv_files/preprocessed_data.csv`
2. **`modeling_workflow.ipynb`** — loads preprocessed data, trains and evaluates 3 models, generates SHAP plots and stability selection results

> Requires a Kaggle API key configured for `kagglehub` to download the dataset automatically.
