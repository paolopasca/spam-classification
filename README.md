# Supervised Learning for Email Classification
## A Comparative Study of CART, k-NN, Random Forest, and Oblique Decision Trees

**Course:** Introduction to Artificial Intelligence – FEUP  
**Academic Year:** 2025/2026  
**Assignment:** Supervised Learning (Assignment 2)

**Authors:**  
Paolo Pascarelli  
João Filipe  
Diogo Teixeira

---

## 1. Problem Description

**Task:** Binary classification for email spam detection.

**Objective:** Automatically classify emails as **Spam (1)** or **Not Spam (0)** based on word frequencies, character frequencies, and capital letter statistics.

**Goal:** Compare four supervised learning algorithms evaluating trade-offs between predictive accuracy, model interpretability, training time, and model complexity.

---

## 2. Dataset

**Name:** Spambase  
**Source:** UCI Machine Learning Repository / OpenML (ID: 44)  
**Reference:** Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999)

| Property | Value |
|----------|-------|
| Instances | 4,601 |
| Features | 57 (continuous) |
| Classes | 2 (binary) |
| Missing Values | 0 |
| Class Balance | 60.6% non-spam / 39.4% spam |

**Feature Groups:**
- **Word Frequencies (48):** Percentage of words matching specific terms (e.g., "make", "money", "free", "credit")
- **Character Frequencies (6):** Percentage of special characters (`;`, `(`, `[`, `!`, `$`, `#`)
- **Capital Letter Statistics (3):** Average length, longest sequence, total count of capital letters

---

## 3. Installation & Requirements

**Prerequisites:**
- Python 3.9+
- Jupyter Notebook or JupyterLab

**Python packages:**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn openml scipy
```

---

## 4. How to Run

From this folder:
```bash
jupyter notebook spam_classification.ipynb
```

Run all cells with `Kernel → Restart & Run All`. Expected runtime is ~5–10 minutes. First run requires internet to download the Spambase dataset from OpenML (cached after first download).

---

## 5. Notebook Structure

The notebook is organized into four main parts:

**Part 1: Algorithm Implementation**
- Custom ISTA (Iterative Shrinkage-Thresholding Algorithm) optimizer
- Oblique Decision Tree class with hybrid splitting strategy
- Soft thresholding for L1 regularization

**Part 2: Data Understanding & Preprocessing**
- Dataset loading from OpenML
- Exploratory Data Analysis (class balance, feature distributions, correlation analysis)
- Preprocessing pipeline construction (leakage-free)

**Part 3: Model Training & Evaluation**
- Nested cross-validation (5 outer folds × 3 inner folds)
- Hyperparameter tuning via GridSearchCV
- Performance metrics computation

**Part 4: Visualization & Analysis**
- ROC curves comparison
- Confusion matrices
- Learning curves
- Feature importance analysis
- Training time comparison

---

## 6. Preprocessing Pipeline

We use scikit-learn's `Pipeline` and `ColumnTransformer` to ensure **no data leakage**.

**Numeric features:**
- `SimpleImputer(strategy='median')`

**Categorical features (if any):**
- `SimpleImputer(strategy='most_frequent')`
- `OneHotEncoder(handle_unknown='ignore')`

**Scaling:**
- `StandardScaler` is applied in the k-NN pipeline.
- The Oblique Decision Tree performs its **own internal global standardization**.
- CART and Random Forest use unscaled features.

The pipeline is wrapped inside cross-validation so preprocessing is fitted only on training folds.

---

## 7. Algorithms Implemented

### CART (Classification and Regression Trees)
Axis-aligned decision tree using Gini impurity. Hyperparameters tuned: `max_depth`, `min_samples_split`, `min_samples_leaf`, `ccp_alpha`.

### k-Nearest Neighbors (k-NN)
Instance-based lazy learner with Minkowski distance. Hyperparameters tuned: `n_neighbors`, `weights`, `p`.

### Random Forest
Ensemble of bagged decision trees with bootstrap sampling and feature subsampling. Hyperparameters tuned: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`.

### Oblique Decision Tree (Custom Implementation)
Decision tree with linear (oblique) splits of the form **w**ᵀ**x** + b < 0. Uses a hybrid two-phase optimization strategy:
1. **Phase 1 (Selection):** `LogisticRegression` with `lbfgs` solver to rapidly evaluate all 2^(K-1)-1 class bipartitions
2. **Phase 2 (Sparsification):** Custom ISTA solver with L1 regularization to drive uninformative weights to zero

Hyperparameters tuned: `max_depth`, `l1_regularization`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`.

---

## 8. Evaluation Methodology

**Nested Stratified Cross-Validation** — 5 outer folds (performance estimation) × 3 inner folds (hyperparameter tuning). This prevents optimistic bias from hyperparameter selection on test data.

**Grid search scoring:** `balanced_accuracy`.

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Balanced Accuracy | Average per-class recall |
| Precision (macro) | TP / (TP + FP) |
| Recall (macro) | TP / (TP + FN) |
| F1-Score (macro) | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve |
| Training Time | Wall-clock time for model fitting |
| Tree Nodes | Model complexity (for tree-based methods) |

---

## 9. Results Summary

Results are taken from `benchmark_results.csv` generated by the notebook.

### Performance Comparison (5-Fold Nested CV)

| Model | Accuracy | F1 (macro) | Precision | Recall | Time (s) | Nodes |
|-------|----------|------------|-----------|--------|----------|-------|
| CART | 0.9165 ± 0.0113 | 0.9121 | 0.9146 | 0.9101 | 2.21 | 97.8 |
| **Oblique Tree** | **0.9313 ± 0.0065** | **0.9281** | **0.9282** | **0.9281** | 31.80 | **13.4** |
| k-NN | 0.9198 ± 0.0092 | 0.9153 | 0.9197 | 0.9119 | 0.54 | — |
| **Random Forest** | **0.9528 ± 0.0085** | **0.9503** | **0.9528** | **0.9482** | 14.27 | 582.5 |

### Key Findings

1. **Random Forest** achieves the highest accuracy (95.28%) but is the least interpretable.
2. **Oblique Tree** outperforms CART (+1.48% accuracy) with far fewer nodes (13.4 vs 97.8).
3. **k-NN** trains fast but shifts cost to prediction time.
4. **Trade-off:** Oblique Tree training time is higher due to iterative ISTA optimization.

---

## 10. File Structure

```
.
├── spam_classification.ipynb   # Main notebook (implementation + analysis)
├── benchmark_results.csv       # Raw benchmark data (generated by notebook)
├── benchmark_results.svg       # Benchmark visualization (snapshot)
└── README.md
```

---

## References

- Hastie, T., Tibshirani, R. & Friedman, J. — *The Elements of Statistical Learning* (2009)
- Breiman, L. et al. — *Classification and Regression Trees* (1984)
- Murthy, S.K., Kasif, S. & Salzberg, S. — *A System for Induction of Oblique Decision Trees* (1994)
- Quinlan, J.R. — *C4.5: Programs for Machine Learning* (1993)
- Shannon, C.E. — *A Mathematical Theory of Communication* (1948)

---

## License

This project was developed for educational purposes at FEUP, University of Porto.

**Last Updated:** February 2026
