# Breast Cancer Diagnosis (WDBC) — End-to-End ML Pipeline

A portfolio-ready machine learning project that builds an end-to-end classification pipeline using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset to predict **malignant vs. benign** tumors.

> **Focus:** reproducible experimentation, data quality checks, feature selection, model evaluation, and interpretability.

---

## 🔗 Quick Links
- **Live Page (GitHub Pages):** _add after publishing_
- **Notebook:** `/<NOTEBOOK_FILE>.ipynb`
- **Slides:** (add your link)
- **Video:** (add your link)

---

## ✨ Highlights
- **Dataset:** 569 samples, 30 numeric features (+ target)
- **Data Quality:** **no missing values** and **no duplicates**
- **Feature Selection:** removed highly correlated predictors (**> 0.95**), reducing from **30 → 23**
- **Models:** Logistic Regression, Random Forest
- **Best Result (Logistic Regression):**
  - **Accuracy:** 96.5%
  - **Precision:** 97.5%
  - **Recall (malignant):** 92.9%

---

## 🧠 Problem & Context
Breast cancer detection benefits from early diagnosis. Using features extracted from digitized images of fine needle aspirate (FNA) samples, this project trains supervised models to distinguish **malignant** from **benign** tumors.

---

## 🧰 Tech Stack
- **Python**, pandas, numpy
- **scikit-learn** (pipelines, preprocessing, modeling, metrics)
- matplotlib, seaborn
- ucimlrepo (dataset fetch)

---

## 🧪 Methodology (Pipeline)
1. **Data Ingestion**
   - Loaded WDBC directly from the UCI repository using `ucimlrepo`.
2. **EDA + Data Quality**
   - Verified shapes, types, descriptive stats, class balance.
   - Confirmed **no missing values** and **no duplicates**.
3. **Feature Engineering**
   - Renamed raw attributes into descriptive names: `mean / se / worst`.
4. **Feature Selection**
   - Removed predictors with high pairwise correlation (**> 0.95**), reducing dimensionality (**30 → 23**).
5. **Model Training**
   - Trained **Logistic Regression** and **Random Forest** using a standardized pipeline:
     - `StandardScaler` + classifier
   - Used stratified split (80/20) to preserve class distribution.
6. **Evaluation**
   - Compared models with accuracy, precision, recall, F1-score.
   - Interpreted results using:
     - Random Forest feature importances
     - Logistic Regression coefficients

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|---------|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 |
| Random Forest | 0.9474 | 0.9737 | 0.8810 | 0.9250 |

**Why recall matters here:** In medical screening, **missing malignant cases (false negatives)** can be particularly costly, so recall is a critical metric.

---

## 🔍 Interpretability (What drives the prediction?)
Top drivers identified via feature importance / coefficients include:
- concave points (mean/worst)
- radius-related features
- concavity-related features

These features align with the morphological differences between malignant and benign cell nuclei.

---

## ▶️ How to Run Locally

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate
