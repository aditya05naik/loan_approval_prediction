# 🏦 Loan Approval Prediction — Machine Learning Project

> **Complete end-to-end ML project for internship submission**

---

## 📋 Project Overview

This project builds a supervised machine learning system that predicts whether a loan application will be **approved (Y)** or **rejected (N)** based on applicant information.

| Item | Detail |
|------|--------|
| **Dataset** | [Kaggle — Loan Approval Prediction Case Study](https://www.kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study) |
| **Task** | Binary Classification (Supervised Learning) |
| **Models** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| **Imbalance Fix** | SMOTE Oversampling |
| **Report** | 12-page Professional PDF |

---

## 📁 Project Structure

```
loan_approval_prediction/
│
├── loan_approval_prediction.ipynb    ← Main Jupyter Notebook (all 13 sections)
├── run_project.py                    ← Standalone Python script (no Jupyter needed)
├── requirements.txt                  ← All dependencies
├── setup.bat                         ← Windows auto-installer
├── README.md                         ← This file
│
├── loan_data.csv                     ← Dataset (auto-generated if not provided)
├── model_comparison_results.csv      ← Generated after running
│
├── models/
│   └── loan_approval_pipeline.pkl    ← Saved ML model + scaler
│
└── plots/                            ← All 11 visualizations
    ├── 01_target_distribution.png
    ├── 02_univariate_categorical.png
    ├── 03_univariate_numerical.png
    ├── 04_bivariate_analysis.png
    ├── 05_correlation_heatmap.png
    ├── 06_missing_values.png
    ├── 07_class_imbalance_smote.png
    ├── 08_model_comparison.png
    ├── 09_best_model_evaluation.png
    ├── 10_all_roc_curves.png
    └── 11_feature_importance.png
```

---

## 🚀 Quick Start

### Step 1 — Install Python

Download and install Python 3.9+ from: https://www.python.org/downloads/

Or install [Anaconda](https://www.anaconda.com/) for a full data science environment.

### Step 2 — Install Dependencies

**Option A: Auto-install (Windows)**
```bat
setup.bat
```

**Option B: Manual install**
```bash
pip install -r requirements.txt
pip install xgboost   # optional but recommended
```

### Step 3 — Get the Dataset (Optional)

Download the dataset from Kaggle:  
🔗 https://www.kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study

Place the CSV file in this folder. **If no CSV is found, the project will auto-generate a synthetic dataset** that matches the Kaggle schema.

### Step 4 — Run the Project

**Option A: Jupyter Notebook (Recommended for internship submission)**
```bash
jupyter notebook loan_approval_prediction.ipynb
```
Then run all cells: `Kernel → Restart & Run All`

**Option B: Python Script (Faster)**
```bash
python run_project.py
```

---

## 📊 Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Project Introduction | Problem statement, ML approach |
| 2 | Import Libraries | pandas, numpy, sklearn, imblearn, xgboost |
| 3 | Load Dataset | Load CSV, display shape/info/stats |
| 4 | EDA | Univariate & bivariate analysis, correlation |
| 5 | Preprocessing | Missing values, encoding, scaling |
| 6 | Class Imbalance | SMOTE before/after comparison |
| 7 | Train-Test Split | 80/20 stratified split |
| 8 | ML Models | Train 4+ models, comparison table |
| 9 | Evaluation | Confusion matrix, ROC-AUC, classification report |
| 10 | Feature Importance | Top features graph |
| 11 | Business Interpretation | Insights, risks, recommendations |
| 12 | Deployment | Best model, threshold, integration steps |
| 13 | PDF Report | Generates professional 12-page PDF |

---

## 📈 Expected Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~78% | ~0.78 | ~0.85 |
| Decision Tree | ~80% | ~0.80 | ~0.87 |
| **Random Forest** | **~87%** | **~0.87** | **~0.94** |
| Gradient Boosting | ~85% | ~0.85 | ~0.93 |
| XGBoost | ~88% | ~0.88 | ~0.95 |

> Actual results depend on whether you use the Kaggle dataset or synthetic data.

---

## 📦 Deliverables Generated

After running the project, you will have:

- ✅ `loan_approval_prediction.ipynb` — Complete Jupyter Notebook
- ✅ `models/loan_approval_pipeline.pkl` — Serialized ML pipeline
- ✅ `Loan_Approval_Prediction_Report.pdf` — Professional PDF report
- ✅ `plots/` — 11 high-quality visualizations
- ✅ `model_comparison_results.csv` — All model metrics

---

## 🔧 Load Saved Pipeline

```python
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load('models/loan_approval_pipeline.pkl')
model   = pipeline['model']
scaler  = pipeline['scaler']
columns = pipeline['feature_columns']

# Predict on new data
# new_data = pd.DataFrame({...})
# new_scaled = scaler.transform(new_data[columns])
# prediction = model.predict(new_scaled)
# probability = model.predict_proba(new_scaled)[:, 1]
```

---

## 📄 License

This project is developed for educational and internship submission purposes.

---

*Built with: Python 3.9+ | scikit-learn | pandas | matplotlib | seaborn | imbalanced-learn | XGBoost | fpdf2*
