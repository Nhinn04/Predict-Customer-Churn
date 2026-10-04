# Customer Churn Prediction & Business Insights
> **Kaggle Playground Series S6E3** — End-to-end Data Science workflow for predicting customer attrition and uncovering behavioral drivers.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-success.svg)](https://pandas.pydata.org/)
[![ROC-AUC](https://img.shields.io/badge/Best%20ROC--AUC-0.91927-orange.svg)]()

---

## Authors
* **Trương Ngô Phát**
* **Nguyễn Thị Yến Nhi**

---

## 1. Business Context & Problem Statement
Customer churn is one of the costliest challenges for subscription-based businesses. This project builds an analytical and predictive framework designed to:
* **Early Identification:** Accurately flag high-risk customers before they cancel their subscriptions.
* **Behavioral Driver Analysis:** Unearth core features (e.g., tenure length, pricing tiers, service engagement) driving churn.
* **Data-Driven Retention:** Provide quantitative insights to help Customer Success and Marketing teams deploy targeted retention campaigns.

---

## 2. Key Exploratory Data Analysis (EDA) & Insights
*Derived from exploratory analysis workflows (`notebooks/eda/`):*
1. **Behavioral Patterns:** Discovered a sharp drop-off curve in customer retention during months 1–6, indicating a critical window for customer onboarding interventions.
2. **Data Cleansing & Missing Values:** Systematic treatment of missing behavioral data using robust segmentation and imputation techniques.
3. **Feature Engineering:** Developed custom features (N-gram string sequences and 5-fold CV Target Encoding) to capture complex non-linear patterns without data leakage.

---

## 3. Modeling & Performance
While a Data Analyst's primary focus is insight discovery, this project also benchmarks over 15 predictive models to ensure maximum generalization, achieving a top-tier leaderboard score of **ROC-AUC: 0.91927**:
* **Baseline & GBDTs:** LightGBM, XGBoost (cuDF), CatBoost.
* **Deep Learning & Ensembling:** TabM, RealMLP, AutoGluon, xLearn FFM, and stacking/blending strategies.

---

## 4. Repository Structure
The repository follows software engineering standards to ensure high readability and seamless collaboration:

```text
Predict-Customer-Churn/
├── data/
│   ├── raw/                 # Raw datasets (train.csv, test.csv, sample_submission.csv)
│   └── processed/           # Cleaned & engineered datasets
├── notebooks/
│   ├── eda/                 # 📊 Data visualization & business insight notebooks
│   ├── modeling/            # Core training pipelines & ensembling scripts
│   └── experiments/         # Deep learning & feature trials (TabM, ResNet, etc.)
├── local_scripts/           # Local execution scripts, logs, & model checkpoints
├── kaggle_outputs/          # Out-of-fold predictions & weights
├── submissions/             # Final submission CSV files
├── docs/                    # Detailed reporting & documentation
└── .gitignore               # Git ignore configuration
```
## 5. Getting Started & Reproducibility
Clone repository:
   git clone [https://github.com/your-username/Predict-Customer-Churn.git](https://github.com/your-username/Predict-Customer-Churn.git)
   cd Predict-Customer-Churn
Install dependencies: pip install -r requirements.txt
Run the workflow:
Explore notebooks/eda/ first to review data visualizations and customer behavior reports.
Execute scripts sequentially in notebooks/modeling/ to reproduce model training and validation results.
   
