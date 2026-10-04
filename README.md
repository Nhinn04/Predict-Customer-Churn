# 📊 Kaggle Playground Series s6e3 - Customer Churn Prediction

An end-to-end Machine Learning and Deep Learning pipeline for predicting customer churn, constructed for the **Kaggle Playground Series s6e3** competition.

---

## 📌 Project Overview

This repository features a comprehensive data science workflow including Exploratory Data Analysis (EDA), advanced Feature Engineering (N-Gram text features, Target Encoding, Pseudo-Labeling), model exploration, and multi-model ensembling strategies.

* **Best CV Score:** `0.91927`
* **Core Models:** XGBoost (cuDF), LightGBM, CatBoost, TabM, RealMLP, AutoGluon, ResNet (Target Encoding), xLearn FFM.

---

## 🗂️ Repository Structure

```text
Predict-Customer-Churn/
├── data/
│   ├── raw/                  # Raw datasets (train.csv, test.csv, sample_submission.csv)
│   └── processed/            # Engineered & preprocessed datasets
├── notebooks/
│   ├── eda/                  # Data visualization & exploratory analysis
│   ├── modeling/             # Main training pipelines & ensembling scripts
│   └── experiments/          # Model trials (TabM, ResNet, Pseudo-Labeling, etc.)
├── local_scripts/            # Local training scripts, logs, & local model checkpoints
├── kaggle_outputs/           # Out-of-fold predictions & external model weights
├── submissions/              # Submission CSV files
├── docs/                     # Detailed analysis reports & documentation
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation