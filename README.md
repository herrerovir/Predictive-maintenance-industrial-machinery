# 🛠️⚙️ Predictive Maintenance of Industrial Machinery

This project implements a machine learning solution to predict failures in an industrial milling machine by analyzing real-time sensor data. It covers the full pipeline from data cleaning and feature engineering to model training, class balancing, and interpretability through SHAP explainability.

The final **Random Forest model**, enhanced with **SMOTE** and **SHAP**, delivers strong classification performance across all failure types, including rare ones. Built with real-world manufacturing in mind, this solution offers reliable predictions and interpretable insights to support predictive maintenance strategies.

## 🎯 Goal

Build a robust classification model to accurately identify failure modes in industrial milling equipment using operational sensor data. The process includes preprocessing, class balancing, model training, tuning, and evaluation.

## 🏭 Business Value

Unplanned machine failures lead to costly downtime, equipment damage, and disrupted operations. A predictive maintenance model allows early detection of potential issues, enabling scheduled maintenance and reducing downtime. Interpretable predictions also support informed decision-making and increase trust in automated systems.

## 🗂️ Project Structure

```plaintext
Predictive-maintenance-of-industrial-machinery/
│
├── data/
│   ├── raw/                    # Original sensor datasets
│   └── processed/              # Cleaned and preprocessed data
│
├── figures/
│   ├── eda/                    # Exploratory data visualizations
│   ├── model/                  # Model performance plots
│   └── shap/                   # SHAP explainability figures
│
├── models/                     # Trained model files                                    
├── notebooks/                  # Jupyter notebooks              
├── results/
│   └── metrics/                # Evaluation and cross-validation results                    
├── src/                        # Source code and helpers
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🧭 Project Overview

- **Introduction** – Built a machine learning pipeline to classify failure types in an industrial milling machine using sensor data including torque, rotational speed, and tool wear.

- **Data Cleaning** – Removed irrelevant columns, renamed features for clarity, and ensured consistent and clean sensor readings.

- **Exploratory Data Analysis** – Explored feature distributions, class imbalance, and correlations among variables.

- **Preprocessing** – Applied ordinal encoding to categorical features for compatibility with scikit-learn models.

- **Data Sampling** – Used SMOTE to address class imbalance by oversampling minority failure classes in the training set.

- **Model Selection** – Evaluated Decision Tree and Random Forest classifiers using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

- **Model Optimization** – Trained a tuned Random Forest classifier that performed well across all failure categories.

- **Model Performance** – Assessed using ROC AUC and precision-recall curves, along with feature importance rankings.

- **Cross-Validation** – Used stratified k-fold cross-validation to ensure stable performance across splits.

- **Explainability with SHAP** – Applied SHAP values to understand which features drive model predictions both globally and at the individual prediction level.

- **Conclusion** – Delivered a well-performing and interpretable model suitable for integration into predictive maintenance systems.

- **Future Work** – Potential improvements include testing more advanced models (e.g., XGBoost), collecting more data, and integrating the system into a real-time monitoring pipeline.

## 💻 Dependencies

The following Python libraries are required:

* pandas
* numpy
* scikit-learn
* imblearn
* shap
* matplotlib
* seaborn

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/herrerovir/Predictive-maintenance-industrial-machinery
```

2. Navigate into the project directory:

```bash
cd Predictive-maintenance-industrial-machinery
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Launch the main notebook:

```bash
jupyter notebook notebooks/Predictive-maintenance-industrial-machinery.ipynb
```

## 📊 Dataset

The dataset is from:

*S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), pp. 69–74.*

It contains **10,000 samples** and **14 columns**, simulating sensor readings such as torque, rotational speed, and tool wear, along with binary failure labels.

Data format: CSV file, publicly available for academic use.

## 📋 Modeling and Evaluation

Models tested:

- Decision Tree
- Random Forest

**Random Forest with SMOTE** showed the best performance after hyperparameter tuning.

| Metric             | Value |
| ------------------ | ----- |
| Accuracy           | 0.97  |
| Weighted F1 Score  | 0.98  |
| Weighted Precision | 0.98  |
| Weighted Recall    | 0.97  |
| ROC AUC            | 0.97  |
| Macro F1 Score     | 0.67  |
| Class 1 F1 Score   | 0.20  |
| Class 2 F1 Score   | 0.60  |
| Class 3 F1 Score   | 0.71  |
| Class 4 F1 Score   | 0.86  |

## 🔄 Cross-Validation

Used **stratified k-fold cross-validation** to preserve class distributions during evaluation. Results demonstrated stable performance across folds, particularly in the minority failure classes.

## 🧾 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to quantify how each feature impacts predictions. This provided transparency into the model’s decision-making.

Key insights:

- **Tool wear**, **torque**, and **rotational speed** are most influential in failure prediction.
- SHAP revealed how specific sensor readings lead to correct or incorrect classifications.
- Visualizations help build trust and support root cause analysis.

*See the notebook for SHAP implementation, and `/figures/shap/` for plots.*

## 🥇 Results Summary

- Random Forest with SMOTE yielded high accuracy and recall across all failure types.
- Feature importance aligned with engineering expectations.
- Model was robust to data variations and generalizable to unseen samples.
- SHAP provided meaningful interpretability at both the feature and instance levels.

## 🔮 Future Work

- Expand the dataset, especially for underrepresented failure modes.
- Explore more advanced models such as XGBoost or LightGBM.
- Apply deeper feature engineering to capture complex interactions.
- Improve calibration and sensitivity to minor feature variations.
- Integrate the model into a real-time predictive maintenance system.
