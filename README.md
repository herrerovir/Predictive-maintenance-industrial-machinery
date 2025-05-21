# 🛠️⚙️ Predictive Maintenance for Industrial Machinery
 
This repository contains a machine learning project focused on predicting the failure of an industrial milling machine, aiming to optimize maintenance schedules and minimize downtime.

## 📚 Table of Contents

- [Introduction](#introduction)
- [Goal](#goal)
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [How to Run the Project](#how-to-run-the-project)
- [Repository Structure](#repository-structure)
- [Technical Skills](#technical-skills)
- [Dataset](#dataset)
- [Data Exploration](#data-exploration)
- [Data Sampling](#data-sampling)
- [Algorithms](#algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Modeling and evaluation](#modeling-and-evaluation)
- [Random Forest Model](#random-forest-model)
- [Conclusions](#conclusions)
- [Learning Outcomes](#learning-outcomes)

## 📌 Introduction

Predictive maintenance anticipates equipment failures before they happen by using data and advanced technology to identify potential issues early. Analyzing (sensor) data and machine learning, companies can prevent breakdowns, avoid expensive unplanned downtime, reduce repair expenses, and extend the lifespan of their machinery. Transitioning from reactive to proactive maintenance enhances productivity, ensures smoother operations, and boosts profitability.

## 🎯 Goal

The goal of this project is to develop a multiclass classification machine learning model capable of predicting failures of an industrial milling machine. The model will not only forecast whether a failure will happen but also identify the type of failure. This predictive capability seeks to improve operational efficiency, reduce repair costs and extend equipment life to ultimately ensure smoother production processes and higher overall productivity.

## 👀 Project Overview

This project is organized into the following phases:

- Loading and cleaning a synthetic dataset that simulates the operation of an industrial milling machine
- Exploring and analyzing how process parameters relate to machine failures
- Testing different models to find the most effective one for predicting failures
- Building and training a Random Forest model
- Evaluating the model’s performance to measure its accuracy and reliability

## 🧰 Dependencies

The libraries used:

- `pandas` – Data manipulation  
- `numpy` – Numerical computation  
- `matplotlib` and `seaborn` – Data visualization  
- `scikit-learn` – Machine learning

## 💻 How to Run the Project

1. **Clone the Repository**

   Start by cloning the repository to your local machine using the following command:

   ```shell
   git clone https://github.com/herrerovir/Predictive-maintenance-industrial-machinery
   ```

   Change to the project directory:

   ```shell
   cd Predictive-maintenance-industrial-machinery
   ```

2. **Install Dependencies**

   Install the required dependencies listed in the `requirements.txt`:

   ```shell
   pip install -r requirements.txt
   ```

   This will install all necessary libraries such as pandas, numpy, matplotlib, and seaborn.

3. **Run the Jupyter Notebook**

   After installing the dependencies, you can run the Jupyter notebook to perform the data analysis. To start the notebook, use the following command:

   ```shell
   jupyter notebook notebooks/Predictive-maintenance-industrial-machinery.ipynb
   ```

## 📂 Repository Structure

```
Predictive-maintenance-industrial-machinery/
│
├── data/
│   └── raw/
│       └── predictive-maintenance-dataset-ai4i2020.csv         # Original dataset
│   └── processed/
│       └── Predictive-maintenance-cleaned-dataset.csv          # Clean and processed dataset
│
├── model/
│   └── best-random-forest-model.pkl                            # Trained model
│
├── notebooks/
│   └── Predictive-maintenance-industrial-machinery.ipynb       # Jupyter Notebook with the full analysis
│
├── results/
│   └── figures/
│       └── Random-Forest-Confusion-Matrix.png                  # Visualizations
│       └── Random-Forest-Feature-Importance.png                # Visualizations
│       └── Random-Forest-Precision-Recall-Curve.png            # Visualizations
│       └── Random-Forest-ROC-AUC-Curve.png                     # Visualizations
│ 
│   └── model-results/
│       └── random-forest-results.txt                           # Results from the model as txt file
│
├── requirements.txt                                            # Requirements file
│
└── README.md                                                   # Project overview and documentation
```

## 🧠 Technical Skills Demonstrated

- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Model development and evaluation  
- Regression techniques in `scikit-learn`  
- Visualization and interpretation of model results

## 🗂️ Dataset

The dataset used for this project is part of the following publication:

_S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), pp. 69-74._

It is available in a CSV file uploaded to this repository under the name "predictive-maintenance-dataset-ai4i2020". The dataset consists of: 10000 rows and 14 columns.

## 🔍 Data exploration

**The key insight of the data exploration are the following:**

- The data is imbalanced, particularly in the following features: **Product_quality, Machine _failure, TWF, HDF, PWF, OSF, and RNF**.
- There are three types of product quality: low, medium, and high.
- Machine failures account for only 3% of the observations in the dataset.
- The incidence of machine failures is notably higher among lower-quality products.
- There are five types of machine failures: TWF, HDF, PWF, OSF, and RNF.
- The most frequent failures are HDF, OSF, and PWF.
- For all types of machine failures, lower-quality products exhibit a higher failure count, with the exception of RNF failures.
- There is a high positive correlation between air temperature and process temperature.
- There is a high negative correlation between torque and rotational speed.

## 🧪 Data sampling

During the data exploration phase it was discovered that the dataset is highly imbalanced. Class imbalance is a major issue in machine learning, as it can skew model training and results. For example, a model could show 97% accuracy without actually predicting any failures. To address thi problems, data augmentation is used to balance the ratio of failure to non-failure observations to 80/20, while also ensuring diverse failure causes. In this project, the data augmentation technique used was SMOTE (Synthethic Minority Oversampling Technique). SMOTE creates new samples by adjusting existing points based on their neighbors, preserving the minority class while expanding the dataset. It's applied only to the training set after splitting the data, ensuring the test set reflects the original distribution and prevents information leakage, thus maintaining evaluation integrity.

## 🤖 Algorithms

Two different algorithms were employed to predict milling machine failures: decision tree and random forest.

- **Decision Tree**

A decision tree model is an intuitive machine learning algorithm that helps you make predictions based on data. Consider it like a flowchart where each question you ask leads you down a different path. As the tree branches, it breaks down the data into smaller parts based on specific characteristics, resulting in a final decision on the leaves. This method makes it easy to see how different factors influence the results, which makes decision trees a popular choice for anyone who wants to understand the reasoning behind the predictions.

- **Random Forest**

A random forest model is a machine learning algorithm that uses multiple decision trees to make predictions. It works by creating a “forest” of trees, each trained on a different subset of data. When making a prediction, the model combines the results from all the trees to reach a final decision, which helps improve accuracy and reduce the risk of overfitting.

## ✅ Evaluation metrics

The evaluation metrics used to assess model performance were as follows:

- **Accuracy:** proportion of cases predicted correctly.
- **F1 score:** the harmonic mean of precision and recall.
- **Precision:** the ratio of true positives to the total predicted positives.
- **Recall:** the ratio of true positives to the total actual positives.
- **Confusion matrix:** a table to display true positives, true negatives, false positives, and false negatives.
- **Classification report:** includes precision, recall, and F1-score for each class.
- **ROC AUC:** the model's ability to distinguish between classes.

## 📋 Modeling and evaluation

The goal is to develop a predictive model capable of forecasting machine failures. As previously mentioned, the algorithms used were decision tree and random forest. Given the highly imbalanced nature of the dataset, two models were created for each algorithm: one using the original training set and another employing an oversampled training set. After evaluation the four models, it was chosen to use the oversampled random forest as the final model to use in this problem. 

## 🌲 Random Forest Model

After sampling using the SMOTE method and applying hyperparameter tuning, the metrics of the random forest model are the following:

| **Metric**                      | **Value**  |
|---------------------------------|------------|
| **Accuracy**                    | 0.97       |
| **F1 Score**                    | 0.98       |
| **Precision**                   | 0.98       |
| **Recall**                      | 0.97       |
| **ROC AUC Score**               | 0.99       |

The **Random Forest** model has high accuracy and F1 scores but struggles with classifying class 1. Despite hyperparameter tuning, the model needs further adjustments to enhance performance across all classes. Nonetheless, it is the best option to solve this problem for several reasons:

- **ROC AUC Score**: The tuned model achieves a higher ROC AUC score of 0.99, indicating better class distinction, which is crucial for effective predictions.

- **Class-Specific Improvements**: Enhancements in precision, recall, and F1 score for classes 1 and 2, which are minority classes, suggest that tuning has helped the model better recognize these less common classes.

- **Overall Metrics**: While accuracy, F1 score, and precision are the same for both models, the tuned version shows slight improvements in other areas, indicating it’s more capable overall.

In conclusion, the tuned model has a good performance and it is better at recognizing important yet infrequent classes, making it a solid choice for real-world applications where understanding all classes is essential.

## 💡 Conclusion

The goal of this project was to develop a predictive model using decision tree and random forest algorithms to predict the failure of an industrial milling machine.

One major challenge faced was the highly unbalanced dataset, which made modeling with these algorithms complicated. To solve this problem, the SMOTE method was employed to oversample the minority classes.

Four models were developed: a decision tree without oversampling, a decision tree with oversampling, a random forest without oversampling, and a random forest with oversampling. The random forest model with oversampled data demonstrated the best performance in distinguishing between classes. However, the accuracy for class 1 remained low. To improve performance, the hyperparameters of the random forest model were tuned, which improved discrimination between classes, as shown by the higher F1, precision, and recall scores for class 1. Despite these improvements, further improvements are still possible.

**Future work**

As previously mentioned, although the current model performs well, it can be further optimized. Future efforts should focus on exploring more advanced algorithms, such as XGBoost, that can provide better class discrimination. In addition, expanding the dataset to include more minority class observations will be essential to improve classification accuracy.

## 🎓 Learning Outcomes

- **Data Preprocessing:** develop skills in cleaning and preparing data, including ordinal encoding and performing data augmentation using SMOTE.

- **Decision Tree Application:** understand the principles of decision trees, model fitting, and interpreting model outcomes in the context of machine failures.

- **Random Forest Application:** gain insights into the functionality and advantages of random forest models.

- **Model Evaluation:** learn to apply evaluation metrics such as accuracy, F1 score, precision, recall, and confusion matrix to assess forecasting performance.

- **Practical Experience with Tools and Libraries:** acquire hands-on experience using libraries like Pandas, NumPy, and Scikit-learn for data manipulation and modeling.

- **Visualization Skills:** enhance abilities in data visualization to effectively present findings and communicate relationships between variables.