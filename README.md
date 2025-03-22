# 🛠️⚙️ Predictive maintenance of an industrial milling machine
 
This repository contains a machine learning project focused on predicting the failure of an industrial milling machine. 

The complete project is available in the Jupyter notebook titled **Predictive-maintenance-of-industrial-machinery.ipynb** in this repository.

## Table of content
 - [Introduction](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Introduction)
 - [Goal](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Goal)
 - [Dependencies](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Dependencies)
 - [Dataset](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Datasest)
 - [Project Overview](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Project-overview)
 - [Data exploration](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Data-exploration)
 - [Data sampling](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Data-sampling)
 - [Algorithms](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Algorithms)
 - [Evaluation metrics](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Evaluation-metrics)
 - [Modeling and evaluation](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Modeling-and-evaluation)
 - [Conclusions](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Conclusions)
 - [Learning outcomes](https://github.com/herrerovir/ML-predictive-maintenance-of-industrial-machinery/blob/main/README.md#Learning-outcomes)

## Introduction

Predictive maintenance anticipates equipment failures before they happen by using data and advanced technology to identify potential issues early. Analyzing (sensor) data and machine learning, companies can prevent breakdowns, avoid expensive unplanned downtime, reduce repair expenses, and extend the lifespan of their machinery. Transitioning from reactive to proactive maintenance enhances productivity, ensures smoother operations, and boosts profitability.

## Goal

The goal of this project is to develop a multiclass classification machine learning model capable of predicting failures of an industrial milling machine. The model will not only forecast whether a failure will happen but also identify the type of failure. This predictive capability seeks to improve operational efficiency, reduce repair costs and extend equipment life to ultimately ensure smoother production processes and higher overall productivity.

## Dependencies

The following tools are required to carry out this project:

* Python 3
* Jupyter Notebooks
* Python libraries: 
    - Numpy
    - Pandas
    - Matplotlib.pyplot
    - Seaborn
    - Scikit-learn

## Dataset

The dataset used for this project is part of the following publication:

_S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), pp. 69-74._

It is available in a CSV file uploaded to this repository under the name "predictive-maintenance-dataset-ai4i2020". The dataset consists of: 10000 rows and 14 columns.

## Project overview

- Data exploration
- Data sampling
- Algorithms
- Evaluation metrics
- Modeling and evaluation
- Insights

## Data exploration

The exploratory data analysis workflow for this project included the following steps: 

- Data cleaning: it included handling of missing values, duplicated values, remove unnecesary columns and enhance readability of the features
- Univariate exploration: it consisted of individual exploration of each feature, leading to the conclusion that this project is facing a highly imbalanced dataset
- Bivariate exploration: it involved exploring relationships between pair of variables to identify dependencies or correlations.

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

## Data sampling

During the data exploration phase it was discovered that the dataset is highly imbalanced. Class imbalance is a major issue in machine learning, as it can skew model training and results. For example, a model could show 97% accuracy without actually predicting any failures. To address thi problems, data augmentation is used to balance the ratio of failure to non-failure observations to 80/20, while also ensuring diverse failure causes. In this project, the data augmentation technique used was SMOTE (Synthethic Minority Oversampling Technique). SMOTE creates new samples by adjusting existing points based on their neighbors, preserving the minority class while expanding the dataset. It's applied only to the training set after splitting the data, ensuring the test set reflects the original distribution and prevents information leakage, thus maintaining evaluation integrity.

## Algorithms

Two different algorithms were employed to predict milling machine failures: decision tree and random forest.

- **Decision Tree**

A decision tree model is an intuitive machine learning algorithm that helps you make predictions based on data. Consider it like a flowchart where each question you ask leads you down a different path. As the tree branches, it breaks down the data into smaller parts based on specific characteristics, resulting in a final decision on the leaves. This method makes it easy to see how different factors influence the results, which makes decision trees a popular choice for anyone who wants to understand the reasoning behind the predictions.

- **Random Forest**

A random forest model is a machine learning algorithm that uses multiple decision trees to make predictions. It works by creating a “forest” of trees, each trained on a different subset of data. When making a prediction, the model combines the results from all the trees to reach a final decision, which helps improve accuracy and reduce the risk of overfitting.

## Evaluation metrics

The evaluation metrics used to assess model performance were as follows:

- **Accuracy:** proportion of cases predicted correctly.
- **F1 score:** the harmonic mean of precision and recall.
- **Precision:** the ratio of true positives to the total predicted positives.
- **Recall:** the ratio of true positives to the total actual positives.
- **Confusion matrix:** a table to display true positives, true negatives, false positives, and false negatives.
- **Classification report:** includes precision, recall, and F1-score for each class.
- **ROC AUC:** the model's ability to distinguish between classes.

## Modeling and evaluation

The goal is to develop a predictive model capable of forecasting machine failures. As previously mentioned, the algorithms used were decision tree and random forest. Given the highly imbalanced nature of the dataset, two models were created for each algorithm: one using the original training set and another employing an oversampled training set.

- **Decision Tree:**

The evaluation metrics of both decision tree models are shown below:

| **Metric**                     | **Original Decision Tree Model** | **Oversampled Decision Tree Model** |
|--------------------------------|----------------------------------|-------------------------------------|
| **Accuracy**                   | 0.98                             | 0.97                                |
| **F1 Score**                   | 0.98                             | 0.97                                |
| **Precision**                  | 0.98                             | 0.98                                |
| **Recall**                     | 0.98                             | 0.97                                |
| **ROC AUC Score**              | 0.80                             | 0.82                                |

![confusion-matrix-dt-comparison](https://github.com/user-attachments/assets/e6b2ddeb-4a44-4ed4-9d4f-9bcac2e6fd2e)

The comparison between both models reveals that the original decision tree model has a slight edge in accuracy and recall compared to the oversampled model. However, the oversampled model stands out by improving the ROC AUC score from 0.80 to 0.82, which shows it’s better at distinguishing between classes. So, while the original model performs well overall, the oversampled model is more effective at tackling class imbalance thanks to its stronger ROC AUC performance.

- **Random Forest:**

| **Metric**                     | **Original Random Forest Model** | **Oversampled Random Forest Model** |
|--------------------------------|----------------------------------|-------------------------------------|
| **Accuracy**                   | 0.98                             | 0.97                                |
| **F1 Score**                   | 0.98                             | 0.98                                |
| **Precision**                  | 0.98                             | 0.98                                |
| **Recall**                     | 0.98                             | 0.97                                |
| **ROC AUC Score**              | 0.98                             | 0.97                                |

![Confusion-matrix-dt-rf-comparison](https://github.com/user-attachments/assets/3f1f4785-c7dc-420b-b455-4a9696442780)

The comparison between the two random forest models shows that the original model slightly outperforms the oversampled model. Both have the same F1 score and precision, indicating similar performance in balancing false positives and negatives. The original model also has higher recall and a better ROC AUC score. However, the confusion matrix and classification report reveal that the original model fails to predict any observations for class 1, while the oversampled model demonstrates better precision and recall than both the original random forest and the oversampled decision tree discrimination classes. For all these reason the oversampled random forest is the best algorithm for this task. 

- **Final Model: Oversampled and Tuned Random Forest**

After sampling and applying hyperparameter tuning, the metrics of the random forest model are the following:

| **Metric**                      | **Value**  |
|---------------------------------|------------|
| **Accuracy**                    | 0.97       |
| **F1 Score**                    | 0.98       |
| **Precision**                   | 0.98       |
| **Recall**                      | 0.97       |
| **ROC AUC Score**               | 0.99       |

![Confusion-matrix-final-model](https://github.com/user-attachments/assets/a1248004-1cd9-43ee-a89b-484576d799ea)

![Roc-curve-final-model](https://github.com/user-attachments/assets/838d4590-cc09-4c6a-8221-b267cf954e7c)

![Precison-recall-final-model](https://github.com/user-attachments/assets/72bd28cb-4393-4b58-afef-fd4603299123)


The **Best Oversampled Random Forest Model** has high accuracy and F1 scores but struggles with classifying class 1. Despite hyperparameter tuning, the model needs further adjustments to enhance performance across all classes. Nonetheless, it generally outperforms other models for several reasons:

- **ROC AUC Score**: The tuned model achieves a higher ROC AUC score of 0.99, indicating better class distinction, which is crucial for effective predictions.

- **Class-Specific Improvements**: Enhancements in precision, recall, and F1 score for classes 1 and 2, which are minority classes, suggest that tuning has helped the model better recognize these less common classes.

- **Overall Metrics**: While accuracy, F1 score, and precision are the same for both models, the tuned version shows slight improvements in other areas, indicating it’s more capable overall.

In conclusion, the tuned model has a good performance and it is better at recognizing important yet infrequent classes, making it a solid choice for real-world applications where understanding all classes is essential.

## Conclusion

The goal of this project was to develop a predictive model using decision tree and random forest algorithms to predict the failure of an industrial milling machine.

One major challenge faced was the highly unbalanced dataset, which made modeling with these algorithms complicated. To solve this problem, the SMOTE method was employed to oversample the minority classes.

Four models were developed: a decision tree without oversampling, a decision tree with oversampling, a random forest without oversampling, and a random forest with oversampling. The random forest model with oversampled data demonstrated the best performance in distinguishing between classes. However, the accuracy for class 1 remained low. To improve performance, the hyperparameters of the random forest model were tuned, which improved discrimination between classes, as shown by the higher F1, precision, and recall scores for class 1. Despite these improvements, further improvements are still possible.

**Future work**

As previously mentioned, although the current model performs well, it can be further optimized. Future efforts should focus on exploring more advanced algorithms, such as XGBoost, that can provide better class discrimination. In addition, expanding the dataset to include more minority class observations will be essential to improve classification accuracy.

## Learning Outcomes

- **Data Preprocessing:** develop skills in cleaning and preparing data, including ordinal encoding and performing data augmentation using SMOTE.

- **Decision Tree Application:** understand the principles of decision trees, model fitting, and interpreting model outcomes in the context of machine failures.

- **Random Forest Application:** gain insights into the functionality and advantages of random forest models.

- **Model Evaluation:** learn to apply evaluation metrics such as accuracy, F1 score, precision, recall, and confusion matrix to assess forecasting performance.

- **Practical Experience with Tools and Libraries:** acquire hands-on experience using libraries like Pandas, NumPy, and Scikit-learn for data manipulation and modeling.

- **Visualization Skills:** enhance abilities in data visualization to effectively present findings and communicate relationships between variables.
