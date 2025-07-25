# Hepatitis B Disease Prediction Using Machine Learning

## Project Overview

This project focuses on developing a robust machine learning model to predict the likelihood of Hepatitis B mortality based on clinical and demographic data. Given the challenges in diagnosing liver diseases, which can often be asymptomatic in early stages, this model provides a data-driven approach to risk assessment. By leveraging a systematic data science workflow, the project evaluates multiple classification algorithms and demonstrates the critical impact of preprocessing techniques—specifically class balancing with SMOTE—on predictive accuracy. The final model is deployed in a simple web application, showcasing an end-to-end machine learning solution.

## Data

The model was trained on the Hepatitis Dataset from the UC Irvine Machine Learning Repository.

- **Source**: UCI Machine Learning Repository: Hepatitis Dataset
- **Instances**: 155
- **Attributes**: 20 (including AGE, SEX, BILIRUBIN, ALK PHOSPHATE, ALBUMIN, HISTOLOGY, etc.)
- **Target Variable**: Class (1 for Die, 2 for Live)

## Methodology

The project followed a structured data science workflow, from data preprocessing to model evaluation, selection, and deployment.

### 1. Data Preprocessing and Feature Engineering

A multi-step preprocessing pipeline was implemented to prepare the raw data for modeling:

**Missing Value Imputation**: The dataset contained missing values represented by `?`. These were first converted to `np.nan`. Subsequently, a dual strategy was used for imputation:
- Numerical features (BILIRUBIN, PROTIME, etc.) had missing values filled with the mean of their respective columns
- Categorical features (STEROID, FATIGUE, etc.) had missing values filled with the most frequent value (mode)

**Data Type Conversion**: All numerical feature columns were explicitly cast to a float data type to ensure consistency for calculations.

**Feature Scaling**: The features were standardized using StandardScaler from Scikit-learn. This process removes the mean and scales each feature to unit variance, a necessary step for distance-based algorithms like KNN and SVM to perform optimally.

### 2. Handling Class Imbalance with SMOTE

The original dataset was highly imbalanced, with the "Live" class constituting 74% of the instances and the "Die" class only 26%. This skewed distribution can lead to a model that is biased towards the majority class.

To mitigate this, the Synthetic Minority Oversampling Technique (SMOTE) was applied exclusively to the training data. SMOTE synthesizes new data points for the minority class by interpolating between existing minority instances. This created a balanced class distribution in the training set, allowing the models to learn the patterns of both classes effectively.

### 3. Model Implementation and Evaluation

Three supervised classification algorithms were implemented and evaluated to predict patient outcomes. The dataset was split into training and testing sets (e.g., 60/40 split) to ensure a fair evaluation.

- **Naïve Bayes**: A probabilistic classifier based on Bayes' theorem with an assumption of feature independence
- **Support Vector Machine (SVM)**: A powerful model that finds an optimal hyperplane to separate the classes in a high-dimensional space
- **K-Nearest Neighbors (KNN)**: A non-parametric, instance-based algorithm that classifies data points based on the majority class of their k-nearest neighbors in the feature space

## Results & Analysis

The models were evaluated based on their predictive accuracy, with a direct comparison between performance on the original imbalanced data and the SMOTE-balanced data. The results clearly highlight the efficacy of addressing class imbalance.

| Algorithm    | Accuracy (Without SMOTE) | Accuracy (With SMOTE) |
|--------------|--------------------------|----------------------|
| Naïve Bayes  | 57%                      | 82%                  |
| SVM          | 85%                      | 92%                  |
| KNN          | 83%                      | 93%                  |

The K-Nearest Neighbors (KNN) model, when trained on the SMOTE-balanced dataset, achieved the highest accuracy of **93%**. This significant improvement underscores the importance of proper data preparation in developing reliable predictive models in the medical field.

## Application Deployment

The final, trained KNN model was serialized using pickle and integrated into a web application using the Flask micro-framework. This provides a simple, user-friendly interface where a user can input the 17 required clinical attributes and receive an instant prediction regarding their Hepatitis B risk status.
