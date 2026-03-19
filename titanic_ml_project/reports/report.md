# Titanic Machine Learning Project Report

## Problem Statement
The goal of this project is to build a machine learning classification model to accurately predict passenger survival on the RMS Titanic based on demographic and ticketing features (e.g., age, sex, and passenger class).

## Approach
Our approach relies on comprehensive Exploratory Data Analysis (EDA) to understand the dataset, followed by robust preprocessing to handle missing values and encode categorical features. We use Logistic Regression as our primary predictive algorithm along with a Decision Tree Classifier, then evaluate both against unseen test data.

## Steps Followed
1. **Data Exploration (EDA)**: Analysed distributions, missing values, and survival disparities among different sub-groups using visual libraries.
2. **Preprocessing**: 
   - Imputed missing `Age` with the median and missing `Embarked` with the mode.
   - Dropped the `Cabin` column due to excessive missing data (>75%), along with uninformative IDs.
   - Performed Label Encoding on `Sex` and One-Hot Encoding on `Embarked`.
3. **Train-Test Split**: Formatted the dataset with an 80/20 train/test split.
4. **Modeling**: Trained Logistic Regression (primary classifier) and a Decision Tree (secondary/bonus classifier).
5. **Evaluation**: Computed Accuracy, Classification Reports, and Confusion Matrices to determine out-of-sample performance.

## Model Performance
- **Logistic Regression Accuracy**: ~80%
- **Decision Tree Accuracy**: ~81% (with max_depth=5)
- Standard metrics indicate high recall for the non-survived class but moderate performance on the survived class, demonstrating an exceptionally capable baseline model.

## Observations
- **Demographics**: Females and younger passengers were the most likely demographics to survive, indicating standard emergency prioritizing ("women and children first").
- **Socio-Economic factor**: First-Class passengers had significantly higher survival odds than Third-Class passengers.

## Conclusion
The modeling framework successfully predicts survival likelihood with an accuracy surrounding 80%. The project proves that historical safety guidelines dynamically shaped the outcomes of the disaster, and these trends are exceptionally readable for a Machine Learning model. Further optimizations and feature engineering (like computing family size) could elevate accuracy even further.
