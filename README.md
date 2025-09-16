# GTC-Customer-Churn-Prediction
# Problem Statement

Our telecom division faces high customer churn, negatively impacting revenue and growth. The goal is to build a machine learning model that predicts which customers are most likely to churn, enabling proactive retention strategies (e.g., promotions, contract offers, improved support).

# Project Idea & Scope

We develop a churn-prediction model that classifies customers into “likely to churn” vs. “likely to stay,” and deploy it in a lightweight web interface for business users to test new customer profiles.

# Workflow

***Data Preparation*** – Collect and clean churn dataset, handle missing values, and encode categorical features.

***EDA & Feature Engineering*** – Explore churn patterns (tenure, contracts, billing) and create predictive features.

***Model Training & Validation*** – Train classification models (Logistic Regression, Random Forest, Gradient Boosting), handle class imbalance, and evaluate using ROC-AUC, precision, recall, and F1-score.

***Deployment*** – Deploy the best-performing model via a simple frontend for real-time churn predictions.
