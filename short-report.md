## Objective
Build a binary classification model to predict customer churn and produce SHAP-based interpretability to drive business recommendations.

## Dataset
Synthetic telecom dataset (5,000 rows) with features: age, tenure_months, monthly_charges, contract_type, num_complaints, num_services, payment_method, etc.

## Methodology
- Preprocessing: one-hot encode categorical variables
- Model: GradientBoostingClassifier with grid search
- Interpretability: SHAP Explainer (TreeExplainer) to produce global and local explanations

## Results
  - Accuracy: 0.7408
  - AUC: 0.700124870706081
  - F1: 0.10497237569060773
