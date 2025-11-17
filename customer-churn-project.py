# 1. Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import joblib
import os

# 2. Synthetic dataset generator (telecom-like)
def generate_telecom_churn(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame()
    # Basic demographics
    df['age'] = np.random.randint(18, 80, size=n_samples)
    df['gender'] = np.random.choice([0,1], size=n_samples)  # 0=F,1=M
    # Account features
    df['tenure_months'] = np.random.poisson(24, size=n_samples).clip(0,120)
    df['monthly_charges'] = np.round(np.random.normal(60, 30, size=n_samples).clip(10,200),2)
    df['contract_type'] = np.random.choice([0,1,2], size=n_samples, p=[0.5,0.3,0.2])  # 0=month-to-month,1=one-year,2=two-year
    df['has_internet'] = np.random.choice([0,1], size=n_samples, p=[0.2,0.8])
    # Usage patterns
    df['avg_call_minutes'] = np.round(np.random.gamma(2.0, 100.0, size=n_samples).clip(5,1000),1)
    df['num_complaints'] = np.random.poisson(0.3, size=n_samples)
    df['num_services'] = np.random.choice([1,2,3,4], size=n_samples, p=[0.4,0.3,0.2,0.1])
    # Payment
    df['paperless_billing'] = np.random.choice([0,1], size=n_samples, p=[0.3,0.7])
    df['payment_method'] = np.random.choice([0,1,2,3], size=n_samples)  # categorical
    # Create churn probability via a nonlinear function
    prob = (
        0.15
        + 0.20*(df['contract_type']==0).astype(int)           # month-to-month more churn
        - 0.12*(df['contract_type']==2).astype(int)           # two-year less churn
        + 0.18*(df['monthly_charges']>100).astype(int)
        + 0.10*(df['num_complaints']>0).astype(int)
        - 0.01*df['tenure_months'].clip(0,60)/60
        - 0.05*(df['num_services']>=3).astype(int)
    )
    # small noise and clip
    prob = (prob + np.random.normal(0, 0.05, size=n_samples)).clip(0.01,0.99)
    df['churn'] = (np.random.rand(n_samples) < prob).astype(int)
    return df

# 3. Create dataset and quick EDA
df = generate_telecom_churn(5000)
print("Dataset shape:", df.shape)
print(df.head())

# 4. Preprocessing - one-hot for categorical vars
X = df.drop(columns=['churn'])
y = df['churn']

# One-hot contract_type and payment_method (keep numeric ones)
X = pd.get_dummies(X, columns=['contract_type','payment_method'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# 5. Model training - GradientBoostingClassifier with a small grid search
model = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_)

# 6. Evaluation
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:,1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print("F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Save model
os.makedirs("outputs", exist_ok=True)
joblib.dump(best, "outputs/best_model.joblib")

# 8. SHAP analysis
explainer = shap.Explainer(best, X_train.astype(float).values)   # TreeExplainer under the hood for tree models
shap_values = explainer(X_test, check_additivity=False)

# 9. Plots: global summary and feature importance
plt.figure(figsize=(8,6))
shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP beeswarm (global feature importance)")
plt.tight_layout()
plt.savefig("outputs/shap_beeswarm.png", dpi=150)
plt.close()

# SHAP bar (mean absolute)
plt.figure(figsize=(8,6))
shap.plots.bar(shap_values, show=False)
plt.title("SHAP mean absolute importance")
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150)
plt.close()

# 10. Select three customers (high-risk, low-risk, borderline)
X_test_df = X_test.reset_index(drop=True)
y_test_df = y_test.reset_index(drop=True)
probs = y_proba
# high risk: max prob
high_idx = np.argmax(probs)
low_idx = np.argmin(probs)
# borderline: prob closest to 0.5
borderline_idx = np.argmin(np.abs(probs - 0.5))

selected_indices = [high_idx, low_idx, borderline_idx]
selected = X_test_df.loc[selected_indices]
selected_probs = probs[selected_indices]
print("selected indices & probs:", list(zip(selected_indices, selected_probs)))

# SHAP force plots & textual outputs
for i, idx in enumerate(selected_indices):
    sv = shap_values[idx]
    # save force plot as html (requires shap.plots.force to produce HTML)
    shap.save_html(f"outputs/shap_force_{i}.html", shap.plots.force(sv, matplotlib=False))
    # also save a matplotlib waterall style (textual)
    plt.figure(figsize=(6,2))
    shap.plots._waterfall.waterfall_legacy(sv.base_values, sv.values, max_display=8, show=False)
    plt.title(f"SHAP waterfall (index {idx})")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_waterfall_{i}.png", dpi=150)
    plt.close()

# 11. Create a small textual report (markdown)
report_md = f"""
# Project: Interpretable Machine Learning — SHAP Analysis of Customer Churn Prediction
*Date:* {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 1. Objective
Build a binary classification model to predict customer churn and use SHAP to explain model predictions both globally and for selected customers.

## 2. Dataset
Synthetic telecom-style dataset with {len(df)} records. Features include age, tenure_months, monthly_charges, contract_type, num_complaints, num_services, etc.

## 3. Model
Algorithm: GradientBoostingClassifier (sklearn), tuned with GridSearchCV.
Best params: {grid.best_params_}

## 4. Performance (on test set)
- Accuracy: {accuracy_score(y_test, y_pred):.3f}
- AUC: {roc_auc_score(y_test, y_proba):.3f}
- F1-score: {f1_score(y_test, y_pred):.3f}

## 5. Interpretability (SHAP)
- Global insights: see outputs/shap_beeswarm.png and outputs/shap_bar.png
- Selected customer analyses:
  - High-risk customer: outputs/shap_force_0.html, outputs/shap_waterfall_0.png
  - Low-risk customer: outputs/shap_force_1.html, outputs/shap_waterfall_1.png
  - Borderline customer: outputs/shap_force_2.html, outputs/shap_waterfall_2.png

### Key findings (example)
- Month-to-month contract increases churn risk significantly.
- High monthly charges and recent complaints are strong drivers of churn.
- Longer tenure reduces churn probability.

## 6. Business recommendations
1. Target month-to-month customers with retention offers.
2. Proactively resolve customer complaints to reduce churn.
3. Offer discounts for high monthly-charge customers at risk.

## 7. Files to submit
- churn_shap_project.ipynb (or .py)
- outputs/ folder with saved plots and model
- report.md (this markdown)

"""

with open("outputs/report.md", "w", encoding="utf-8") as f:
    f.write(report_md)

print("All outputs saved in 'outputs/' folder. Files: ", os.listdir("outputs"))
