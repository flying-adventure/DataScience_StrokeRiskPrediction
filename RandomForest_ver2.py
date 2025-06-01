# Pre-processing → SMOTE → Clustering → Regression Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ------------------------------
# 1. Data Loading and Initial Exploration
# ------------------------------
df = pd.read_csv("/content/drive/MyDrive/데이터/healthcare-dataset-stroke-data.csv")
print("Original Data set:")
print(df.head())
print("\nBasic Info:")
print(df.info())
print("\nBasis Statistics:")
print(df.describe())
print("\nCheck Missing Values:")
print(df.isnull().sum())

# ------------------------------
# 2. Preprocessing (Handling Missing Values and Outliers)
# ------------------------------
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df = df[df['gender'] != 'Other']
df = df[df['age'] >= 20]
df = df[df['bmi'] < 80]

print("\nDataset After Pre-processing:")
print(df.describe())

# ------------------------------
# 3.  One-Hot Encoding (only Categorical)
# ------------------------------
print("\nApplying Categorical variables one-hot encoding...")
categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print("\nOne-hot encoding Result Columns:")
print(df_encoded.columns)

# ------------------------------
# 4. Feature Importance Analysis
# ------------------------------
X_feat = df_encoded.drop(columns=['stroke', 'id'])
y_feat = df_encoded['stroke']
model = ExtraTreesClassifier(random_state=42)
model.fit(X_feat, y_feat)
feat_importances = pd.Series(model.feature_importances_, index=X_feat.columns)

print("\nImportant Feature Top 10:")
print(feat_importances.nlargest(10))
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()

# ------------------------------
# 5. SMOTE Oversampling
# ------------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_feat, y_feat)
df_resampled = pd.DataFrame(X_res, columns=X_feat.columns)
df_resampled['stroke'] = y_res
print("\nClass Distribution After SMOTE:")
print(df_resampled['stroke'].value_counts())

# ------------------------------
# 5-1. Split Data
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ------------------------------
# 6. K-Means Clustering + PCA Visualization
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# -----------------------------------------------------------------------------
# 7. Optimal K Search
# -----------------------------------------------------------------------------

best_k, best_score = 0, -1
print("\nKMeans Optimal K Search:")
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=30, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k} Silhouette Score={score:.4f}")
    if score > best_score:
        best_k, best_score = k, score

print(f"\Optional K: {best_k}, Silhouette Score: {best_score:.4f}")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# -----------------------------------------------------------------------------
# 8. PCA Visualization
# -----------------------------------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = kmeans_labels

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
plt.title('PCA Clustering Visualization')
plt.show()

# -----------------------------------------------------------------------------
# 9. Random Forest Hyperparameter Tuning and Cross-Validation
# -----------------------------------------------------------------------------
param_grid_rf = {
    'n_estimators': [200],
    'max_depth': [20],  
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=30),
    param_grid_rf,
    cv=5,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
# -----------------------------------------------------------------------------
# 10. Random Forest Model Evaluation
# -----------------------------------------------------------------------------
y_pred_rf = best_rf_model.predict(X_test)
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]

print("\n[Random Forest Evaluation Results - Depth Limited]")
print("Accuracy:", best_rf_model.score(X_test, y_test))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['0', '1']))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))
# -----------------------------------------------------------------------------
# 11. ROC Curve Visualization
# -----------------------------------------------------------------------------
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest ROC curve (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve (Unlimited Depth)')
plt.legend(loc="lower right")
plt.show()
