# 전처리 → SMOTE → 클러스터링 → 회귀 분석

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
# 1. 데이터 로딩 및 초기 탐색
# ------------------------------
df = pd.read_csv("/content/drive/MyDrive/데이터/healthcare-dataset-stroke-data.csv")
print("원본 데이터셋:")
print(df.head())
print("\n기본 정보:")
print(df.info())
print("\n기초 통계:")
print(df.describe())
print("\n결측치 확인:")
print(df.isnull().sum())

# ------------------------------
# 2. 전처리 (결측치 및 이상치 제거)
# ------------------------------
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df = df[df['gender'] != 'Other']
df = df[df['age'] >= 20]
df = df[df['bmi'] < 80]

print("\n전처리 후 데이터셋:")
print(df.describe())

# ------------------------------
# 3. One-Hot Encoding (범주형만)
# ------------------------------
print("\n범주형 변수 one-hot encoding 적용 중...")
categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print("\nOne-hot encoding 결과 컬럼:")
print(df_encoded.columns)

# ------------------------------
# 4. Feature Importance 분석
# ------------------------------
X_feat = df_encoded.drop(columns=['stroke', 'id'])
y_feat = df_encoded['stroke']
model = ExtraTreesClassifier(random_state=42)
model.fit(X_feat, y_feat)
feat_importances = pd.Series(model.feature_importances_, index=X_feat.columns)

print("\n중요한 Feature 상위 10개:")
print(feat_importances.nlargest(10))
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()

# ------------------------------
# 5. SMOTE 오버샘플링
# ------------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_feat, y_feat)
df_resampled = pd.DataFrame(X_res, columns=X_feat.columns)
df_resampled['stroke'] = y_res
print("\nSMOTE 후 클래스 분포:")
print(df_resampled['stroke'].value_counts())

# ------------------------------
# 5-1. 데이터 분할
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ------------------------------
# 6. KMeans 클러스터링 + PCA 시각화
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# -----------------------------------------------------------------------------
# 7. 최적 K 탐색
# -----------------------------------------------------------------------------

best_k, best_score = 0, -1
print("\nKMeans 최적 K 탐색:")
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=30, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k} Silhouette Score={score:.4f}")
    if score > best_score:
        best_k, best_score = k, score

print(f"\n최적 K: {best_k}, Silhouette Score: {best_score:.4f}")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# -----------------------------------------------------------------------------
# 8. PCA 시각화
# -----------------------------------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = kmeans_labels

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
plt.title('PCA Clustering Visualization')
plt.show()

# -----------------------------------------------------------------------------
# 9. 랜덤포레스트 하이퍼파라미터 튜닝 및 교차검증
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
# 10. 랜덤포레스트 모델 평가
# -----------------------------------------------------------------------------
y_pred_rf = best_rf_model.predict(X_test)
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]

print("\n[Random Forest 평가 결과 - 깊이제한 있음]")
print("Accuracy:", best_rf_model.score(X_test, y_test))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['0', '1']))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))
# -----------------------------------------------------------------------------
# 11. ROC Curve 시각화
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
