# 전처리 → SMOTE → 클러스터링 → 회귀 분석

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
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
# 6. KMeans 클러스터링 + PCA 시각화
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# 최적 K 탐색
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

# PCA 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = kmeans_labels

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
plt.title('PCA Clustering Visualization')
plt.show()

# ------------------------------
# 7. 로지스틱 회귀 학습 및 평가
# ------------------------------
X_clustered = pd.DataFrame(X_scaled, columns=X_feat.columns)
X_clustered['cluster_label'] = kmeans_labels

X_train, X_test, y_train, y_test = train_test_split(
    X_clustered, y_res, test_size=0.25, random_state=42, stratify=y_res
)

logreg = LogisticRegression(solver='liblinear', random_state=42)
param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n로지스틱 회귀 성능:")
print("Accuracy:", best_model.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.show()
