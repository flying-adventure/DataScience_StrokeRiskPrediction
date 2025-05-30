import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def stroke_prediction_pipeline_ver1(df):
    df = df.copy()

    # 1. 결측치 및 이상치 처리
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]

    # 2. One-hot 인코딩
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 3. Feature/Target 분리
    X = df_encoded.drop(columns=['stroke', 'id'])
    y = df_encoded['stroke']

    # 4. Feature Importance
    model = ExtraTreesClassifier(random_state=42)
    model.fit(X, y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("Top 10 Feature Importances:")
    print(feat_importances.nlargest(10))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

    # 5. SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # 6. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # 7. KMeans 클러스터링
    best_k, best_score = 0, -1
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score

    print(f"\nBest K: {best_k}, Silhouette Score: {best_score:.4f}")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 8. PCA 시각화
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
    plt.title('PCA Clustering Visualization')
    plt.tight_layout()
    plt.show()

    # 9. 클러스터별 통계
    X_clustered = pd.DataFrame(X_scaled, columns=X.columns)
    X_clustered['cluster_label'] = cluster_labels
    df_cluster = X_clustered.copy()
    df_cluster['stroke'] = y_res.reset_index(drop=True)
    print("\n[클러스터별 stroke 비율 통계]")
    print(df_cluster.groupby('cluster_label')['stroke'].value_counts(normalize=True).unstack().fillna(0))

    # 10. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clustered, y_res, test_size=0.25, random_state=42, stratify=y_res
    )

    # 11. Logistic Regression ver1 적용 (C=1, l2)
    logreg = LogisticRegression(C=1, penalty='l2', solver='liblinear', random_state=42)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]

    print("\n[로지스틱 회귀 평가 결과 - C=1, L2 규제]")
    print("Accuracy:", logreg.score(X_test, y_test))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # 12. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve (C=1, L2)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 실행 예시
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
stroke_prediction_pipeline_ver1(df)
