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

def stroke_prediction_pipeline_v2(df):
    df = df.copy()

    # 1. Handle missing values and outliers
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]

    # 2. One-hot Encoding
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 3. Divide Feature/Target
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

    # 7.  K-Means Clustering
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

    # 8. PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
    plt.title('PCA Clustering Visualization')
    plt.tight_layout()
    plt.show()

    # 9. Statistics by Cluster
    X_clustered = pd.DataFrame(X_scaled, columns=X.columns)
    X_clustered['cluster_label'] = cluster_labels
    df_cluster = X_clustered.copy()
    df_cluster['stroke'] = y_res.reset_index(drop=True)
    print("\n[Stroke ratio statistics by cluster]")
    print(df_cluster.groupby('cluster_label')['stroke'].value_counts(normalize=True).unstack().fillna(0))

    # 10. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clustered, y_res, test_size=0.25, random_state=42, stratify=y_res
    )

    # 11. Apply Logistic Regression ver2 (C=0.1, l1)
    logreg = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]

    print("\n[Logistic Regression Evaluation Results - C=0.1, L1 Regularization]")
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
    plt.title('ROC Curve (C=0.1, L1)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example of Execution
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
stroke_prediction_pipeline_v2(df)
