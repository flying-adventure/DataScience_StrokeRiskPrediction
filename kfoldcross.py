import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

def show_feature_importance(X, y):
    model_feat = RandomForestClassifier(random_state=42)
    model_feat.fit(X, y)
    feat_importances = pd.Series(model_feat.feature_importances_, index=X.columns)

    print("\n[Feature Importance Analysis]")
    print("Top 10 Feature Importances:")
    print(feat_importances.nlargest(10))

    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

def find_best_k_and_evaluate(X, y, k_values=range(5, 11)):
    show_feature_importance(X, y)

    best_k = None
    best_score = 0
    all_scores = {}

    print("▶ Performing Stratified K-Fold + SMOTE + RandomForestClassifier for each K value...")

    for k in k_values:
        print(f"\n=== K={k} ===")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_auc_scores = []

        fold = 1
        for train_idx, test_idx in skf.split(X, y):
            print(f"Fold {fold} in progress...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train_res, y_train_res)

            y_prob = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
            fold_auc_scores.append(auc_score)

            print(f"Fold {fold} ROC AUC: {auc_score:.4f}")
            fold += 1

        mean_auc = np.mean(fold_auc_scores)
        all_scores[k] = mean_auc
        print(f"K={k} Average ROC AUC: {mean_auc:.4f}")

        if mean_auc > best_score:
            best_score = mean_auc
            best_k = k

    print(f"\n✅ The best K is {best_k}, and the average ROC AUC is {best_score:.4f}")
    return best_k, all_scores
