import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

def find_best_k_and_evaluate(df: pd.DataFrame, k_values=range(5, 11)):
    print("▶ Preprocessing data...")


    # 1. Handle missing values and outliers.
    df = df.copy()
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]


    # 2. One-hot Encoding
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print("One-hot Encoding of Categorical variables..")


    # 3. Divide Feature/Target
    X = df_encoded.drop(columns=['stroke', 'id'])
    y = df_encoded['stroke']
    print("Complete Feature (X) and Target (y) separation..")

    # 4. Feature Importance
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
    plt.savefig('feature_importances.png')
    plt.close()
    print("Feature Importance Visualization ('feature_importances.png') complete.\n")

    # 5. Perform Cross-Validation for each K value (Example Model: ExtraTreesClassifier)
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

            # Training Model
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train_res, y_train_res)

            # Prediction and Evaluation
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

    # 6. Perform Cross-Validation with the optimal K again (including result report output)
    print(f"\n▶ Final evaluation in progress with optimal K={best_k}...")
    skf = StratifiedKFold(n_splits=best_k, shuffle=True, random_state=42)
    fold = 1
    final_auc_scores = []

    for train_idx, test_idx in skf.split(X, y):
        print(f"\nFold {fold}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_prob)
        final_auc_scores.append(auc_score)

        print(f"ROC AUC: {auc_score:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['0', '1']))

        fold += 1

    print(f"\n ROC AUC (K={best_k}): {np.mean(final_auc_scores):.4f}")
    return best_k, all_scores

