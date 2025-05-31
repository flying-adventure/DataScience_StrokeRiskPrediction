import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

def find_best_k_and_evaluate(df: pd.DataFrame, k_values=range(5, 11)):
    print("▶ 데이터 전처리 중...")

    # 1. 결측치 및 이상치 처리
    df = df.copy()
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]

    # 2. One-hot 인코딩
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print("범주형 변수 One-hot 인코딩 완료.")

    # 3. Feature/Target 분리
    X = df_encoded.drop(columns=['stroke', 'id'])
    y = df_encoded['stroke']
    print("Feature (X)와 Target (y) 분리 완료.")

    # 4. Feature Importance (분석 단계 유지)
    model_feat = ExtraTreesClassifier(random_state=42)
    model_feat.fit(X, y)
    feat_importances = pd.Series(model_feat.feature_importances_, index=X.columns)
    print("\n[Feature Importance 분석]")
    print("Top 10 Feature Importances:")
    print(feat_importances.nlargest(10))

    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    print("Feature Importance 시각화 ('feature_importances.png') 완료.\n")

    # 5. K 값별 교차검증 수행 (예시 모델: ExtraTreesClassifier)
    best_k = None
    best_score = 0
    all_scores = {}

    print("▶ K 값별 Stratified K-Fold + SMOTE + ExtraTreesClassifier 수행 중...")

    for k in k_values:
        print(f"\n=== K={k} ===")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_auc_scores = []

        fold = 1
        for train_idx, test_idx in skf.split(X, y):
            print(f"Fold {fold} 진행 중...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            # 모델 학습
            model = ExtraTreesClassifier(random_state=42, n_estimators=100)
            model.fit(X_train_res, y_train_res)

            # 예측 및 평가
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
            fold_auc_scores.append(auc_score)

            print(f"Fold {fold} ROC AUC: {auc_score:.4f}")
            fold += 1

        mean_auc = np.mean(fold_auc_scores)
        all_scores[k] = mean_auc
        print(f"K={k} 평균 ROC AUC: {mean_auc:.4f}")

        if mean_auc > best_score:
            best_score = mean_auc
            best_k = k

    print(f"\n✅ 가장 좋은 K는 {best_k}이며, 평균 ROC AUC는 {best_score:.4f}입니다.")

    # 6. 최적 K로 다시 한 번 교차검증 (결과 리포트 출력 포함)
    print(f"\n▶ 최적 K={best_k} 로 최종 평가 진행 중...")
    skf = StratifiedKFold(n_splits=best_k, shuffle=True, random_state=42)
    fold = 1
    final_auc_scores = []

    for train_idx, test_idx in skf.split(X, y):
        print(f"\nFold {fold}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = ExtraTreesClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_prob)
        final_auc_scores.append(auc_score)

        print(f"ROC AUC: {auc_score:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['0', '1']))

        fold += 1

    print(f"\n최종 평균 ROC AUC (K={best_k}): {np.mean(final_auc_scores):.4f}")
    return best_k, all_scores

