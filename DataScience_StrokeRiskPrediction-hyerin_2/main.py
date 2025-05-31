import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import importlib.util
import os
import numpy as np

# 모델 파일 경로 설정
MODEL_FILES = [
    r"C:\Users\heejin\Downloads\DataScience_StrokeRiskPrediction-hyerin_2\knn.py",
    r"C:\Users\heejin\Downloads\DataScience_StrokeRiskPrediction-hyerin_2\logisticregression_ver1.py",
    r"C:\Users\heejin\Downloads\DataScience_StrokeRiskPrediction-hyerin_2\logisticregression_ver2.py",
    r"C:\Users\heejin\Downloads\DataScience_StrokeRiskPrediction-hyerin_2\randomforest_ver1.py",
    r"C:\Users\heejin\Downloads\DataScience_StrokeRiskPrediction-hyerin_2\randomforest_ver2.py"
]

def load_models_from_files():
    models = {}
    for filepath in MODEL_FILES:
        model_name = os.path.basename(filepath).split('.')[0]
        spec = importlib.util.spec_from_file_location(model_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if 'knn' in filepath:
            models[model_name] = module.KNeighborsClassifier(
                n_neighbors=13, weights='uniform', p=2
            )
        elif 'logisticregression' in filepath:
            if 'ver2' in filepath:
                models[model_name] = module.LogisticRegression(
                    C=0.1, penalty='l1', solver='liblinear', random_state=42
                )
            else:
                models[model_name] = module.LogisticRegression(
                    C=1, penalty='l2', solver='liblinear', random_state=42
                )
        elif 'randomforest' in filepath:
            if 'ver2' in filepath:
                models[model_name] = module.RandomForestClassifier(
                    n_estimators=200, max_depth=20, random_state=42
                )
            else:
                models[model_name] = module.RandomForestClassifier(
                    n_estimators=100, max_depth=None, random_state=42
                )
    return models

def preprocess_data(df):
    df = df.copy()
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    X = df_encoded.drop(columns=['stroke', 'id'])
    y = df_encoded['stroke']
    return X, y

# K-Fold 기반 모델 평가
def cross_validate_model(model, X, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc_scores = []
    auc_scores = []

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"Fold {fold} 평가 중...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train_res)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        acc_scores.append(acc)
        auc_scores.append(auc)

        fold += 1

    return np.mean(acc_scores), np.mean(auc_scores)

# 메인 실행 흐름
if __name__ == "__main__":
    # 모델 로드
    models = load_models_from_files()

    # 데이터 로드 및 전처리
    df = pd.read_csv("C:/Users/heejin/Downloads/DataScience_StrokeRiskPrediction-hyerin_2/healthcare-dataset-stroke-data.csv")
    
    # kfoldcross.py 동적 import 및 함수 호출
    kfoldcross_path = r"C:\Users\heejin\Downloads\DataScience_StrokeRiskPrediction-hyerin_2\kfoldcross.py"
    spec = importlib.util.spec_from_file_location("kfoldcross", kfoldcross_path)
    kfoldcross = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kfoldcross)

    print("▶ 최적 K 찾기 (K-Fold Cross Validation) 중...")
    best_k, k_auc_dict = kfoldcross.find_best_k_and_evaluate(df)
    print(f"최적 K: {best_k}")
    print(f"K별 AUC 값: {k_auc_dict}")

    # 전처리
    X, y = preprocess_data(df)

    # 모델별 교차검증
    results = {}
    for name, model in models.items():
        print(f"\n▶ 모델: {name}")
        acc, auc = cross_validate_model(model, X, y, best_k)
        results[name] = {'accuracy': acc, 'roc_auc': auc}
        print(f"[{name}] 평균 Accuracy: {acc:.4f}, 평균 ROC AUC: {auc:.4f}")

    # 최종 결과 출력
    print("\n⭐ 모델별 최종 평가 결과")
    for name, res in results.items():
        print(f"<{name}>")
        print(f"Accuracy: {res['accuracy']:.4f}, ROC AUC: {res['roc_auc']:.4f}")