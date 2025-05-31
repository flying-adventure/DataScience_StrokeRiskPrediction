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
    r"/content/drive/MyDrive/데이터/knn.py",
    r"/content/drive/MyDrive/데이터/LogisticRegression_ver1.py",
    r"/content/drive/MyDrive/데이터/LogisticRegression_ver2.py",
    r"/content/drive/MyDrive/데이터/RandomForest_ver1.py",
    r"/content/drive/MyDrive/데이터/RandomForest_ver2.py"
]

# 불러온 모델들을 분류하는 함수
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
        elif 'LogisticRegression' in filepath:
            if 'ver2' in filepath:
                models[model_name] = module.LogisticRegression(
                    C=0.1, penalty='l1', solver='liblinear', random_state=42
                )
            else:
                models[model_name] = module.LogisticRegression(
                    C=1, penalty='l2', solver='liblinear', random_state=42
                )
        elif 'RandomFores' in filepath:
            if 'ver2' in filepath:
                models[model_name] = module.RandomForestClassifier(
                    n_estimators=200, max_depth=20, random_state=42
                )
            else:
                models[model_name] = module.RandomForestClassifier(
                    n_estimators=100, max_depth=None, random_state=42
                )
    return models

# 데이터 전처리 함수
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

# K-Fold를 기반으로 모델들을 평가하는 함수 (kfoldcross.py 와 관련없이 따로 설정한 모델을 평가. k는 kfoldcross.py 에서 찾은거임)
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
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train) # smote는 train set에만 적용

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

# 새로운 환자 예측용 전처리 함수
def preprocess_new_data(new_data_list, X_columns):
    df = pd.DataFrame(new_data_list)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    # 누락된 컬럼 보정
    for col in X_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[X_columns]
    return df_encoded

# 최종 모델 선택 함수 (ROC AUC 60%, Accuracy 40% 가중치)
def select_best_model(results, weights={'roc_auc': 0.6, 'accuracy': 0.4}):
    scores = {}
    for name, res in results.items():
        score = (res['roc_auc'] * weights['roc_auc'] +
                 res['accuracy'] * weights['accuracy'])
        scores[name] = score
    return max(scores.items(), key=lambda x: x[1])

# 환자 정보 입력 함수
def input_patient_data():
    patients = []
    num_patients = int(input("예측할 환자 수를 입력하세요: "))

    for i in range(num_patients):
        print(f"\n환자 {i+1} 정보 입력:")
        patient = {
            'gender': input("성별 (Male/Female): ").strip(),
            'age': float(input("나이: ")),
            'hypertension': int(input("고혈압 여부 (0: 없음, 1: 있음): ")),
            'heart_disease': int(input("심장병 여부 (0: 없음, 1: 있음): ")),
            'avg_glucose_level': float(input("평균 혈당 수치: ")),
            'bmi': float(input("BMI 지수: ")),
            'work_type': input("직업 유형 (Private/Self-employed/Govt_job/children/Never_worked): ").strip(),
            'smoking_status': input("흡연 상태 (never smoked/formerly smoked/smokes/Unknown): ").strip(),
            'Residence_type': input("거주지 (Urban/Rural): ").strip(),
            'ever_married': input("결혼 여부 (Yes/No): ").strip()
        }
        patients.append(patient)
    return patients



# 메인 실행 흐름
if __name__ == "__main__":

    # 모델 로드
    models = load_models_from_files()

    # 데이터 로드
    df = pd.read_csv("/content/drive/MyDrive/데이터/healthcare-dataset-stroke-data.csv")

    # kfoldcross.py 동적 import 및 함수 호출 (여기선 최적 k만 찾음)
    kfoldcross_path = r"/content/drive/MyDrive/데이터/kfoldcross_pr.py"
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

    # 최고 성능 모델 선택
    best_model_name, best_score = select_best_model(results)
    best_model = models[best_model_name]
    print(f"\n=== 최종 선택 모델: {best_model_name} (종합 점수: {best_score:.4f}) ===")

    # 전체 데이터 재학습 및 스케일러 정의
    print("\n=== 최종 모델 재학습 ===")
    X, y = preprocess_data(df)
    
    # SMOTE 적용 (전체 데이터)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # 스케일러 재정의 및 학습
    global_scaler = StandardScaler()
    X_scaled = global_scaler.fit_transform(X_res)
    
    # 최종 모델 재학습
    best_model.fit(X_scaled, y_res)
  
    # 사용자 입력으로 환자 데이터 받기
    print("\n=== 새로운 환자 정보 입력 ===")
    new_patients_data = input_patient_data()

    # 예측 데이터 전처리
    X_new = preprocess_new_data(new_patients_data, X.columns)
    X_new_scaled = global_scaler.transform(X_new)

    # 예측 및 결과 출력
    probs = best_model.predict_proba(X_new_scaled)[:, 1]
    for i, prob in enumerate(probs):
        print(f"\n--- 환자 {i+1} ---")
        print(f"입력 데이터: {new_patients_data[i]}")
        print(f"예측된 뇌졸중 확률: {prob:.4f}")
        if prob >= 0.5:
            print("뇌졸중 발생 가능성이 높다고 예측합니다.")
        else:
            print("뇌졸중 발생 가능성이 낮다고 예측합니다.")
