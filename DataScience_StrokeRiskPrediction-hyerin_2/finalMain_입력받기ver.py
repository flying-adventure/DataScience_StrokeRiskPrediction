import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, average_precision_score, 
    confusion_matrix, classification_report
)
import importlib.util
import os

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
            models[model_name] = module.KNeighborsClassifier(n_neighbors=13, weights='uniform', p=2)
        elif 'LogisticRegression' in filepath:
            if 'ver2' in filepath:
                models[model_name] = module.LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42)
            else:
                models[model_name] = module.LogisticRegression(C=1, penalty='l2', solver='liblinear', random_state=42)
        elif 'RandomForest' in filepath:
            if 'ver2' in filepath:
                models[model_name] = module.RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
            else:
                models[model_name] = module.RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    return models

# 데이터 전처리
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


# 모델 학습 및 평가
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'pr_auc': average_precision_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

# 평가 결과 출력 및 비교 함수
def compare_models(results):
    # 테이블 헤더 출력
    print(f"{'Model':<10} | {'Accuracy':<8} | {'ROC AUC':<8} | {'PR AUC':<8} | {'Precision':<8} | {'Recall':<8} | {'F1':<8}")
    print("-" * 85)
    
    # 각 모델 지표 출력
    for name, res in results.items():
        print(f"{name:<10} | {res['accuracy']:.4f}   | {res['roc_auc']:.4f}   | {res['pr_auc']:.4f}   | "
              f"{res['precision']:.4f}   | {res['recall']:.4f}   | {res['f1']:.4f}")

# 최종 모델 선택 함수 (PR AUC 60%, Recall 40% 가중치)
def select_best_model(results, weights={'pr_auc': 0.6, 'recall': 0.4}):
    scores = {}
    for name, res in results.items():
        score = (res['pr_auc'] * weights['pr_auc'] + 
                 res['recall'] * weights['recall'])
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

# 메인 실행 흐름 수정 부분
if __name__ == "__main__":
    # 모델 동적 로드
    models = load_models_from_files()

    # 데이터 준비
    df = pd.read_csv("C:/Users/heejin/Downloads/DataScience_StrokeRiskPrediction-hyerin_2/healthcare-dataset-stroke-data.csv")
    X, y = preprocess_data(df)

    # SMOTE 적용
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.25, random_state=42, stratify=y_res
    )

    # 모델 평가
    results = {}
    for name, model in models.items():
      results[name] = train_and_evaluate(X_train, X_test, y_train, y_test, model)

    # 결과 출력 
    print("[모델 성능 비교 테이블]")
    compare_models(results)

    # 최고 성능 모델 선택
    best_model_name, best_score = select_best_model(results)
    best_model = models[best_model_name]
    print(f"\n=== 최종 선택 모델: {best_model_name} (종합 점수: {best_score:.4f}) ===")
    print("주요 성능 지표:")
    print(f"- ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"- PR AUC: {results[best_model_name]['pr_auc']:.4f}")
    print(f"- F1 Score: {results[best_model_name]['f1']:.4f}")
    print(f"- Recall: {results[best_model_name]['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(results[best_model_name]['confusion_matrix'])
    
    # 사용자 입력으로 환자 데이터 받기
    print("\n=== 새로운 환자 정보 입력 ===")
    new_patients_data = input_patient_data()
    
    # 예측 데이터 전처리
    X_new = preprocess_new_data(new_patients_data, X.columns)
    X_new_scaled = scaler.transform(X_new)
    
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
