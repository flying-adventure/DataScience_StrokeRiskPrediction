import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import importlib.util
import os

# 모델 파일 경로 설정
MODEL_FILES = [
    '/content/drive/MyDrive/데이터/knn.py',
    '/content/drive/MyDrive/데이터/LogisticRegression_ver2.py',
    '/content/drive/MyDrive/데이터/LogisticRegression_ver1.py',
    '/content/drive/MyDrive/데이터/RandomForest_ver2.py',
    '/content/drive/MyDrive/데이터/RandomForest_ver1.py'
]

def load_models_from_files():
    models = {}
    for filepath in MODEL_FILES:
        # 파일 이름에서 모델 이름 추출
        model_name = os.path.basename(filepath).split('.')[0]
        
        # 모듈 동적 로드
        spec = importlib.util.spec_from_file_location(model_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 각 파일별 모델 생성 로직 추출
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
        elif 'RandomForest' in filepath:
            if 'ver2' in filepath:
                models[model_name] = module.RandomForestClassifier(
                    n_estimators=200, max_depth=20, random_state=42
                )
            else:
                models[model_name] = module.RandomForestClassifier(
                    n_estimators=100, max_depth=None, random_state=42
                )
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

# 모델 학습 및 평가
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, roc_auc, report

# 메인 실행 흐름
if __name__ == "__main__":
    # 모델 동적 로드
    models = load_models_from_files()
    
    # 데이터 준비
    df = pd.read_csv('/content/drive/MyDrive/데이터/healthcare-dataset-stroke-data.csv')
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
        accuracy, roc_auc, report = train_and_evaluate(X_train, X_test, y_train, y_test, model)
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': report
        }
    
    # 결과 출력
    for name, res in results.items():
        print(f"\n{name}")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print(f"ROC AUC: {res['roc_auc']:.4f}")
