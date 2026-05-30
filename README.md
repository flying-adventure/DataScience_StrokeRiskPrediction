# DataScience - Stroke Risk Prediction

## 📊 프로젝트 소개

**Stroke Risk Prediction**은 머신러닝 기반의 뇌졸중(Stroke) 위험도 예측 모델입니다. Kaggle의 **Healthcare Stroke Dataset**을 활용하여 환자의 건강 정보를 바탕으로 뇌졸중 발생 확률을 예측합니다.

본 프로젝트는 **5가지 머신러닝 알고리즘**을 비교하고, **K-Fold Cross Validation**, **SMOTE 불균형 데이터 처리**, **클러스터링 분석** 등 다양한 데이터 과학 기법을 적용한 종합적인 머신러닝 파이프라인입니다.

---

## 🎯 프로젝트 목표

- 환자의 건강 데이터(나이, BMI, 혈당, 흡연 여부 등)를 바탕으로 뇌졸중 발생 확률 예측
- 여러 머신러닝 알고리즘의 성능 비교 및 평가
- 불균형 데이터셋 처리 방법 검증
- 최적의 모델 선택 및 실무 적용 방안 제시

---

## 🛠 기술 스택

| 분류 | 내용 |
|------|------|
| 언어 | Python 3 |
| 주요 라이브러리 | scikit-learn, pandas, NumPy, matplotlib, seaborn |
| 머신러닝 알고리즘 | KNN, Logistic Regression, Random Forest |
| 데이터 처리 | SMOTE (불균형 처리), StandardScaler (정규화) |
| 검증 방법 | K-Fold Cross Validation, Stratified K-Fold |
| 평가 지표 | Accuracy, ROC-AUC, Classification Report |
| 개발 환경 | Google Colab (권장), Jupyter Notebook |

---

## 📊 데이터셋

**Healthcare Stroke Dataset (Kaggle)**

### 데이터 특성
- **총 샘플**: 약 5,110개 환자 기록
- **특성(Features)**: 11개
- **목표 변수(Target)**: stroke (0: 정상, 1: 뇌졸중)
- **클래스 불균형**: 약 95:5 (정상 vs 뇌졸중)

### 주요 특성
| 특성 | 설명 | 데이터 타입 |
|------|------|-----------|
| gender | 성별 | 범주형 (Male/Female/Other) |
| age | 나이 | 수치형 |
| hypertension | 고혈압 여부 | 이진형 (0/1) |
| heart_disease | 심장질환 여부 | 이진형 (0/1) |
| ever_married | 결혼 여부 | 범주형 (Yes/No) |
| work_type | 직업 유형 | 범주형 |
| Residence_type | 거주 유형 | 범주형 (Urban/Rural) |
| avg_glucose_level | 평균 혈당 | 수치형 |
| bmi | 체질량지수 | 수치형 |
| smoking_status | 흡연 상태 | 범주형 |
| stroke | **뇌졸중 여부** | 이진형 (0/1) - **목표 변수** |

---

## 🤖 머신러닝 모델

### 1️⃣ K-Nearest Neighbors (KNN)

**파일**: `knn.py`

**모델 설정**
```python
KNeighborsClassifier(
    n_neighbors=13,
    weights='uniform',
    p=2  # 유클리디안 거리
)
```
특징

가장 가까운 13개의 이웃 데이터 기반 분류
빠른 학습, 간단한 구현
고차원 데이터에서는 성능 저하 가능
2️⃣ Logistic Regression (Ver 1)
파일: LogisticRegression_ver1.py

모델 설정
```
LogisticRegression(
    C=1,           # 정규화 강도
    penalty='l2',  # L2 정규화
    solver='liblinear',
    random_state=42
)
```
특징

선형 분류 알고리즘
해석이 용이하고 확률 출력 가능
계산 효율성 우수
3️⃣ Logistic Regression (Ver 2)
파일: LogisticRegression_ver2.py

설명: Ver 1과 다른 하이퍼파라미터 또는 전처리 방식으로 성능 비교

4️⃣ Random Forest (Ver 1)
파일: RandomForest_ver1.py

모델 설정
```
RandomForestClassifier(
    n_estimators=100,  # 100개 의사결정트리
    max_depth=None,    # 무제한 깊이
    random_state=42
)
```

특징

앙상블 학습 (100개 트리 조합)
높은 예측 정확도
특성 중요도 분석 가능
5️⃣ Random Forest (Ver 2)
파일: RandomForest_ver2.py

설명: Ver 1과 다른 하이퍼파라미터(깊이, 트리 개수 등)로 성능 비교

📁 디렉토리 구조
```
DataScience_StrokeRiskPrediction/
├── FinalMain.py                          # 메인 파이프라인 (모든 모델 통합)
├── knn.py                                # KNN 모델
├── LogisticRegression_ver1.py            # 로지스틱 회귀 (버전 1)
├── LogisticRegression_ver2.py            # 로지스틱 회귀 (버전 2)
├── RandomForest_ver1.py                  # 랜덤 포레스트 (버전 1)
├── RandomForest_ver2.py                  # 랜덤 포레스트 (버전 2)
├── kfoldcross.py                         # K-Fold Cross Validation 구현
├── healthcare-dataset-stroke-data.csv    # 원본 데이터셋
├── DataScience_D_FinalReport.docx        # 최종 보고서
├── DataScience_TermProject_proposal_D조.pptx  # 프로젝트 제안
├── termproject_groupD.pptx               # 최종 발표 자료
└── README.md
```
---

## 🚀 실행 방법
1. 환경 설정
Google Colab (권장)
```

# Colab 환경에서 실행
from google.colab import files
uploaded = files.upload()  # 파일 업로드
```
로컬 환경
```pip install pandas scikit-learn imbalanced-learn matplotlib seaborn numpy```

2. 데이터셋 준비
```healthcare-dataset-stroke-data.csv을 프로젝트 폴더에 배치```
3. 메인 파이프라인 실행
```python FinalMain.py```
FinalMain.py 실행 순서:

데이터 로드 및 전처리
5개 모델 동적 로드
SMOTE를 활용한 불균형 데이터 처리
각 모델 학습 및 평가
K-Fold Cross Validation 검증
클러스터링 분석 (KMeans + PCA)
성능 비교 및 최종 결과 출력

---

## 📊 데이터 전처리 (FinalMain.py)
1. 결측치 처리
BMI 결측값: 중앙값으로 대체
성별 'Other': 제거
극단값(BMI >= 80): 제거
2. 데이터 필터링
성인만 포함 (age >= 20)
분명한 성별만 유지 (Male/Female)
3. 인코딩
범주형 변수: One-Hot Encoding
gender, ever_married, Residence_type, work_type, smoking_status
4. 정규화
StandardScaler: 모든 특성을 동일 스케일로 정규화
5. 불균형 처리
SMOTE (Synthetic Minority Over-sampling Technique)
소수 클래스(뇌졸중) 데이터 합성 생성
클래스 비율을 균형있게 조정
---

## ✅ 평가 지표
지표	설명
Accuracy	전체 예측 중 정확한 예측의 비율
ROC-AUC	모든 임계값에서의 분류 성능 (0~1, 1에 가까울수록 좋음)
Precision	양성 예측 중 실제 양성의 비율 (위양성 최소화)
Recall	실제 양성 중 정확히 예측한 비율 (위음성 최소화)
F1-Score	Precision과 Recall의 조화 평균

---

## 🔍 클러스터링 분석
KMeans Clustering + PCA Visualization

최적 클러스터 개수 결정 (Silhouette Score 활용)
고차원 데이터를 2D로 축소 (PCA)
환자 그룹의 자연스러운 분류 확인

---

## 📈 교차 검증 (K-Fold Cross Validation)
파일: kfoldcross.py

Stratified K-Fold (클래스 비율 유지)
k=5 (5-Fold CV)
모든 모델에 동일 검증 방식 적용

---

## 🎯 모델 비교 및 선택
성능 평가 기준
Accuracy (정확도)
ROC-AUC (곡선 아래 면적)
Cross-Validation Score (일반화 성능)
Precision/Recall Balance (의료 데이터 특성상 중요)
최종 권장 모델
데이터셋 특성과 성능 평가 결과 기반 선택
프로덕션 환경에서의 실행 속도 고려
모델 해석 가능성 고려

---

## 💡 주요 발견사항
📌 클래스 불균형의 영향

SMOTE 처리 전후 성능 비교
불균형 데이터의 문제점 및 해결 방안
📌 알고리즘별 성능 차이

KNN: 직관적이지만 고차원에서 약함
Logistic Regression: 빠르고 안정적
Random Forest: 높은 정확도, 과적합 위험
📌 특성 중요도

Random Forest의 특성 중요도 분석
뇌졸중 예측에 가장 영향력 있는 요인 파악

---

## 🔧 확장 가능성
🚀 모델 개선

XGBoost, LightGBM 등 고급 알고리즘 추가
하이퍼파라미터 튜닝 (GridSearchCV, RandomSearchCV)
앙상블 기법 (Voting, Stacking)
🚀 데이터 확장

더 큰 규모 데이터셋 활용
실시간 환자 데이터 통합
시계열 데이터 추가 (여러 시점의 측정)
🚀 배포 및 운영

웹 서비스화 (Flask/Django)
REST API 구축
모니터링 및 모델 재학습 시스템

---

## 📝 실행 환경
Google Colab (권장)
```
1. 새로운 코드 셀 생성
2. 파일 경로 확인
3. 아래 코드 실행
   - from google.colab import files
   - uploaded = files.upload()
4. FinalMain.py 실행
```

로컬 Jupyter Notebook

파이썬 3.8+
필요한 라이브러리 설치 후 실행

---

## 📚 참고자료
Dataset: Kaggle Healthcare Stroke Dataset
알고리즘: scikit-learn 공식 문서
불균형 처리: SMOTE 논문 및 imbalanced-learn 라이브러리
교차검증: k-fold cross-validation 기법

---

## 📄 프로젝트 산출물
📊 최종 보고서: DataScience_D_FinalReport.docx
🎯 프로젝트 제안: DataScience_TermProject_proposal_D조.pptx
🎤 최종 발표 자료: termproject_groupD.pptx
Claude Haiku 4.5 • 1x
