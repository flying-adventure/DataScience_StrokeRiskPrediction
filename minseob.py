import matplotlib
matplotlib.use('Agg') # GUI 관련 오류 방지를 위해 코드 상단에 추가

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def stroke_prediction_pipeline(df):
    df = df.copy()

    # ------------------------
    # 1. 결측치 및 이상치 처리
    # ------------------------
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]
    print("데이터 전처리 완료: 결측치 및 이상치 처리, 20세 미만 및 BMI 80 이상 데이터 제거.")

    # ------------------------
    # 2. One-hot 인코딩
    # ------------------------
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print("범주형 변수 One-hot 인코딩 완료.")

    # ------------------------
    # 3. Feature/Target 분리
    # ------------------------
    X = df_encoded.drop(columns=['stroke', 'id'])
    y = df_encoded['stroke']
    print("Feature (X)와 Target (y) 분리 완료.")

    # ------------------------
    # 4. Feature Importance
    # ------------------------
    model = ExtraTreesClassifier(random_state=42)
    model.fit(X, y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n[Feature Importance 분석]")
    print("Top 10 Feature Importances:")
    print(feat_importances.nlargest(10))

    # 시각화 및 파일 저장
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig('feature_importances.png') # 파일로 저장
    plt.close() # 그래프 창 닫기
    print("Feature Importance 시각화 ('feature_importances.png') 완료.")

    # ------------------------
    # 5. SMOTE로 클래스 불균형 처리
    # ------------------------
    print("\n[SMOTE 적용]")
    print("SMOTE 적용 전 클래스 분포:\n", y.value_counts())
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("SMOTE 적용 후 클래스 분포:\n", y_res.value_counts())
    print("SMOTE로 클래스 불균형 처리 완료.")

    # ------------------------
    # 6. Scaling
    # ------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    print("데이터 스케일링 완료 (StandardScaler).")

    # ------------------------
    # 7. KMeans 클러스터링
    # ------------------------
    print("\n[KMeans 클러스터링]")
    best_k, best_score = 0, -1
    print("최적 K 탐색 (Silhouette Score 기준):")
    for k in range(2, 11):
        kmeans_eval = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_eval.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"K={k}, Silhouette Score={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score

    print(f"\n최적 K: {best_k}, Best Silhouette Score: {best_score:.4f}")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    print("KMeans 클러스터링 완료.")

    # ------------------------
    # 8. 클러스터 시각화 (PCA)
    # ------------------------
    print("\n[클러스터 시각화 (PCA)]")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
    plt.title('PCA Clustering Visualization')
    plt.tight_layout()
    plt.savefig('pca_clustering_visualization.png') # 파일로 저장
    plt.close() # 그래프 창 닫기
    print("PCA를 이용한 클러스터 시각화 ('pca_clustering_visualization.png') 완료.")

    # ------------------------
    # 9. 클러스터별 통계 출력
    # ------------------------
    X_clustered_features = pd.DataFrame(X_scaled, columns=X.columns)
    X_clustered = X_clustered_features.copy()
    X_clustered['cluster_label'] = cluster_labels
    df_cluster = X_clustered.copy()
    df_cluster['stroke'] = y_res.reset_index(drop=True)
    print("\n[클러스터별 stroke 비율 통계 (스케일링된 데이터 기반)]")
    print(df_cluster.groupby('cluster_label')['stroke'].value_counts(normalize=True).unstack().fillna(0))
    print("클러스터별 통계 출력 완료.")

    # ------------------------
    # 10. Train/Test Split
    # ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_clustered, y_res, test_size=0.25, random_state=42, stratify=y_res
    )
    print("데이터 분리 완료 (Train:", X_train.shape, "Test:", X_test.shape, ").")

    # ------------------------
    # 11. 여러 분류 모델 학습 및 평가
    # ------------------------
    print("\n[여러 분류 모델 학습 및 평가 시작]")

    models_to_evaluate = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, solver='liblinear'),
            'param_grid': {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
            }
        },
        'SVC': {
            'model': SVC(random_state=42, probability=True),
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf']
            }
        },
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        }
    }

    all_models_results = []
    generated_files = [] # 생성된 파일 목록을 저장할 리스트

    for model_name, config in models_to_evaluate.items():
        print(f"\n--- {model_name} 모델 튜닝 시작 ---")
        grid_search = GridSearchCV(config['model'], config['param_grid'], cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)

        all_models_results.append({
            'Model': model_name,
            'Best Parameters': grid_search.best_params_,
            'ROC-AUC Score': roc_auc,
            'Accuracy': best_estimator.score(X_test, y_test),
            'Best Estimator': best_estimator
        })

        print(f"{model_name} - 최적 하이퍼파라미터: {grid_search.best_params_}")
        print(f"{model_name} - 교차 검증 Best ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"{model_name} - 테스트 세트 ROC-AUC: {roc_auc:.4f}")
        print(f"{model_name} - 테스트 세트 정확도: {best_estimator.score(X_test, y_test):.4f}")
        print(f"\n{model_name} 분류 리포트 (테스트 세트):")
        print(classification_report(y_test, y_pred))
        print(f"\n{model_name} 혼동 행렬 (테스트 세트):")
        print(confusion_matrix(y_test, y_pred))

        # ROC Curve 시각화 및 파일 저장
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_curve_filename = f'{model_name}_roc_curve.png'
        plt.savefig(roc_curve_filename) # 파일로 저장
        plt.close() # 그래프 창 닫기
        print(f"'{roc_curve_filename}' 파일이 생성되었습니다.")
        generated_files.append(roc_curve_filename)

        # 모델별 모든 GridSearchCV 결과 텍스트 파일 저장
        all_results_df_model = pd.DataFrame(grid_search.cv_results_)
        all_results_filename = f"{model_name}_all_results.txt"
        with open(all_results_filename, 'w', encoding='utf-8') as f:
            f.write(f"--- {model_name} - 모든 GridSearchCV 결과 ---\n\n")
            f.write(all_results_df_model.to_string())
            f.write("\n\n--- Best Parameters ---\n")
            f.write(str(grid_search.best_params_))
            f.write(f"\nBest ROC-AUC Score (Cross-Validation): {grid_search.best_score_:.4f}\n")
        print(f"'{all_results_filename}' 파일이 생성되었습니다.")
        generated_files.append(all_results_filename)

    print("\n[모든 모델 평가 완료]")
    results_df = pd.DataFrame(all_models_results)
    results_df_display = results_df[['Model', 'Best Parameters', 'ROC-AUC Score', 'Accuracy']].sort_values(by='ROC-AUC Score', ascending=False)
    print("\n--- 모든 모델의 최종 성능 비교 ---")
    print(results_df_display)

    print("\n가장 성능이 좋은 모델:")
    best_overall_model_row = results_df_display.iloc[0]
    print(best_overall_model_row)

    # 최종 성능 요약 테이블 이미지 저장
    results_df_display['Best Parameters Display'] = results_df_display['Best Parameters'].apply(lambda x: str(x))
    fig, ax = plt.subplots(figsize=(16, (len(results_df_display) * 0.7) + 2))
    ax.axis('tight')
    ax.axis('off')
    table_data = results_df_display[['Model', 'Best Parameters Display', 'ROC-AUC Score', 'Accuracy']].round(4)
    table = ax.table(cellText=table_data.values,
                         colLabels=table_data.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    plt.title('Overall Model Performance Summary with Best Parameters', fontsize=16, y=1.05)
    overall_summary_filename = 'overall_model_performance_summary.png'
    plt.savefig(overall_summary_filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"'{overall_summary_filename}' 파일이 생성되었습니다.")
    generated_files.append(overall_summary_filename)

    # 클러스터링된 데이터 CSV 파일도 생성된 파일 목록에 추가
    generated_files.append('clustered_data.csv')
    generated_files.append('feature_importances.png')
    generated_files.append('pca_clustering_visualization.png')


    print("\n\n--- 모든 작업 완료 ---")
    print("생성된 파일 목록:")
    for f_name in sorted(list(set(generated_files))): # 중복 제거 후 정렬하여 출력
        print(f"- {f_name}")
    print("--------------------")

# ✅ 실행 코드
# 'healthcare-dataset-stroke-data.csv' 파일이 같은 디렉토리에 있는지 확인하세요.
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
stroke_prediction_pipeline(df)
