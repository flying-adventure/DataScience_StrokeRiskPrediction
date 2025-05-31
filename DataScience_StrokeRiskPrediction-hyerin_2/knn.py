import matplotlib
matplotlib.use('Agg') # GUI 관련 오류 방지를 위해 코드 상단에 추가

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def stroke_prediction_pipeline_fixed_knn(df):
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
    # 4. Feature Importance (분석 단계는 유지)
    # ------------------------
    model = ExtraTreesClassifier(random_state=42)
    model.fit(X, y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n[Feature Importance 분석]")
    print("Top 10 Feature Importances:")
    print(feat_importances.nlargest(10))
    # 시각화 파일 저장 (콘솔 출력용이므로 plt.show() 대신 savefig 사용)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
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
    # 7. KMeans 클러스터링 (단계 유지)
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
    # 8. 클러스터 시각화 (PCA) (단계 유지)
    # ------------------------
    print("\n[클러스터 시각화 (PCA)]")
    pca = PCA(n_components=2)
    X_pca_full = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca_full, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2', alpha=0.7)
    plt.title('PCA Clustering Visualization (KMeans Clusters)')
    plt.tight_layout()
    plt.savefig('pca_clustering_visualization.png')
    plt.close()
    print("PCA를 이용한 클러스터 시각화 ('pca_clustering_visualization.png') 완료.")

    # ------------------------
    # 9. 클러스터별 통계 출력 및 데이터 저장 (단계 유지)
    # ------------------------
    X_clustered_features = pd.DataFrame(X_scaled, columns=X.columns)
    X_clustered = X_clustered_features.copy()
    X_clustered['cluster_label'] = cluster_labels
    df_cluster = X_clustered.copy()
    df_cluster['stroke'] = y_res.reset_index(drop=True)
    print("\n[클러스터별 stroke 비율 통계 (스케일링된 데이터 기반)]")
    print(df_cluster.groupby('cluster_label')['stroke'].value_counts(normalize=True).unstack().fillna(0))
    print("클러스터별 통계 출력 완료.")
    df_cluster.to_csv('clustered_data.csv', index=False)
    print("클러스터링된 데이터가 'clustered_data.csv' 파일로 저장되었습니다.")

    # ------------------------
    # 10. Train/Test Split
    # ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_clustered, y_res, test_size=0.25, random_state=42, stratify=y_res
    )
    print("데이터 분리 완료 (Train:", X_train.shape, "Test:", X_test.shape, ").")






    # -----------------------------------------------
    # 11. KNN 모델 (n_neighbors=13 고정) 상세 평가 보고서
    # -----------------------------------------------
    print("\n[KNN 모델 (n_neighbors=13 고정) 상세 평가 보고서]")

    fixed_n_neighbors = 13
    weights_options = ['uniform', 'distance']
    p_options = [1, 2]

    # 각 조합별 평가 결과 출력을 위한 루프
    all_knn_test_evaluations = [] # 모든 KNN 조합의 테스트 세트 평가 결과를 저장할 리스트
    for weights_val in weights_options:
        for p_val in p_options:
            current_params = {'n_neighbors': fixed_n_neighbors, 'weights': weights_val, 'p': p_val}
            
            # 현재 파라미터 조합으로 KNN 모델 생성 및 훈련
            current_knn_model = KNeighborsClassifier(**current_params)
            current_knn_model.fit(X_train, y_train)
            
            # 테스트 세트 예측
            current_y_pred = current_knn_model.predict(X_test)
            current_y_prob = current_knn_model.predict_proba(X_test)[:, 1]

            # 평가 지표 계산
            current_accuracy = current_knn_model.score(X_test, y_test)
            current_report_dict = classification_report(y_test, current_y_pred, output_dict=True, zero_division=0)
            current_roc_auc = roc_auc_score(y_test, current_y_prob)

            # Classification Report 출력 
            print(f"\n--- KNN 평가 결과 (n_neighbors={fixed_n_neighbors}, weights='{weights_val}', p={p_val}) ---")
            print(f"Accuracy: {current_accuracy:.15f}")

            print("Classification Report:")
            # 헤더 출력 (정렬 맞춤)
            print(" " * 11 + "{:<10}{:<10}{:<10}{:>10}".format("precision", "recall", "f1-score", "support"))

            # 클래스 0, 1 결과 출력 (정렬 맞춤)
            for cls_label in ['0', '1']:
                metrics = current_report_dict[cls_label]
                print(f" {cls_label:<10}  {metrics['precision']:.2f}    {metrics['recall']:.2f}    {metrics['f1-score']:.2f}    {metrics['support']:.0f}")

            # 'accuracy' 출력 (정렬 맞춤)
            print(f"\n accuracy      {current_report_dict['accuracy']:.4f}    {len(y_test):.0f}")

            # 'macro avg' 출력 (정렬 맞춤)
            macro_avg = current_report_dict['macro avg']
            print(f" macro avg   {macro_avg['precision']:.2f}    {macro_avg['recall']:.2f}    {macro_avg['f1-score']:.2f}    {macro_avg['support']:.0f}")

            # 'weighted avg' 출력 (정렬 맞춤)
            weighted_avg = current_report_dict['weighted avg']
            print(f"weighted avg {weighted_avg['precision']:.2f}    {weighted_avg['recall']:.2f}    {weighted_avg['f1-score']:.2f}    {weighted_avg['support']:.0f}")

            print(f"\nROC AUC: {current_roc_auc:.15f}")
            print("-" * 70)

            # -----------------------------------------------
            # 각 파라미터 조합별 ROC Curve 저장 (이미지 파일)
            # -----------------------------------------------
            fpr_current, tpr_current, _ = roc_curve(y_test, current_y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_current, tpr_current, color='darkorange', lw=2, label=f'AUC = {current_roc_auc:.2f} (Test Set)')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            param_str = "_".join([f"{k}{v}" for k, v in current_params.items()]) # current_params 사용
            roc_curve_filename_combo = f'KNeighborsClassifier_roc_curve_{param_str}.png'
            plt.title(f'KNN ROC Curve (Params: {current_params})')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_curve_filename_combo)
            plt.close()
            print(f"'{roc_curve_filename_combo}' 파일이 생성되었습니다.")

            # 모든 KNN 조합의 테스트 세트 평가 결과를 리스트에 추가 (상세 포함)
            all_knn_test_evaluations.append({
                'Parameters': current_params,
                'Accuracy': current_accuracy,
                'Classification Report': current_report_dict,
                'Confusion Matrix': confusion_matrix(y_test, current_y_pred).tolist(),
                'ROC AUC': current_roc_auc
            })

    print("\n[KNN 모델 (n_neighbors=5 고정) 상세 평가 보고서 출력 완료]")

    # 모든 KNN 조합의 테스트 세트 평가 결과를 텍스트 파일로 저장
    all_test_eval_filename = 'KNeighborsClassifier_all_test_evaluations.txt'
    with open(all_test_eval_filename, 'w', encoding='utf-8') as f:
        f.write("--- KNeighborsClassifier - 모든 파라미터 조합별 테스트 세트 평가 결과 ---\n\n")
        for eval_result in all_knn_test_evaluations:
            f.write(f"Parameters: {eval_result['Parameters']}\n")
            f.write(f"Accuracy: {eval_result['Accuracy']}\n")
            f.write("Classification Report:\n")
            for cls_name, metrics in eval_result['Classification Report'].items():
                if isinstance(metrics, dict):
                    f.write(f"  Class {cls_name}:\n")
                    f.write(f"    Precision: {metrics.get('precision', 'N/A'):.4f}\n")
                    f.write(f"    Recall: {metrics.get('recall', 'N/A'):.4f}\n")
                    f.write(f"    F1-Score: {metrics.get('f1-score', 'N/A'):.4f}\n")
                    f.write(f"    Support: {metrics.get('support', 'N/A')}\n")
                else:
                    f.write(f"  {cls_name}: {metrics:.4f}\n")
            f.write(f"Confusion Matrix: {eval_result['Confusion Matrix']}\n")
            f.write(f"ROC AUC: {eval_result['ROC AUC']}\n")
            f.write("-" * 70 + "\n\n")
    print(f"모든 KNN 조합의 테스트 세트 평가 결과가 '{all_test_eval_filename}' 파일로 저장되었습니다.")
