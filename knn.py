import matplotlib
matplotlib.use('Agg') # Add this to the top of your code to prevent GUI-related errors.

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
    # 1. Handle missing values and outliers.
    # ------------------------
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]
    print("Completed Data Pre-processing: Remove data for individuals under 20 years old and those with a BMI over 80...")

    # ------------------------
    # 2. One-hot Encoding
    # ------------------------
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print("One-hot Encoding of Categorical variables..")

    # ------------------------
    # 3. Divide Feature/Target
    # ------------------------
    X = df_encoded.drop(columns=['stroke', 'id'])
    y = df_encoded['stroke']
    print("Complete Feature (X) and Target (y) separation..")

    # ------------------------
    # 4. Feature Importance
    # ------------------------
    model = ExtraTreesClassifier(random_state=42)
    model.fit(X, y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n[Feature Importance Analysis]")
    print("Top 10 Feature Importances:")
    print(feat_importances.nlargest(10))
    # Save the visualization file (use savefig instead of plt.show() for console output)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    print("Complete Feature Importance VIsualization ('feature_importances.png')..")


    # ------------------------
    # 5. Handle class imbalance with SMOTE
    # ------------------------
    print("\n[Apply SMOTE]")
    print("Class distribution before applying SMOTE:\n", y.value_counts())
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("Class distribution after applying SMOTE:\n", y_res.value_counts())
    print("Class imbalance handled with SMOTE..")

    # ------------------------
    # 6. Scaling
    # ------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    print("Complete Data Scaling (StandardScaler)..")

    # ------------------------
    # 7. K-Means Clustering (Maintain steps)
    # ------------------------
    print("\n[K-Means Clustering]")
    best_k, best_score = 0, -1
    print("Optimal K search (based on Silhouette Score):")
    for k in range(2, 11):
        kmeans_eval = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_eval.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"K={k}, Silhouette Score={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"\nOptional K: {best_k}, Best Silhouette Score: {best_score:.4f}")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    print("Complete K-Means Clustering..")

    # ------------------------
    # 8. Cluster Visualization (PCA) (Maintain steps)
    # ------------------------
    print("\n[Cluster Visualization (PCA)]")
    pca = PCA(n_components=2)
    X_pca_full = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca_full, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2', alpha=0.7)
    plt.title('PCA Clustering Visualization (K-Means Clusters)')
    plt.tight_layout()
    plt.savefig('pca_clustering_visualization.png')
    plt.close()
    print("Complete Cluster visualization using PCA ('pca_clustering_visualization.png')..")

    # ------------------------
    # 9. Output cluster-wise statistics and save data (Maintain steps)
    # ------------------------
    X_clustered_features = pd.DataFrame(X_scaled, columns=X.columns)
    X_clustered = X_clustered_features.copy()
    X_clustered['cluster_label'] = cluster_labels
    df_cluster = X_clustered.copy()
    df_cluster['stroke'] = y_res.reset_index(drop=True)
    print("\n[Stroke ratio statistics per cluster (based on scaled data)]")
    print(df_cluster.groupby('cluster_label')['stroke'].value_counts(normalize=True).unstack().fillna(0))
    print("Complete Cluster-wise statistics output..")
    df_cluster.to_csv('clustered_data.csv', index=False)
    print("Clustered data saved as 'clustered_data.csv'.")

    # ------------------------
    # 10. Train/Test Split
    # ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_clustered, y_res, test_size=0.25, random_state=42, stratify=y_res
    )
    print("Data Separation Complete.. (Train:", X_train.shape, "Test:", X_test.shape, ").")


    # -----------------------------------------------
    # 11. KNN Model (n_neighbors=5 fixed) Detailed Evaluation Report
    # -----------------------------------------------
    print("\n[Detailed Evaluation Report of KNN Model (n_neighbors=13 fixed)]")

    fixed_n_neighbors = 13
    weights_options = ['uniform', 'distance']
    p_options = [1, 2]

    # Loop for outputting evaluation results for each combination
    all_knn_test_evaluations = [] # List to store test set evaluation results for all KNN combinations
    for weights_val in weights_options:
        for p_val in p_options:
            current_params = {'n_neighbors': fixed_n_neighbors, 'weights': weights_val, 'p': p_val}
            
            # Create and train KNN model with current parameter combination
            current_knn_model = KNeighborsClassifier(**current_params)
            current_knn_model.fit(X_train, y_train)
            
            # Test set prediction
            current_y_pred = current_knn_model.predict(X_test)
            current_y_prob = current_knn_model.predict_proba(X_test)[:, 1]

            # Calculate evaluation metrics
            current_accuracy = current_knn_model.score(X_test, y_test)
            current_report_dict = classification_report(y_test, current_y_pred, output_dict=True, zero_division=0)
            current_roc_auc = roc_auc_score(y_test, current_y_prob)

            # Print Classification Report
            print(f"\n--- KNN evaluation results (n_neighbors={fixed_n_neighbors}, weights='{weights_val}', p={p_val}) ---")
            print(f"Accuracy: {current_accuracy:.15f}")

            print("Classification Report:")
            # Print header (alignment adjusted)
            print(" " * 11 + "{:<10}{:<10}{:<10}{:>10}".format("precision", "recall", "f1-score", "support"))

            # Class 0, 1 Output results (alignment adjusted)
            for cls_label in ['0', '1']:
                metrics = current_report_dict[cls_label]
                print(f" {cls_label:<10}  {metrics['precision']:.2f}    {metrics['recall']:.2f}    {metrics['f1-score']:.2f}    {metrics['support']:.0f}")

            # 'accuracy' Output results (alignment adjusted)
            print(f"\n accuracy      {current_report_dict['accuracy']:.4f}    {len(y_test):.0f}")

            # 'macro avg' Output results (alignment adjusted)
            macro_avg = current_report_dict['macro avg']
            print(f" macro avg   {macro_avg['precision']:.2f}    {macro_avg['recall']:.2f}    {macro_avg['f1-score']:.2f}    {macro_avg['support']:.0f}")

            # 'weighted avg' Output results (alignment adjusted)
            weighted_avg = current_report_dict['weighted avg']
            print(f"weighted avg {weighted_avg['precision']:.2f}    {weighted_avg['recall']:.2f}    {weighted_avg['f1-score']:.2f}    {weighted_avg['support']:.0f}")

            print(f"\nROC AUC: {current_roc_auc:.15f}")
            print("-" * 70)

            # -----------------------------------------------
            # Save ROC Curve for each parameter combination (image file)
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
            print(f"'{roc_curve_filename_combo}' File created.")

            # Add test set evaluation results for all KNN combinations to the list (including details)
            all_knn_test_evaluations.append({
                'Parameters': current_params,
                'Accuracy': current_accuracy,
                'Classification Report': current_report_dict,
                'Confusion Matrix': confusion_matrix(y_test, current_y_pred).tolist(),
                'ROC AUC': current_roc_auc
            })

    print("\n[KNN Model (n_neighbors=5 fixed) detailed evaluation report output complete]")

    # Save test set evaluation results for all KNN combinations to a text file
    all_test_eval_filename = 'KNeighborsClassifier_all_test_evaluations.txt'
    with open(all_test_eval_filename, 'w', encoding='utf-8') as f:
        f.write("--- KNeighborsClassifier - Test set evaluation results for all parameter combinations ---\n\n")
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
    print(f"All KNN combination test set evaluation results saved to '{all_test_eval_filename}'.")


df = pd.read_csv("healthcare-dataset-stroke-data.csv")
stroke_prediction_pipeline_fixed_knn(df)
