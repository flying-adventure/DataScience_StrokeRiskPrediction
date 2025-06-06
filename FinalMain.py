import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import importlib.util
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# List of model file paths
MODEL_FILES = [
    r"knn.py",
    r"LogisticRegression_ver1.py",
    r"LogisticRegression_ver2.py",
    r"RandomForest_ver1.py",
    r"RandomForest_ver2.py"
]

# Dynamically load models from Python files using get_model() function

def load_models_from_files():
    models = {}
    for filepath in MODEL_FILES:
        model_name = os.path.basename(filepath).split('.')[0]
        spec = importlib.util.spec_from_file_location(model_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        models[model_name] = module.get_model()  # Load model instance
    return models

# Data preprocessing function with optional visualization
def preprocess_data(df, plot=False):
    df = df.copy()

    if plot:
        # 1. Display missing values per column
        missing_values = df.isnull().sum()
        # 2. Display class distribution for target label 'stroke'
        stroke_distribution = df['stroke'].value_counts(normalize=True) * 100

        # 3. Visualize potential outliers
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x='age', y='bmi', alpha=0.5)
        plt.title('Scatter Plot: Age vs BMI')
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=df, x='age', y='avg_glucose_level', alpha=0.5)
        plt.title('Scatter Plot: Age vs Avg Glucose Level')
        plt.tight_layout()
        plt.show()

        print(missing_values)
        print(stroke_distribution)

    # Fill missing BMI values with median
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    # Remove rows with undefined gender
    df = df[df['gender'] != 'Other']
    # Keep adults (age >= 20)
    df = df[df['age'] >= 20]
    # Remove outliers with extreme BMI values
    df = df[df['bmi'] < 80]

    # One-hot encode categorical variables
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=['stroke', 'id'])
    y = df_encoded['stroke']
    return X, y

# KMeans clustering with silhouette score and PCA visualization
def run_clustering_analysis(X_scaled, y_res):
    best_k, best_score = 0, -1
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score

    print(f"\n[Clustering] Best K: {best_k}, Silhouette Score: {best_score:.4f}")

    # Final clustering with best_k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Reduce dimensions using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels

    # Visualize clusters
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
    plt.title('PCA Clustering Visualization')
    plt.tight_layout()
    plt.show()

    # Stroke distribution by cluster
    X_clustered = pd.DataFrame(X_scaled, columns=X.columns)
    X_clustered['cluster_label'] = cluster_labels
    df_cluster = X_clustered.copy()
    df_cluster['stroke'] = y_res.reset_index(drop=True)
    print("\n[Stroke ratio statistics by cluster]")
    print(df_cluster.groupby('cluster_label')['stroke'].value_counts(normalize=True).unstack().fillna(0))

    return X_clustered

# Evaluate model performance using Stratified K-Fold Cross-Validation
def cross_validate_model(model, X, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc_scores = []
    auc_scores = []

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"Evaluating Fold {fold}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)  # Apply SMOTE only to training set

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train_res)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))
        fold += 1

    return np.mean(acc_scores), np.mean(auc_scores)

# Plot bar chart comparing model performance
def plot_model_comparison(results):
    df_plot = pd.DataFrame(results).T
    df_plot = df_plot[['accuracy', 'roc_auc']]
    df_plot.plot(kind='bar', figsize=(10,6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Process input patient data and format as encoded DataFrame for prediction
def preprocess_new_data(new_data_list, X_columns):
    df = pd.DataFrame(new_data_list)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    for col in X_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[X_columns]
    return df_encoded

# Select the best model based on a weighted scoring system
# The default weights prioritize ROC AUC (60%) over Accuracy (40%)
def select_best_model(results, weights={'roc_auc': 0.6, 'accuracy': 0.4}):
    scores = {}
    for name, res in results.items():
        # Compute weighted score for each model
        score = (res['roc_auc'] * weights['roc_auc'] + res['accuracy'] * weights['accuracy'])
        scores[name] = score
    # Return model with the highest overall score
    return max(scores.items(), key=lambda x: x[1])

# Function to collect patient information from the user through CLI inputs
# Ensures data integrity and valid ranges for each input
# Returns a list of dictionaries, each representing one patient

def input_patient_data():
    patients = []
    num_patients = None
    while num_patients is None:
        try:
            num_patients = int(input("Enter the number of patients to predict: "))
            if num_patients <= 0:
                print("You must enter at least 1 patient.")
                num_patients = None
        except ValueError:
            print("Please enter a number.")

    for i in range(num_patients):
        print(f"\nEnter information for patient {i+1}:")

        # Gender
        while True:
            gender = input("Gender (Male/Female): ").strip()
            if gender in ['Male', 'Female']:
                break
            print("Gender must be either 'Male' or 'Female'.")

        # Age
        while True:
            try:
                age = float(input("Age: "))
                if 20 <= age <= 120:
                    break
                else:
                    print("Age must be between 20 and 120.")
            except ValueError:
                print("Please enter a number.")

        # Hypertension
        while True:
            try:
                hypertension = int(input("Hypertension (0: No, 1: Yes): "))
                if hypertension in [0, 1]:
                    break
                else:
                    print("Only 0 or 1 is allowed.")
            except ValueError:
                print("Please enter a number.")

        # Heart disease
        while True:
            try:
                heart_disease = int(input("Heart disease (0: No, 1: Yes): "))
                if heart_disease in [0, 1]:
                    break
                else:
                    print("Only 0 or 1 is allowed.")
            except ValueError:
                print("Please enter a number.")

        # Average glucose level
        while True:
            try:
                avg_glucose_level = float(input("Average glucose level: "))
                if 0 <= avg_glucose_level <= 500:
                    break
                else:
                    print("Glucose level must be between 0 and 500.")
            except ValueError:
                print("Please enter a number.")

        # BMI
        while True:
            try:
                bmi = float(input("BMI: "))
                if 10 <= bmi < 80:
                    break
                else:
                    print("BMI must be between 10 and 80.")
            except ValueError:
                print("Please enter a number.")

        # Work type selection
        work_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        while True:
            work_type = input("Work type (Private/Self-employed/Govt_job/children/Never_worked): ").strip()
            if work_type in work_types:
                break
            print(f"Work type must be one of: {', '.join(work_types)}.")

        # Smoking status
        smoking_statuses = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
        while True:
            smoking_status = input("Smoking status (never smoked/formerly smoked/smokes/Unknown): ").strip()
            if smoking_status in smoking_statuses:
                break
            print(f"Smoking status must be one of: {', '.join(smoking_statuses)}.")

        # Residence type
        while True:
            residence_type = input("Residence Type (Urban/Rural): ").strip()
            if residence_type in ['Urban', 'Rural']:
                break
            print("Residence type must be either 'Urban' or 'Rural'.")

        # Marital status
        while True:
            ever_married = input("Marital Status (Yes/No): ").strip()
            if ever_married in ['Yes', 'No']:
                break
            print("Marital status must be 'Yes' or 'No'.")

        # Aggregate patient input into dictionary format
        patient = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'work_type': work_type,
            'smoking_status': smoking_status,
            'Residence_type': residence_type,
            'ever_married': ever_married
        }

        patients.append(patient)

    return patients

# Running Main Function
if __name__ == "__main__":

    # Load Model
    models = load_models_from_files()

    # Load Data
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")

    # Dynamic import and function call of kfoldcross.py (only finding optimal k here)
    kfoldcross_path = r"kfoldcross.py"
    spec = importlib.util.spec_from_file_location("kfoldcross", kfoldcross_path)
    kfoldcross = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kfoldcross)

    print("▶ Finding Optimal K (K-Fold Cross Validation) in progress...")
    X, y = preprocess_data(df, plot=True)
    best_k, k_auc_dict = kfoldcross.find_best_k_and_evaluate(X, y)

    print(f"Optional K: {best_k}")
    print(f"AUC values per K: {k_auc_dict}")

    # Cross-validation per model
    results = {}
    for name, model in models.items():
        print(f"\n▶ Model: {name}")
        acc, auc = cross_validate_model(model, X, y, best_k)
        results[name] = {'accuracy': acc, 'roc_auc': auc}
        print(f"[{name}] Average Accuracy: {acc:.4f}, Average ROC AUC: {auc:.4f}")

    # Output Final Results
    print("\nFinal Evaluation Results per Model")
    for name, res in results.items():
        print(f"<{name}>")
        print(f"Accuracy: {res['accuracy']:.4f}, ROC AUC: {res['roc_auc']:.4f}")
        
    plot_model_comparison(results)
    
    # Select Best Performing Model
    best_model_name, best_score = select_best_model(results)
    best_model = models[best_model_name]
    print(f"\n=== Final Selected Model: {best_model_name} (Overall Score: {best_score:.4f}) ===")
    

    # Re-train on the entire dataset and define the scaler
    print("\n=== Re-train Final Model ===")
    X, y = preprocess_data(df, plot=False)
    
    # Apply SMOTE (entire dataset)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Re-define and train the scaler
    global_scaler = StandardScaler()
    X_scaled = global_scaler.fit_transform(X_res)
    
    # Re-train the final model
    best_model.fit(X_scaled, y_res)

    # Run clustering analysis (KMeans + PCA)
    X_clustered = run_clustering_analysis(X_scaled, y_res)

    # Receive patient data as user input
    print("\n=== Input new patient information ===")
    new_patients_data = input_patient_data()

    # Preprocess prediction data
    X_new = preprocess_new_data(new_patients_data, X.columns)
    X_new_scaled = global_scaler.transform(X_new)

    # Predict and output results
    probs = best_model.predict_proba(X_new_scaled)[:, 1]
    for i, prob in enumerate(probs):
        print(f"\n--- Patient {i+1} ---")
        print(f"Input data: {new_patients_data[i]}")
        print(f"Predicted stroke probability: {prob:.4f}")
        if prob >= 0.5:
            print("Predicts a high likelihood of stroke.")
        else:
            print("Predicts a low likelihood of stroke.")