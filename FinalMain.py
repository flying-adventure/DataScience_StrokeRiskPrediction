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

# Set model file path
MODEL_FILES = [
    r"/content/drive/MyDrive/데이터/knn.py",
    r"/content/drive/MyDrive/데이터/LogisticRegression_ver1.py",
    r"/content/drive/MyDrive/데이터/LogisticRegression_ver2.py",
    r"/content/drive/MyDrive/데이터/RandomForest_ver1.py",
    r"/content/drive/MyDrive/데이터/RandomForest_ver2.py"
]

# Function to classify loaded models
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

# Data pre-processing Function
def preprocess_data(df):
    df = df.copy()
    
    # 1. Check missing values
    missing_values = df.isnull().sum()
    # 2. Check class imbalance (stroke distribution)
    stroke_distribution = df['stroke'].value_counts(normalize=True) * 100

    # 3. Visualize outliers using scatter plots
    plt.figure(figsize=(15, 5))
    # age vs bmi
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='age', y='bmi', alpha=0.5)
    plt.title('Scatter Plot: Age vs BMI')
    # age vs avg_glucose_level
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='age', y='avg_glucose_level', alpha=0.5)
    plt.title('Scatter Plot: Age vs Avg Glucose Level')
    plt.tight_layout()
    plt.show()
    # Return missing value counts and class imbalance
    print(missing_values)
    print(stroke_distribution)

    
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    X = df_encoded.drop(columns=['stroke', 'id'])
    y = df_encoded['stroke']
    return X, y

# Function to evaluate models based on K-Fold (evaluates separately configured models,
# independent of kfoldcross.py; k is obtained from kfoldcross.py)
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
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train) #  SMOTE is applied only to the training set

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

# Preprocessing function for new patient predictions.
def preprocess_new_data(new_data_list, X_columns):
    df = pd.DataFrame(new_data_list)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['age'] >= 20]
    df = df[df['bmi'] < 80]
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    # Correct for missing columns
    for col in X_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[X_columns]
    return df_encoded

# Final model selection function (ROC AUC 60%, Accuracy 40% weight)
def select_best_model(results, weights={'roc_auc': 0.6, 'accuracy': 0.4}):
    scores = {}
    for name, res in results.items():
        score = (res['roc_auc'] * weights['roc_auc'] +
                 res['accuracy'] * weights['accuracy'])
        scores[name] = score
    return max(scores.items(), key=lambda x: x[1])

# Patient information input function
def input_patient_data():
    patients = []
    num_patients = int(input("Enter the number of patients to predict: "))

    for i in range(num_patients):
        print(f"\nEnter patient {i+1} information:")
        patient = {
            'gender': input("Gender (Male/Female): ").strip(),
            'age': float(input("Age (more than 19): ")),
            'hypertension': int(input("Hypertension status (0: no, 1: yes): ")),
            'heart_disease': int(input("Heart disease (0: no, 1: yes): ")),
            'avg_glucose_level': float(input("Average Glucose Level: ")),
            'bmi': float(input("BMI: ")),
            'work_type': input("Work Type (Private/Self-employed/Govt_job/children/Never_worked): ").strip(),
            'smoking_status': input("Smoking Status (never smoked/formerly smoked/smokes/Unknown): ").strip(),
            'Residence_type': input("Residence Type (Urban/Rural): ").strip(),
            'ever_married': input("Marital status (Yes/No): ").strip()
        }
        patients.append(patient)
    return patients



# Running Main Function
if __name__ == "__main__":

    # Load Model
    models = load_models_from_files()

    # Load Data
    df = pd.read_csv("/content/drive/MyDrive/데이터/healthcare-dataset-stroke-data.csv")

    # Dynamic import and function call of kfoldcross.py (only finding optimal k here)
    kfoldcross_path = r"/content/drive/MyDrive/데이터/kfoldcross.py"
    spec = importlib.util.spec_from_file_location("kfoldcross", kfoldcross_path)
    kfoldcross = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kfoldcross)

    print("▶ Finding Optimal K (K-Fold Cross Validation) in progress...")
    best_k, k_auc_dict = kfoldcross.find_best_k_and_evaluate(df)
    print(f"Optional K: {best_k}")
    print(f"AUC values per K: {k_auc_dict}")

    # Pre-Processing
    X, y = preprocess_data(df)

    # Cross-validation per model
    results = {}
    for name, model in models.items():
        print(f"\n▶ Model: {name}")
        acc, auc = cross_validate_model(model, X, y, best_k)
        results[name] = {'accuracy': acc, 'roc_auc': auc}
        print(f"[{name}] Average Accuracy: {acc:.4f}, Average ROC AUC: {auc:.4f}")

    # Output Final Results
    print("\n⭐Final Evaluation Results per Model⭐")
    for name, res in results.items():
        print(f"<{name}>")
        print(f"Accuracy: {res['accuracy']:.4f}, ROC AUC: {res['roc_auc']:.4f}")

    # Select Best Performing Model
    best_model_name, best_score = select_best_model(results)
    best_model = models[best_model_name]
    print(f"\n=== Final Selected Model: {best_model_name} (Overall Score: {best_score:.4f}) ===")

    # Re-train on the entire dataset and define the scaler
    print("\n=== Re-train Final Model ===")
    X, y = preprocess_data(df)
    
    # Apply SMOTE (entire dataset)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Re-define and train the scaler
    global_scaler = StandardScaler()
    X_scaled = global_scaler.fit_transform(X_res)
    
    # Re-train the final model
    best_model.fit(X_scaled, y_res)
  
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
