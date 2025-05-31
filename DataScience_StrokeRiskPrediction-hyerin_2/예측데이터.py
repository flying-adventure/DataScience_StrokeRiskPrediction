# -----------------------------------------------------------------------------
# 새로운 환자 정보로 뇌졸중 확률 예측
# (team.py 실행 후 동일 세션에서 실행 필요)
# -----------------------------------------------------------------------------

print("하이퍼파라미터 튜닝, 교차 검증, 최종 모델 평가가 완료되었습니다.")

# 새로운 환자 정보로 뇌졸중 확률 예측
print("\n--- 새로운 환자 뇌졸중 확률 예측 ---")

# 예측할 환자 정보 입력 (다양한 나이대 포함)
new_patients_data = [
    {
        'gender': 'Female', 'age': 30.0, 'hypertension': 0, 'heart_disease': 0,
        'avg_glucose_level': 90.0, 'bmi': 25.0, 'work_type': 'Private',
        'smoking_status': 'never smoked', 'Residence_type': 'Urban', 'ever_married': 'Yes'
    },
    {
        'gender': 'Male', 'age': 55.0, 'hypertension': 1, 'heart_disease': 0,
        'avg_glucose_level': 130.0, 'bmi': 30.0, 'work_type': 'Self-employed',
        'smoking_status': 'formerly smoked', 'Residence_type': 'Rural', 'ever_married': 'Yes'
    },
    {
        'gender': 'Female', 'age': 75.0, 'hypertension': 0, 'heart_disease': 1,
        'avg_glucose_level': 200.0, 'bmi': 35.0, 'work_type': 'Private',
        'smoking_status': 'smokes', 'Residence_type': 'Urban', 'ever_married': 'Yes'
    },
    {
        'gender': 'Male', 'age': 25.0, 'hypertension': 0, 'heart_disease': 0,
        'avg_glucose_level': 80.0, 'bmi': 22.0, 'work_type': 'Govt_job',
        'smoking_status': 'never smoked', 'Residence_type': 'Rural', 'ever_married': 'No'
    },
    {
        'gender': 'Female', 'age': 65.0, 'hypertension': 0, 'heart_disease': 0,
        'avg_glucose_level': 100.0, 'bmi': 28.0, 'work_type': 'Private',
        'smoking_status': 'never smoked', 'Residence_type': 'Urban', 'ever_married': 'Yes'
    }
]

for i, patient_data in enumerate(new_patients_data):
    print(f"\n--- 환자 {i+1} 정보 예측 ---")
    new_patient_df = pd.DataFrame([patient_data])

    print("입력 환자 데이터:")
    print(new_patient_df)

    # 입력 데이터 전처리 (훈련 데이터와 동일한 방식 적용)
    new_patient_df['gender'] = le_g.transform(new_patient_df['gender'])
    new_patient_df['Residence_type'] = le_r.transform(new_patient_df['Residence_type'])
    new_patient_df['ever_married'] = le_e.transform(new_patient_df['ever_married'])

    categorical_cols_one_hot_predict = ['work_type', 'smoking_status']
    new_patient_df = pd.get_dummies(new_patient_df, columns=categorical_cols_one_hot_predict, drop_first=True, dtype=int)

    # 훈련 데이터(X)의 컬럼 순서와 이름에 맞게 예측 데이터 정렬
    final_prediction_data_aligned = new_patient_df.reindex(columns=X.columns, fill_value=0)

    # 특성 스케일링 적용
    new_patient_scaled = scaler.transform(final_prediction_data_aligned)
    new_patient_scaled_df = pd.DataFrame(new_patient_scaled, columns=final_prediction_data_aligned.columns)

    # 클러스터 라벨 예측
    predicted_cluster_label = kmeans.predict(new_patient_scaled_df)
    new_patient_scaled_df['cluster_label'] = predicted_cluster_label[0] # 예측은 배열로 나오므로 첫 번째 요소 사용

    # 예측을 위한 최종 데이터 준비
    final_prediction_data = new_patient_scaled_df 

    print("\n처리된 환자 데이터 (스케일링 + 클러스터 라벨):")
    print(final_prediction_data)

    # 뇌졸중 발병 확률 예측
    stroke_probability = best_model.predict_proba(final_prediction_data)[:, 1] # 뇌졸중(클래스 1) 확률

    print(f"\n예측된 뇌졸중 확률: {stroke_probability[0]:.4f}")

    # 예측 결과 해석
    if stroke_probability[0] >= 0.5: # 일반적으로 0.5를 임계값으로 사용
        print("뇌졸중 발생 가능성이 높다고 예측합니다.")
    else:
        print("뇌졸중 발생 가능성이 낮다고 예측합니다.")

print("\n--- 모든 환자 예측 완료 ---")
