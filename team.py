# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/heejin/Downloads/healthcare-dataset-stroke-data.csv")
df
# %%
df.shape
# %%
df.columns
# %%
# 결측치 확인
df.isnull().sum()
# %%
# bmi가 결측치인 row 추출
# mis_c = df[df.isnull().any(axis=1)]
# mis_c.head(60)

#%%
df.describe()
# %%
# bmi열의 결측치에 bmi의 중앙값인 28.1을 입력함
df['bmi'].fillna(df['bmi'].median(), inplace=True)
# %%
# 결측치 -> 평균으로 대체되었는지 확인용
df.isnull().sum()
# %%
# 통계값 바뀌었는지 확인 
# df.describe()
df.info
# %%
# 성별이 Other인 row만 삭제 == 성별이 Other이 아닌 데이터들만 남기고 새로운 df인 df_g 만들기기
df_g = df[df['gender'] != "Other"]
df_g
# %%
# 나이가 20세 미만인 row를 search
df_a = df_g[df_g['age'] >= 20.00 ]
df_a
# %%
df_a.info()
# %%
# bmi에서 이상치 찾음(97.6)
# bmi에 대한 scatter 또는 histogram으로 분포 확인 필요성 있음
# df로
import seaborn as sns

sns.scatterplot(data=df, x='age', y='bmi')
plt.show()
# %%
# df_a로
sns.scatterplot(data=df_a, x='age', y='bmi',  color='green')
plt.show()

# --> df와 df_a의 결과가 같음
# outlier로 bmi >= 80을 제외하고 df 새로 만들기
# %%
print(df_a[df_a['bmi'] >= 80])
# %%
df_b = df_a[df_a['bmi'] < 80 ]
df_b
# %%
# scatter plot = df_b로
sns.scatterplot(data=df_b, x='age', y='bmi',  color='orange')
plt.show()
# %%
# 범주형 데이터(문자열 타입) 원핫인코딩? (라벨 분류로 함함)
# 대상: work_type, Residence_type, smoking_status
from sklearn.preprocessing import LabelEncoder
# 1) work_type 대상 
le_w = LabelEncoder()
label_w = le_w.fit_transform(df_b['work_type'])
df_b.drop("work_type", axis=1, inplace=True)
df_b['work_type'] = label_w
df_b.head(50)

# %%
sns.scatterplot(data=df_b, x='work_type', y='age',  color='purple')
plt.show()
# %%
# work_type=1.0인 row 찾기(테스트) <-- 혹시 children인지 확인하고 싶었음
# children은 20세 이상이기 때문에 df_b에 없음을 위의 plt에서 확인할 수 있음
f = df_b[df_b['work_type'] == 1.0 ]
f

# 61408,Male,23,0,0,No,Never_worked,Urban,125.26,18.7,never smoked,0
# 분류> 
# Govt_job : 0.0
# Never_worked : 1.0
# Private : 2.0
# Self-employed : 3.0
# %%
# 2) smoking_status 대상 
le_s = LabelEncoder()
label_s = le_s.fit_transform(df_b['smoking_status'])
df_b.drop("smoking_status", axis=1, inplace=True)
df_b['smoking_status'] = label_s
df_b

# 분류> 
# Unknown : 0
# formerly smoked : 1
# never smoked : 2
# smokes : 3
# %%
# 3) Residence_type 대상 
le_r = LabelEncoder()
label_r = le_r.fit_transform(df_b['Residence_type'])
df_b.drop("Residence_type", axis=1, inplace=True)
df_b['Residence_type'] = label_r
df_b

# 분류>
# Rural : 0
# Urban : 1
# %%
# 4) ever_married 대상
le_e = LabelEncoder()
label_e = le_e.fit_transform(df_b['ever_married'])
df_b.drop("ever_married", axis=1, inplace=True)
df_b['ever_married'] = label_e
df_b

# 분류> 
# Yes : 1
# No : 0

# %%
# 5) gender 대상
le_g = LabelEncoder()
label_g = le_g.fit_transform(df_b['gender'])
df_b.drop("gender", axis=1, inplace=True)
df_b['gender'] = label_g
df_b

# 분류> 
# Yes : 1
# No : 0

# %%
# stroke를 target feature로 하기 위해서 데이터 프레임 reshape
df_b=df_b[['id', 'gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'work_type', 'smoking_status', 'Residence_type', 'ever_married', 'stroke']]
df_b

# 분류 >
# Male : 1
# Female : 0
# %%
# Feature Selection으로 가장 stroke에 영향력 있는 feature 알아보기

X = df_b.iloc[:, 0:20]
y = df_b.iloc[:, -1] #target 컬럼

# 상관성
corrmat = df_b.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(df_b[top_corr_features].corr(),annot=True, cmap="Reds")


# heatmap은 별 필요가 없는 듯... string을 numeric으로 바꿔서 헷갈림 
# %%
# Feature importance scoring

X = df_b.iloc[:, 1:11]
y = df_b.iloc[:, -1] #target 컬럼

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')

plt.show()

# age, 글루코스 평균 수치, bmi 가 가장 큰 3요소로 나타남
# %%
