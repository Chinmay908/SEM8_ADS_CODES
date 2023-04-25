import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv("loan_data_set.csv")

target = df['Loan_Status']
print(target.value_counts())
sns.countplot(x="Loan_Status",data=df)
plt.show()

#preproces
df.drop(['Loan_ID','Dependents'],axis=1,inplace=True)
df.dropna(inplace=True)
print(df.isnull().any())

#encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])
df['Education'] = le.fit_transform(df['Education'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
df['Property_Area'] = le.fit_transform(df['Property_Area'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

print(df.info())

inputs = df.drop(['Loan_Status'],axis=1)
target = df['Loan_Status']

x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.3)

model = KNeighborsClassifier()
model1 = DecisionTreeClassifier()

model.fit(x_train,y_train)
model1.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred1 = model1.predict(x_test)

f1_before = f1_score(y_test,y_pred1)
print("\nF-Score before SMOTE:", f1_before)

smote = SMOTE(sampling_strategy="not majority",k_neighbors=3,random_state=42)
x_resampled,y_resampled = smote.fit_resample(x_train,y_train)
model1.fit(x_resampled,y_resampled)

y_pred1 = model1.predict(x_test)
f1_after = f1_score(y_test,y_pred1)
print("\nF-Score after SMOTE:", f1_after)

