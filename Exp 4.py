import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score, f1_score, roc_curve, r2_score, mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

pd.set_option('display.max_columns', None)

df = pd.read_csv("Churn_Modelling.csv")

print(df.head())

print()

#pre-processing
print(df.isnull().any())

df.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)

print()
print(df.head())

print()
print(df.dtypes)

lbe = LabelEncoder()
cat_cols = [col for col in df.columns if df[col].dtype == 'O']

print()
print(cat_cols)

print()
print(df['Geography'].unique())
print(df['Gender'].unique())

for col in cat_cols:
    if len(df[col].unique()) < 10:
        df[col] = lbe.fit_transform(df[col])
    else:
        df.drop(col,axis=1,inplace=True)

print()
print(df.head())

print()
print(df.dtypes)

inputs = df.drop('Exited',axis=1)

print("\nInputs")
print(inputs.head())

target = df['Exited']

print("\nTarget")
print(target.head())

x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)

#Classification

model = RandomForestClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print()
print(y_pred)

cm_res = confusion_matrix(y_test,y_pred)

print("\nConfusion Matrix")
print(cm_res)

acc = accuracy_score(y_test,y_pred)

print("\nAccuracy")
print(acc)

err_rate = 1 - acc

print("\nError Rate")
print(err_rate)

prec = precision_score(y_test,y_pred)

print("\nPrecision")
print(prec)

tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()

print("\ntn fp fn tp")
print(tn,fp,fn,tp)

sensitiviy = tp / (tp+fn)

print("\nSensitivity,Recall,Power")
print(sensitiviy)

specificity = tn / (tn + fp)

print("\nSpecificity")
print(specificity)

roc = roc_auc_score(y_test,y_pred)

print("\nROC")
print(roc)

f1_sco = f1_score(y_test,y_pred)

print("\nF1 Score")
print(f1_sco)

gm = math.sqrt(sensitiviy * specificity)

print("\nGM: ",gm)

print("\nFPR:",1-specificity)

print("\nFNR:",1-sensitiviy)

fpr, tpr, threshold = roc_curve(y_test,y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
# plt.show()


#Regression

df = pd.read_csv("Salary_Data.csv")

print()
print(df.head())

X = df['YearsExperience']
Y = df['Salary']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

print()
print(Y_pred)

corr , _ = pearsonr(X,Y)
print("\nKarl Pearson's coefficient of correlation:", corr)

R_sq = r2_score(Y_test,Y_pred)
print("\nR-squared:", R_sq)

mse = mean_squared_error(Y_test,Y_pred)
print("\nMean Squared Error:", mse)

rmse = math.sqrt(mse)
print("\nRMSE:", rmse)

Y_pred_df = pd.DataFrame(Y_pred)

num = np.sum(np.square(Y - Y_pred_df))
den = np.sum(np.square(Y_pred_df))
n = Y.shape[0]
squared_error = num/(n*den)
rmsre = np.sqrt(squared_error)
print("\nRMSRE:",rmsre[0])

mae = mean_absolute_error(Y_test,Y_pred)
print("\nMAE:",mae)

mape = mean_absolute_percentage_error(Y_test,Y_pred)
print("\nMAPE: ",mape)

