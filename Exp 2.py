import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)

df = pd.read_csv('loan_data_set.csv')

# print(df.head())

x = df.isnull().any()

print(df['LoanAmount'].head())

#mean imputation
df_loanamt_mean = df['LoanAmount'].fillna(df['LoanAmount'].mean())

print()

print(df_loanamt_mean.head())

#median imputation
df_loanamt_median = df['LoanAmount'].fillna(df['LoanAmount'].median())

print(df_loanamt_median.head())

#mode imputation
df_loanamt_mode = df['LoanAmount'].fillna(df['LoanAmount'].mode())

print(df_loanamt_mode.head())

#End of tail imputation
dff = df['LoanAmount']

extreme = dff.mean() + 3 * dff.std()

df_loanamt_tail = dff.fillna(extreme)

print(df_loanamt_tail.head())

df_loanamt_random = dff.fillna(dff.isnull().sum())

print(df_loanamt_random.head())

#Regression
lr = LinearRegression()

df1 = df[["CoapplicantIncome","LoanAmount"]]

print(df1.head())

testdf = df1[df1['LoanAmount'].isnull() == True]

print()
print(testdf)

traindf = df1[df1['LoanAmount'].isnull()==False]
print()
print(traindf.head())

print()
lr.fit(traindf[['CoapplicantIncome']],traindf[['LoanAmount']])

pred = lr.predict(testdf[['CoapplicantIncome']])

print()
print(pred)
