import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)
# plt.interactive(False)

df = pd.read_csv("diamonds.csv")

df.drop("Unnamed: 0",axis=1,inplace=True)

print(df.head())

print()

print(df.describe())

print()

print(df.isnull().any())

print()

df_numeric = df.select_dtypes(include=np.number)

print(df_numeric.head())

print()

print(df_numeric.median())

print()

print(df.mode())

print()

print(df_numeric.sem())

print()

print(df_numeric.skew())

print()

print(df_numeric.std())

print()

print(df_numeric.var())

print()

print(df_numeric.kurtosis())

corr = df_numeric.corr()
sns.heatmap(corr,cmap="Blues",annot=True)


g = sns.displot(df['price'])

plt.scatter(df['carat'],df['price'])

plt.boxplot(df['price'])

print()

cv = lambda x: np.std(x) / np.mean(x) * 100
res = df_numeric.apply(cv)
print(res)

print()

res = df_numeric.cumsum().head()
print(res)

print()

trimmed_mean = stats.trim_mean(df['price'],0.05)
print(trimmed_mean)

print()

des = df.describe()
print(df_numeric.max() - df_numeric.min())
plt.show()
