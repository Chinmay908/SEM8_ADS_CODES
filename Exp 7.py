import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN

df = pd.read_csv("Churn_Modelling.csv")

sns.displot(df['Balance'])
plt.xlabel('Balance')
plt.ylabel('Density')
plt.show()

plt.boxplot(df['CreditScore'])
plt.xlabel('CreditScore')
plt.show()

plt.scatter(df['CreditScore'],df['Age'])
plt.xlabel('Age')
plt.ylabel('Balance')
plt.show()

#pie-chart
data = df['Tenure'].value_counts()
data = data.reset_index(name='Count')
arr = data['Count'].to_numpy()
ten = data['Tenure'].to_numpy()
# print(ten)
plt.pie(arr,labels=ten,autopct="%1.2f%%")
plt.legend()
plt.show()

#outlier detection - dist based
nbrs = NearestNeighbors(n_neighbors=3)

X = df[['CreditScore','Age']].values

nbrs.fit(X)
dist,indexes = nbrs.kneighbors(X)

plt.plot(dist.mean(axis=1))
plt.show()

outlier_index = np.where(dist.mean(axis = 1) > 8)
print(outlier_index)

outlier_values = df[['CreditScore','Age']].iloc[outlier_index]
print(outlier_values)

# # plot data
plt.scatter(df['CreditScore'], df['Age'],color="b")
plt.xlabel('Credit Score')
plt.ylabel('Age')

# plot outlier values
plt.scatter(outlier_values["CreditScore"], outlier_values["Age"], color = "r")
plt.show()

#oulier detection using LOF - density
lof = LocalOutlierFactor(n_neighbors=3,contamination=0.001)
X = df[['CreditScore','Age']].values
preds = lof.fit_predict(X)


# Identify outliers
outliers = np.where(preds == -1)[0]
print(outliers)

outlier_values = df[['CreditScore','Age']].iloc[outliers]
print(outlier_values)

# # plot data
plt.scatter(df['CreditScore'], df['Age'],color="b")
plt.xlabel('Credit Score')
plt.ylabel('Age')

# # plot outlier values
plt.scatter(outlier_values["CreditScore"], outlier_values["Age"], color = "r")
plt.show()

#oulier detection using DBSCAN - density
dbscan = DBSCAN(eps=0.05,min_samples=10)
dbscan.fit(df[['CreditScore','Age']])

colors = dbscan.labels_
plt.scatter(df["CreditScore"], df["Age"],c=colors)
plt.show()

#trimming
upper_limit = df['CreditScore'].quantile(0.99)
lower_limit = df['CreditScore'].quantile(0.01)
print(lower_limit,upper_limit)

new_df = df[(df['CreditScore'] <= upper_limit) & (df['CreditScore'] >= lower_limit)]
# print(new_df.head())
sns.boxplot(new_df['CreditScore'])
plt.show()

#winsorization (capping)
df['CreditScore'] = np.where(df['CreditScore'] >= upper_limit,upper_limit,np.where(df['CreditScore'] <= lower_limit, lower_limit, df['CreditScore']))
sns.boxplot(df['CreditScore'])
plt.show()

