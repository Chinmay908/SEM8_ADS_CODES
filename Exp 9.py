import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from bioinfokit.analys import stat

df = pd.read_csv("diamonds.csv")

df_mean = df['depth'].mean()
df_std = df['depth'].std()

pdf = stats.norm.pdf(df['depth'].sort_values(),df_mean,df_std)

plt.plot(df['depth'].sort_values(),pdf)
plt.xlabel("depth",size=13)
plt.ylabel("frequency",size=13)
plt.show()

pmf = stats.poisson.pmf(df['depth'],mu=df_mean)
plt.plot(df['depth'],pmf)
plt.xlabel("depth",size=13)
plt.ylabel("probablity",size=13)
plt.show()

#Z-test (1 sample)
print('Null Hypothesis: mu equal to mu0')
pop_mean = df['depth'].mean()

print("Pop Mean: ",pop_mean)

pop_std = df['depth'].std()

sample = df[['depth']].sample(n=29)

res = stat()
res.ztest(df=sample,x='depth',mu=pop_mean,x_std=pop_std,alpha=0.05,test_type=1)
print(res.summary)
print(res.result)

#t-test, type-I, type-II error (2 sample)
print('Null Hypothesis: mu1 equal to mu2')
sample_x = df['x'].sample(50)
sample_y = df['y'].sample(50)
res = stats.ttest_ind(a=sample_x,b=sample_y,equal_var=True)

print(res)

#ANOVA (more than 2 sample)
print("H0: μ1 = μ2 = μ3")
print("H1: The means are not equal")

s1 = df['x'].sample(50)
s2 = df['y'].sample(50)
s3 = df['z'].sample(50)


f,p = stats.f_oneway(s1,s2,s3)
print(f,p)

if(p<0.05):
    print('Reject H0')
else:
    print('Accept H0')
