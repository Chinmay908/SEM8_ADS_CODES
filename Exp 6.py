import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.arima.model as ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("supermarket_sales.csv")

print(df.info())

#Additive Decompostion
add_result = seasonal_decompose(df['Unit price'],model="additive",period=1)
add_result.plot()
plt.show()

#ACF
fig, ax = plt.subplots(figsize=(12,5))
plot_acf(df['Quantity'])
plt.show()

#PACF
fig, ax = plt.subplots(figsize=(12,5))
plot_pacf(df['Quantity'])
plt.show()

#ADF
adf_result = adfuller(df['Unit price'],autolag="AIC")
print(adf_result)

#Linear Regression
x = df['Quantity']
y = df['Total']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

x_train = x_train.values
x_train = x_train.reshape(-1,1)

x_test = x_test.values
x_test = x_test.reshape(-1,1)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print()
# print(y_pred)

#moving avg
moving_avg = df['Quantity'].rolling(window=3).mean()
print(moving_avg)

#ARIMA
train_size = int(len(df) * 0.8)
train_data,test_data = df[:train_size], df[train_size:]

p=1
d=1
q=1


arima_model = ARIMA.ARIMA(train_data['Unit price'],order=(p,d,q))

arima_fit = arima_model.fit()

predictions = arima_fit.forecast(steps=len(test_data)).values

plt.plot(train_data.index, train_data['Unit price'], label='Train Data')
plt.plot(test_data.index, test_data['Unit price'], label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.legend()
plt.show()


