import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn import metrics

carData = pd.read_csv('carData.csv')
# print(carData.isnull().sum())
# print(carData['Fuel_Type'].value_counts())
# print(carData['Seller_Type'].value_counts())
# print(carData['Transmission'].value_counts())

carData['Transmission'] = carData['Transmission'].replace({'Manual': 1, 'Automatic': 0})
carData['Seller_Type'] = carData['Seller_Type'].replace({'Dealer': 0, 'Individual': 1})
carData['Fuel_Type'] = carData['Fuel_Type'].replace({'Petrol': 0, 'Diesel': 1, 'CNG': 2})


# Splitting tha data into test and training
x = carData.drop(columns=['Selling_Price', 'Car_Name'],axis=1)
y = carData['Selling_Price']
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=2)

# Training Score: 0.8799451660493708
model1 = LinearRegression()
# model1.fit(x_train, y_train)

# Training Score: 0.9999896680984235
model2 = XGBRegressor()
model2.fit(x_train, y_train)

# Model Evaluation
# trainPredict = model1.predict(x_train)
trainPredict = model2.predict(x_train)
trainScore = metrics.r2_score(y_train, trainPredict)
print("Training Score:", trainScore)

# Visualisation of actual price and predicted price(XgbRegressor)
plt.scatter(y_train,trainPredict)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title("Actual Price Vs Predicted Prices")
plt.show()

# Predicting on test data
testPredict = model2.predict(x_test)
testScore = metrics.r2_score(y_test, testPredict)
print("Test Score:", testScore)

# Plotting actual vs predicted prices for test data
plt.scatter(y_test, testPredict)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title("Test Data: Actual vs Predicted Prices")
plt.show()
