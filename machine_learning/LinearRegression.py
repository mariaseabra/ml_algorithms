import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#read .csv into a DataFrame

house_data = pd.read_csv("house_prices.csv")
size = house_data['sqft_living']
price = house_data['price']

# machine learning handles arrays, not dataframes (get rid of indexes)
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

# Linear Regression + fit() to train the model - find a linear relation between size and prices
model = LinearRegression()
model.fit(x, y)

# MSE and R value - evaluating the function

regression_model_use = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_use))
print ("R squared value: ", model.score(x, y))

# Geting b values after model fit
# b1 - defines the linear regression's slope:
print(model.coef_[0])
# b0 - where the linear regression model intersects with the y axis
print(model.intercept_[0])

# Visualising the dataset with the fitted modeç
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

# Predicting the prices
print("Prediction by the model: ", model.predict([[200]]))