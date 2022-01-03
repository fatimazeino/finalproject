# bostonprice.py
# This program  foresee Boston House Prices
# Author: Fatima Zeino

#import dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#Load the Boston Housing Data Set from sklearn.datasets and print it
from sklearn.datasets import load_boston
boston = load_boston()
#print(boston)

#Transform the data set into a data frame 
df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)

import matplotlib.pyplot as plt
df_x.plot(kind = 'hist')
plt.show() 

#Get some statistics from our data set, count, mean ..
df_x.describe()

#Initialize the linear regression model
reg = linear_model.LinearRegression()

#Split the data into 67% training and 33% testing data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)

#Train our model with the training data
reg.fit(x_train, y_train)
print(reg.coef_)

#print our price predictions on our test data
y_pred = reg.predict(x_test)
print(y_pred)

#print the actual values
print(y_test)

#check the model performance / accuracy using mean squared error (MSE)
print(np.mean((y_pred-y_test)**2))

#check the model performance / accuracy using mean squared error (MSE) & sklearn.metrics 
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))