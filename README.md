# finalproject
This project use a machine learning algorithm (linear regression) to expect boston house prices
-----------------------------------------------------------------------------------------------
Import dependencies (the machine learning library sklearn, numpy, and pandas)
Load the Boston Housing Data Set from the scikit-learn library using load_boston() function and print it
Print the value of the dictionary (BostonDataset) to understand what it contains:
-data: contains the information for various houses
-target: prices of the house
-feature_names: names of the features
-DESCR: describes the dataset
-filename: the physical location of Boston CSV data set
Print the description of all the features BostonDataset.DESCR
Transform the data set into a data frame using the pandas library and print the first 5 rows of the data using head() 
Create a new column of target values and add it to the dataframe, target value: MEDV
Count the number of missing values for each feature using isnull() ... the data set is clean meaning there is no corrupt, inaccurate, or missing data
Exploratory Data Analysis: some visualizations to understand the relationship of the target variable with other features
First plot the distribution of the target variable MEDV using the distplot function from the seaborn library ... the values of MEDV are distributed normally 
Second create a correlation matrix that measures the linear relationships between the variables, The correlation matrix can be formed from the pandas dataframe library, and plot from the seaborn library
Then scatter plot, in linear regression model have to select features which have a high correlation with our target variable MEDV, RM has a strong positive correlation with MEDV (0.7) where as LSTAT has a high negative correlation with MEDV(-0.74) 
when training the model, can't choose feature pairs are strongly correlated to each other (it is called check for multi-co-linearity) like the features RAD, TAX have a correlation of 0.91 ...
The prices increase as the value of RM(Average number of rooms per dwelling) increases linearly
The prices tend to decrease with an increase in LSTAT(Percentage of lower status of the population), it doesn’t look to be following exactly a linear line.
Training the model 
Transform columns LSTAT, RM (have a high correlation with our target variable MEDV) into a X data frame and MEDV to a Y data frame
Get some statistics from the X data set like the count or the number of rows, the data contains for each column, the minimum value for each column, the maximum value for each column, and the mean for each column.
Then, initialize the Linear Regression model, split the data using train_test_split function into 80% training and 20% testing data to help us assess the model’s performance on unseen data
After that, train the model with the training data set that contains the independent variables by Inilializing the Linear Regression model and print the estimated coefecients for each column of our model
Training the Linear Regression model using coefficients that describe the linear function and print the model's predictions
Check the model's performance/accuracy using a metric called mean squared error (MSE) in two different ways, one using numpy and the other using sklearn.metrics ... the model had predicted the exact value as the actual values which is good
Check r2_score (performance / accuracy ) for the modeles to compare:
-Linear Regression model r2_score = 0.6628996975186952 
-DecisionTreeRegressor model r2_score = 0.5853630693055365
-RandomForestRegressor model r2_score = 0.7710084613587442
-RandomForestRegressor with n_estimators=100 r2_score = 0.778327124123161
-neural network r2_score = 0.7817100025274083
Notice the Neural Network score the highest accuraacy


References:
-----------
https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
https://scikit-learn.org/stable/modules/neural_networks_supervised.html#regression
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score