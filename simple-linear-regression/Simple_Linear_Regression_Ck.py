# Simple Linear Regression Salary Data Set

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing datasets

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values #This is matrix of independant variables
y = dataset.iloc[:,1].values  

#Splitting train and test set

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3 , random_state = 0) 

#Fitting Simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Prediction
y_pred = regressor.predict(X_test)

#Visualizing the Training set results 
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary') 
plt.show()

#Visualizing the Test set results
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_test,regressor.predict(X_test))
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary') 
plt.show()
