import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

training_data = pd.read_csv("ticdata2000.txt", header=None, sep="\t")
testing_data = pd.read_csv("ticeval2000.txt", header=None, sep="\t")
predicting_data = pd.read_csv("tictgts2000.txt", header=None)

X_train = training_data.iloc[:, :-1].values
y_train = training_data.iloc[:, -1].values

X_test = testing_data.iloc[:, :].values
y_test = predicting_data.iloc[:, :].values

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

cv_score = cross_val_score(regressor, X_train, y_train, cv=10)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(np.mean(cv_score))