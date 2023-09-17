import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

dataSet = pd.read_csv("ENB2012_data.csv", sep=";", decimal=",", skip_blank_lines=True).dropna()
X = dataSet.iloc[:, :-2].values
y = dataSet.iloc[:, -2:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = XGBRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

cv_score = cross_val_score(regressor, X_train, y_train, cv=10)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(np.mean(cv_score))