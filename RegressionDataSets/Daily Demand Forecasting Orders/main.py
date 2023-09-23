import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

dataset = pd.read_csv("Daily_Demand_Forecasting_Orders.csv", sep=";")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
#regressor_Lasso = Lasso(alpha=0.01)
#regressor_Ridge = Ridge(alpha=0.01)

regressor.fit(X_train, y_train)
#regressor_Lasso.fit(X_train, y_train)
#regressor_Ridge.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
#y_pred_Lasso = regressor_Lasso.fit(X_train, y_train)
#y_pred_Ridge = regressor_Ridge.fit(X_train, y_train)

cv_score = cross_val_score(regressor, X_train, y_train, cv=24)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(np.mean(cv_score))