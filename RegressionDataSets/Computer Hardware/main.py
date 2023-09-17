import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

dataSet = pd.read_csv("machine.data", header=None)
X = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, -1].values

le = LabelEncoder()
for f in X.columns:
    if X[f].dtype != "int64":
        X[f] = le.fit_transform(X[f])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

poly_reg = PolynomialFeatures(degree=2)
X_train = poly_reg.fit_transform(X_train)
X_test = poly_reg.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(regressor.coef_)
print(regressor.intercept_)
