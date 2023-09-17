import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

dataSet = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",")
dataSet = dataSet.drop(dataSet.columns[[0, 1, 15, 16]], axis=1)
dataSet = dataSet.dropna(subset="CO(GT)")
X = dataSet.iloc[:, 1:].values
y = dataSet.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressor = RandomForestRegressor(n_estimators=42, random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
