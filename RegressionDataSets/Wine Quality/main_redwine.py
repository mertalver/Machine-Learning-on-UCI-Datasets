import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataSet = pd.read_csv("winequality-red.csv", delimiter=";")
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

regressor = RandomForestRegressor(n_estimators=15, random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
