import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

dataSet = pd.read_csv("SeoulBikeData.csv", encoding="ISO-8859-1")
X = dataSet.iloc[:, 2:]
y = dataSet.iloc[:, 1].values

le = LabelEncoder()
for f in X.columns:
    if X[f].dtype == "object":
        X[f] = le.fit_transform(X[f])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

cv_score = cross_val_score(regressor, X_train, y_train, cv=10)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(np.mean(cv_score))

