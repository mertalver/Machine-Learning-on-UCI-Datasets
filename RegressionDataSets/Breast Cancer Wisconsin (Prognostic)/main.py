import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

dataSet = pd.read_csv("wpbc.data", header=None, na_values="?")
X = dataSet.iloc[:, 2:].values
y = dataSet.iloc[:, 1].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X = imputer.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

cv_score = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(np.mean(cv_score))
