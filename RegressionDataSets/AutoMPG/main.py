import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

dataSet = pd.read_csv("auto-mpg.data-original", sep="\s+", na_values="NA")
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values
y = y.reshape(-1, 1)

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X = imputer.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=19, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
print(math.sqrt(mean_squared_error(y_test, y_pred)))
