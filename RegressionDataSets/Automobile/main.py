import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

dataSet = pd.read_csv("imports-85.data", na_values="?")
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values
y = y.reshape(-1, 1)

le = LabelEncoder()
for col in range(X.shape[1]):
    if isinstance(X[0, col], str):
        X[:, col] = le.fit_transform(X[:, col])

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X = imputer.fit_transform(X)
y = imputer.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
