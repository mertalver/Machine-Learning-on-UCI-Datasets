import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataSet = pd.read_csv("forestfires.csv")
X = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, -1].values

le = LabelEncoder()
for f in X.columns:
    if X[f].dtype == "object":
        X[f] = le.fit_transform(X[f])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))