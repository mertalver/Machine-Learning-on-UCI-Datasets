import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

dataSet = pd.read_csv("student-mat.csv", sep=";")
X = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, -1]

categorical_features = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities",
                        "nursery", "higher", "internet", "romantic"]

le = LabelEncoder()
for f in categorical_features:
    X[f] = le.fit_transform(X[f])

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(), [8, 9, 10, 11])], remainder="passthrough")
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=11, random_state=1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
