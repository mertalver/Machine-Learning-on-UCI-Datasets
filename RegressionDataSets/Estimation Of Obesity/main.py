import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataSet = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
X = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, -1]

le = LabelEncoder()
for f in X.columns:
    if X[f].dtype == "object":
        X[f] = le.fit_transform(X[f])
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(n_estimators=145, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
