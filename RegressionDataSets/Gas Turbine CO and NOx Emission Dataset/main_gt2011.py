import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

dataset = pd.read_csv("gt_2011.csv")
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)

regressor = MultiOutputRegressor(SVR(kernel="rbf"))
regressor.fit(X_train, y_train)

y_pred = sc_y.inverse_transform(regressor.predict(X_test))

cv_score = cross_val_score(regressor, X_train, y_train, cv=10)

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(np.mean(cv_score))