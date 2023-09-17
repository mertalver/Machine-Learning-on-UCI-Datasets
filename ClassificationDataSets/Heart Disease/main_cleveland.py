import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataSet = pd.read_csv("processed.cleveland.data", na_values="?")
X = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, -1].values
y = y.reshape(-1, 1)

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X = imputer.fit_transform(X)
y = imputer.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=11, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
