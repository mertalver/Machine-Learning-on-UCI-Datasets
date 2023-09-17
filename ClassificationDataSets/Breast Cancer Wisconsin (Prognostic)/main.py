import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataSet = pd.read_csv("wpbc.data", header=None, na_values="?")
X = dataSet.iloc[:, 2:].values
y = dataSet.iloc[:, 1].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X = imputer.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression(max_iter=1500, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cv_score = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

parameters = [{"C": [0.25, 0.5, 0.75, 1],
               "penalty": ["l2", None]}
              ]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search.fit(X_train, y_train)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred), "\n")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("10-Fold CV Accuracy Score:", np.mean(cv_score))
print("Best Accuracy Score:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)


