import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

training_data = pd.read_csv("ticdata2000.txt", header=None, sep="\t")
testing_data = pd.read_csv("ticeval2000.txt", header=None, sep="\t")
predicting_data = pd.read_csv("tictgts2000.txt", header=None)

X_train = training_data.iloc[:, :-1].values
y_train = training_data.iloc[:, -1].values

X_test = testing_data.iloc[:, :].values
y_test = predicting_data.iloc[:, :].values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel="rbf", random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

params = {"C": [0, 0.25, 0.5, 0.75, 1],
          "kernel": ["linear", "poly", "rbf", "sigmoid"],
          "gamma": ["scale", "auto"]}

gridcv_model = GridSearchCV(estimator=classifier, param_grid=params, cv=10, scoring="accuracy", n_jobs=-1, verbose=2).fit(X_train, y_train)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(gridcv_model.best_params_)
print(gridcv_model.best_score_)
