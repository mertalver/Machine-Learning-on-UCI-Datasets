import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train_set = pd.read_csv("pendigits.tra", header=None)
test_set = pd.read_csv("pendigits.tes", header=None)

X_train = train_set.iloc[:, :-1].values
y_train = train_set.iloc[:, -1].values

X_test = test_set.iloc[:, :-1].values
y_test = test_set.iloc[:, -1].values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cv_score = cross_val_score(classifier, X_train, y_train, cv=10)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(np.mean(cv_score))