import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataSet = pd.read_csv("car.data")
X = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, -1].values

le_x = LabelEncoder()
le_y = LabelEncoder()
y = le_y.fit_transform(y)
for f in X.columns:
    if X[f].dtype != "int64":
        X[f] = le_x.fit_transform(X[f])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = DecisionTreeClassifier(criterion="entropy", random_state=1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
