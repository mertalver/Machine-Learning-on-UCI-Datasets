import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataSet = pd.read_csv("adult.data", header=None, na_values=" ?")
X = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, -1].values

X = X.fillna(X.mode().iloc[0])

le = LabelEncoder()
for f in X.columns:
    if X[f].dtype != "int64":
        X[f] = le.fit_transform(X[f])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(criterion="gini", random_state=1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
