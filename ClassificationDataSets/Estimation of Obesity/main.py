import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataSet = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
X = dataSet.iloc[:, :-1]
y = dataSet.iloc[:, -1]

le = LabelEncoder()
for f in X.columns:
    if X[f].dtype == "object":
        X[f] = le.fit_transform(X[f])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=220, criterion="gini", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
