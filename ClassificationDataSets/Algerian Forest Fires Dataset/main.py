import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

dataset = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1]
y = y.str.strip()

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cv_score = cross_val_score(classifier, X_train, y_train, cv=10)

cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)

print(classification_report(y_test, y_pred))
print(cm)
print(accuracy_score(y_test, y_pred))
print(np.mean(cv_score))
disp.plot()
plt.show()
