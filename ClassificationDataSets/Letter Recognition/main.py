import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

dataset = pd.read_csv("letter-recognition.data", header=None)
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cv_score = cross_val_score(classifier, X_train, y_train, cv=10)

confusion_matrix = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
display = ConfusionMatrixDisplay(confusion_matrix, display_labels=classifier.classes_)

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(figsize=(10, 10))
ax.set(xlabel="Predicted Label", ylabel="True Label", title="Confusion Matrix Display")
display.plot(ax=ax, cmap="jet")
plt.show()

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(np.mean(cv_score))
plt.show()
