import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataSet = pd.read_csv("Folds5x2_pp.csv", sep=";", decimal=",")
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer="adam", loss="mean_squared_error", metrics=['mean_squared_error', 'mean_absolute_error'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)

y_pred = ann.predict(X_test)

print(r2_score(y_test, y_pred))
