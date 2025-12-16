import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.linear_model import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./data/cancer.csv")
df.drop("id", axis = 1, inplace = True)
df["diagnosis"] = df["diagnosis"] == "M"

X = df.drop("diagnosis", axis = 1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

p = StandardScaler()
p.fit(X_train, y_train)

model = GridSearchCV(KNeighborsClassifier(), param_grid = {
    'n_neighbors': [1, 3, 5, 9, 15, 25],
    'p': [1, 2]
}, cv = RepeatedKFold())

model.fit(p.transform(X_train), y_train)

print(model.best_score_)

print(model.score(p.transform(X_test), y_test))