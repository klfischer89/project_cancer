import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("../data/Krebs/cancer.csv")
df.drop("id", axis = 1, inplace = True)
df["diagnosis"] = df["diagnosis"] == "M"

X = df[["concave points_worst", "perimeter_worst", "perimeter_mean", "radius_mean"]]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

model = GridSearchCV(LogisticRegression(), param_grid = {
    'max_iter': [10000]
}, cv = RepeatedKFold())

model.fit(X_train, y_train)

print(model.best_score_)

print(model.score(X_test, y_test))