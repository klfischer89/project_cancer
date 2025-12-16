import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.svm import SVC

df = pd.read_csv("../data/Krebs/cancer.csv")
df.drop("id", axis = 1, inplace = True)
df["diagnosis"] = df["diagnosis"] == "M"

X = df[["concave points_worst", "perimeter_worst", "perimeter_mean", "radius_mean"]]

# X = df.drop("diagnosis", axis = 1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

model = GridSearchCV(SVC(), param_grid = {
    'kernel': ["rbf"],
    'gamma': [0.01, 0.1, 1, 10],
    "C": [0.01, 0.1, 0.5, 1, 5, 10]
}, cv = RepeatedKFold(), n_jobs = 15)

model.fit(X_train, y_train)

print(model.best_score_)