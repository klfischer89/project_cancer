import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("./data/cancer.csv")
df.drop("id", axis = 1, inplace = True)
df["diagnosis"] = df["diagnosis"] == "M"

X = df.drop("diagnosis", axis = 1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

model = GridSearchCV(RandomForestClassifier(), param_grid = {
    'max_depth': list(range(1, 21, 5)),
    "min_samples_split": [1, 5, 10],
    'n_estimators': [1, 10, 100]
}, cv = RepeatedKFold(), n_jobs = 4)

model.fit(X_train, y_train)

print(model.best_score_)

print(model.best_params_)