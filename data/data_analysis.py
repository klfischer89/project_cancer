import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv("../data/Krebs/cancer.csv")
df.drop("id", axis = 1, inplace = True)
df["diagnosis"] = df["diagnosis"] == "M"

sns.heatmap(df.corr())

np.mean(df["diagnosis"])