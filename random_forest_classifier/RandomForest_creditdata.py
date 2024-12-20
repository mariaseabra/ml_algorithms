import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv("credit_data.csv")

features = credit_data[["income", "age", "loan"]]
targets = credit_data.default

X = np.array(features).reshape(-1, 3)
y = np.array(targets)

model = RandomForestClassifier()
predicted = cross_validate(model, X, y, cv=10) #using 10 folds

print(np.mean(predicted['test_score']))