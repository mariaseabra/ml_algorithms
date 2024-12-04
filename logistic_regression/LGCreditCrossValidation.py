import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv("credit_data.csv")

features = credit_data[["income", "age", "loan"]] #array with subarray
target = credit_data.default #label

# machine learning handles arrays not data-frames - remove index

X = np.array(features).reshape((-1,3)) #-1 means that python will figure out the number of rows and 3 is the number of columns
y = np.array(target)

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5) # cv=k

print(predicted['test_score']) #accuracy per iteration
print(np.mean(predicted['test_score']))