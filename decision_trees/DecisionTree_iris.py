import numpy as np
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import datasets

iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

model = DecisionTreeClassifier(criterion='entropy')

predicted = cross_validate(model, features, targets, cv=10)
print(np.mean(predicted['test_score']))

# Issue: every split (cv) it makes at each node is optimised for the dataset it is fit to
# this splitting process will rarely generalise well to other data, which leads to overfitting
# Because of that, parameter tuning is indicated