import numpy as np
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

# with grid search we can fing an optimal parameter --> "parameter tuning"
param_grid = {'max_depth': np.arange(1, 10)} # using a sequence of 1 to 10 exclusively (1-9 one-dimensional array) as possible values for the max_depth parameter

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

#in every interation we split the data randomly in cross validation + DecisionTreeClassifier
#initialises the tree randomly: that's why you get different results

tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

tree.fit(feature_train, target_train)

print("Best parameter with Grid Search: ", tree.best_params_)

grid_predictions = tree.predict(feature_test)
print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))
