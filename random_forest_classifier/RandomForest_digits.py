from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # to find optimal parameters for RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

digit_data = datasets.load_digits()

image_features = digit_data.images.reshape((len(digit_data.images), -1))
image_targets = digit_data.target

random_forest_model = RandomForestClassifier(n_jobs=-1, max_features='sqrt') #n_jobs indicates the number of processors for parallel computing (-1 all possible processors)

feature_train, feature_test, target_train, target_test = train_test_split(image_features, image_targets, test_size=0.3)

#dictionary of possible values for parameters to be tuned
param_grid = {
    "n_estimators": [10, 100, 500, 1000], # number of trees in the RandomForest
    "max_depth": [1, 5, 10, 15], #max depth of the tree
    "min_samples_leaf": [1, 2, 4, 10, 15, 30, 50] #min num of samples required to be a leaf node
}

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=10)
grid_search.fit(feature_train, target_train)
print(grid_search.best_params_)

optimal_estimators = grid_search.best_params_.get("n_estimators")
optimal_depth = grid_search.best_params_.get("max_depth")
optimal_leaf = grid_search.best_params_.get("min_samples_leaf")

print("Optimal n_estimators: %s" % optimal_estimators)
print("Optimal depth: %s" % optimal_depth)
print("Optimal leaf: %s" % optimal_leaf)

grid_predictions = grid_search.predict(feature_test)
print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))