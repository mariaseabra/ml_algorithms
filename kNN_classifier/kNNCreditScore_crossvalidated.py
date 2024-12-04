import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

#Using cross validation to find the optimal value for k (which will bring highest possible accuracy)

data = pd.read_csv("credit_data.csv")

features = data [["income", "age", "loan"]]
target = data.default

#transform dataset as machine learning handles arrays instead of data-frames
X = np.array(features).reshape(-1,3) #creates 3 columns (1 for each feature) and the # of rows will be figured out
y = np.array(target)

#preprocess and apply MinMax transformation on the feature values --> NORMALIZATION to make sure features are in the same range
X = preprocessing.MinMaxScaler().fit_transform(X)

# split dataset

feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=20)
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

cross_valid_scores = []

for k in range (1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') #cv = k, so in this case there will be 10 k
    cross_valid_scores.append(scores.mean())

print("Optimal k with cross-validation: ", np.argmax(cross_valid_scores)) #max value

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))

#Here accuracy is higher than in Logistic Regression