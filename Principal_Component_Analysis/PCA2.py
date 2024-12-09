from mlxtend.data import mnist_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

mnist_data = fetch_openml('mnist_784')

features = mnist_data.data
targets = mnist_data.target

train_img, test_img, train_lbl, test_lbl = train_test_split(features, targets, test_size=0.15, random_state=123)

scaler = StandardScaler()
scaler.fit(train_img)
train_img = scaler.transform(train_img)
test_img = scaler. transform(test_img)

print(train_img.shape)

#keeping 95% variance (95% of the original information)
pca = PCA(.95)
pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

print(train_img.shape)