from scipy.stats import multinomial
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.python.keras.utils.version_utils import training

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=123)

# print("\n".join(training_data.data[1].split("\n")[:30])) #checking first 30 lines of second dataset from the archive
# print("Target is:", training_data.target_names[training_data.target[0]])

# counting word occurrences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
print(count_vector.vocabulary_)

# transforming word occurrences into tf-idf
# TfidfVectorizer = CountVectorizer + TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts) # transforms countvectorizer into tfidf vectorizer

print(x_train_tfidf)

model = MultinomialNB().fit(x_train_tfidf, training_data.target)

new = ['My favourite topic has something to do with quantum physics and quantum mechanics',
       'This has nothing to do with church or religion',
       'Software engineering is getting hotter and hotter nowadays']

x_new_counts = count_vector.transform(new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf)

print(predicted)

for doc, category in zip(new, predicted):
    print('%r --------> %s' % (doc, training_data.target_names[category]))
