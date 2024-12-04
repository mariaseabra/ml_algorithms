import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Load the manually mapped datasets
design_within_reach_mapped = pd.read_csv('Design Within Reach.csv')
discount_school_supply_mapped = pd.read_csv('Discount_School_Supply.csv')

# Combine the datasets for training
combined_data = pd.concat([design_within_reach_mapped, discount_school_supply_mapped])

# Preprocessing function to clean text data
def preprocess_text(text):
    if isinstance(text, str):
        # Lowercase the text
        text = text.lower()
        # Remove special characters
        text = ''.join(e for e in text if e.isalnum() or e.isspace())
        return text
    else:
        return ''

# Apply preprocessing
combined_data['src_pt'] = combined_data['src_pt'].apply(preprocess_text)
combined_data['src_cat'] = combined_data['src_cat'].apply(preprocess_text)
combined_data['src_sc'] = combined_data['src_sc'].apply(preprocess_text)

# Combine the text features
combined_data['combined'] = combined_data['src_pt'] + ' ' + combined_data['src_cat'] + ' ' + combined_data['src_sc']

# Encode target labels (productType, category, subCategory)
le_pt = LabelEncoder()
le_cat = LabelEncoder()
le_sc = LabelEncoder()

combined_data['ent_pt_2_encoded'] = le_pt.fit_transform(combined_data['ent_pt_2'])
combined_data['ent_cat_2_encoded'] = le_cat.fit_transform(combined_data['ent_cat_2'])
combined_data['ent_sc_2_encoded'] = le_sc.fit_transform(combined_data['ent_sc_2'])

 #Split the data into training and testing sets
X_train, X_test, y_train_pt, y_test_pt, y_train_cat, y_test_cat, y_train_sc, y_test_sc = train_test_split(
    combined_data['combined'],
    combined_data['ent_pt_2_encoded'],
    combined_data['ent_cat_2_encoded'],
    combined_data['ent_sc_2_encoded'],
    test_size=0.2,
    random_state=42
)

# Create a pipeline for text processing and classification
pipeline_pt = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_cat = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_sc = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the models
pipeline_pt.fit(X_train, y_train_pt)
pipeline_cat.fit(X_train, y_train_cat)
pipeline_sc.fit(X_train, y_train_sc)