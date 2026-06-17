import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("data/Personal_Finance_Dataset.csv")

# Rename columns
df.rename(columns={
    "Transaction Description": "description",
    "Category": "category"
}, inplace=True)

# Remove missing values
df = df.dropna(subset=["description", "category"])

X = df["description"]
y = df["category"]

# Train model
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

pipeline.fit(X, y)

# Save model
pickle.dump(pipeline, open("model/category_model.pkl", "wb"))

print("Model trained successfully!")