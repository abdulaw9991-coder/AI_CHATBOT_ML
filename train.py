import json
import random
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle
import datetime

nltk.download('punkt')

# ---------------- LOAD DATA ----------------
data = json.loads(open("intents.json").read())

texts = []
labels = []

# ---------------- PREPROCESS ----------------
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# ---------------- TEXT TO NUMBERS ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# ---------------- LABEL ENCODING ----------------
le = LabelEncoder()
y = le.fit_transform(labels)

# ---------------- TRAIN MODEL ----------------
model = MultinomialNB()
model.fit(X, y)

# ---------------- SAVE MODEL ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("✅ Model trained successfully!")