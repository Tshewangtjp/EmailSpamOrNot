import pandas as pd
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode labels
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Remove duplicates
df.drop_duplicates(inplace=True)

# -----------------------------
# Text Preprocessing
# -----------------------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()

    words = word_tokenize(text)

    y = []

    for word in words:
        if word.isalnum():
            y.append(word)

    words = y[:]
    y.clear()

    for word in words:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    words = y[:]
    y.clear()

    for word in words:
        y.append(ps.stem(word))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

# -----------------------------
# TF-IDF
# -----------------------------
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()

y = df['target']

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))

# -----------------------------
# Save Model
# -----------------------------
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("Model Saved Successfully!")