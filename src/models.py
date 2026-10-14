from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_baseline_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

def build_ngram_pipeline():
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=15000)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
