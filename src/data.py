from sklearn.datasets import fetch_20newsgroups

TARGET_CATEGORIES = ['rec.sport.baseball', 'rec.sport.hockey', 'sci.space']

def load_dataset():
    remove = ('headers', 'footers', 'quotes')
    train_data = fetch_20newsgroups(subset='train', categories=TARGET_CATEGORIES, remove=remove, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=TARGET_CATEGORIES, remove=remove, random_state=42)
    return train_data, test_data
