import string
import pandas as pd
from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def filter_by_ascii_rate(text, threshold=0.9):
    ascii_letters = set(string.printable)
    rate = sum(c in ascii_letters for c in text) / len(text)
    return rate <= threshold


def load_dataset(filename, n=5000, state=6):
    df = pd.read_csv(filename, sep='\t')

    # Converts multi-class to binary-class.
    mapping = {1: 0, 2: 0, 4: 1, 5: 1}
    df = df[df.star_rating != 3]
    df.star_rating = df.star_rating.map(mapping)

    # extracts Japanese texts.
    is_jp = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_jp]

    # sampling.
    df = df.sample(frac=1, random_state=state)  # shuffle
    grouped = df.groupby('star_rating')
    df = grouped.head(n=n)
    return df.review_body.values, df.star_rating.values

t = Tokenizer(wakati=True)
def tokenize(text):
    return t.tokenize(text)


def clean_html(html, strip=False):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(strip=strip)
    return text

def train_and_eval(x_train, y_train, x_test, y_test, vectorizer):
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_vec, y_train)
    y_pred = clf.predict(x_test_vec)
    score = accuracy_score(y_test, y_pred)
    print('{:.4f}'.format(score))
    
    
def main():
    x,y = load_dataset("D:/baikabai/data_set/amazon/amazon_reviews_multilingual_JP_v1_00.tsv",n=5000)
    x = [clean_html(text, strip=True) for text in x]
    print('Tokenization for faster experiments')
    x_tokenized = [' '.join(tokenize(text)) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x_tokenized, y, test_size=0.2, random_state=42)
    print('Binary')
    vectorizer = CountVectorizer(binary=True)
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)
    print('Count')
    vectorizer = CountVectorizer(binary=False)
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    print('TF-IDF')
    vectorizer = TfidfVectorizer()
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    print('Bigram')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)
    
    print('Vectorizing...')
    vectorizer = CountVectorizer(tokenizer=tokenize)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    print(x_train.shape)
    print(x_test.shape)

    print('Selecting features...')
    selector = SelectKBest(k=7000, score_func=mutual_info_classif)
    # selector = SelectKBest(k=7000)
    selector.fit(x_train, y_train)
    x_train_new = selector.transform(x_train)
    x_test_new = selector.transform(x_test)
    print(x_train_new.shape)
    print(x_test_new.shape)

    print('Evaluating...')
    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_new, y_train)
    y_pred = clf.predict(x_test_new)
    score = accuracy_score(y_test, y_pred)
    print('{:.4f}'.format(score))


if __name__ == '__main__':
    main()
    