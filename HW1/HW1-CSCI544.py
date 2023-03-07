import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def remove_HTML(s):
    return re.sub(r'<.*?>',' ',s)

def remove_URL(s):
    return re.sub(r'https?:\/\/.*\/\w*',' ',s)

def remove_nonalphabets(s):
    return re.sub(r'[^a-zA-Z]',' ',s)

def remove_multispace(s):
    return re.sub(r'\s+',' ',s)

data = pd.read_csv('amazon_reviews_us_Beauty_v1_00.tsv', sep='\t', encoding='utf-8', on_bad_lines='skip')
data['star_rating'] = pd.to_numeric(data['star_rating'],errors='coerce') #results in NaNs
df = data[['review_body','star_rating']] #dropping other columns
df = df.dropna(thresh=2)

class_a = df.loc[df['star_rating'].isin([1,2])].sample(n=20000, random_state=1)
class_b = df.loc[df['star_rating'].isin([3])].sample(n=20000,random_state=2)
class_c = df.loc[df['star_rating'].isin([4,5])].sample(n=20000,random_state=3)

df_sampled = pd.concat([class_a, class_b, class_c])
df_sampled['star_rating'] = df_sampled['star_rating'].replace([1,2],"A")
df_sampled['star_rating'] = df_sampled['star_rating'].replace([3],"B")
df_sampled['star_rating'] = df_sampled['star_rating'].replace([4,5],"C")

before_clean = np.mean(df_sampled['review_body'].apply(lambda x: len(str(x))))

df_sampled['review_body'] = df_sampled['review_body'].apply(lambda x: str(x).lower())
df_sampled['review_body'] = df_sampled['review_body'].apply(lambda x:remove_HTML(x))
df_sampled['review_body'] = df_sampled['review_body'].apply(lambda x:remove_URL(x))
df_sampled['review_body'] = df_sampled['review_body'].apply(lambda x:contractions.fix(x))
df_sampled['review_body'] = df_sampled['review_body'].apply(lambda x:remove_nonalphabets(x))
df_sampled['review_body'] = df_sampled['review_body'].apply(lambda x:remove_multispace(x))

after_clean = np.mean(df_sampled['review_body'].apply(lambda x: len(x)))
print(str(before_clean)+", "+str(after_clean))

df_sampled['tokenized'] = df_sampled['review_body'].apply(lambda x: nltk.word_tokenize(x))

stemmer = PorterStemmer()
df_sampled['stemmed'] = df_sampled['tokenized'].apply(lambda x: [stemmer.stem(w) for w in x])
df_sampled['stemmed'] = df_sampled['stemmed'].apply(lambda x: ' '.join(x))

after_preprocessing = np.mean(df_sampled['stemmed'].apply(lambda x : len(x)))
print(str(after_clean)+", "+str(after_preprocessing))

stemVectorizer = TfidfVectorizer()
stemtfidf = stemVectorizer.fit_transform(df_sampled['stemmed'])

X_train, X_test, y_train, y_test  = train_test_split(stemtfidf, 
                            df_sampled['star_rating'],
                            stratify=df_sampled['star_rating'],
                            test_size=0.2, random_state=1)

p = Perceptron(random_state=7)
p.fit(X_train, y_train)

scores = classification_report(y_test, p.predict(X_test),output_dict=True)
print(str(scores['A']['precision'])+", "+str(scores['A']['recall'])+", "+str(scores['A']['f1-score']))
print(str(scores['B']['precision'])+", "+str(scores['B']['recall'])+", "+str(scores['B']['f1-score']))
print(str(scores['C']['precision'])+", "+str(scores['C']['recall'])+", "+str(scores['C']['f1-score']))
print(str(scores['weighted avg']['precision'])+", "+str(scores['weighted avg']['recall'])+", "+str(scores['weighted avg']['f1-score']))

s = LinearSVC(random_state=7, tol= 1e-5)
s.fit(X_train, y_train)
scores = classification_report(y_test, s.predict(X_test),output_dict=True)
print(str(scores['A']['precision'])+", "+str(scores['A']['recall'])+", "+str(scores['A']['f1-score']))
print(str(scores['B']['precision'])+", "+str(scores['B']['recall'])+", "+str(scores['B']['f1-score']))
print(str(scores['C']['precision'])+", "+str(scores['C']['recall'])+", "+str(scores['C']['f1-score']))
print(str(scores['weighted avg']['precision'])+", "+str(scores['weighted avg']['recall'])+", "+str(scores['weighted avg']['f1-score']))

lr = LogisticRegression(random_state=7, max_iter=300)
lr.fit(X_train, y_train)

scores = classification_report(y_test, lr.predict(X_test),output_dict=True)
print(str(scores['A']['precision'])+", "+str(scores['A']['recall'])+", "+str(scores['A']['f1-score']))
print(str(scores['B']['precision'])+", "+str(scores['B']['recall'])+", "+str(scores['B']['f1-score']))
print(str(scores['C']['precision'])+", "+str(scores['C']['recall'])+", "+str(scores['C']['f1-score']))
print(str(scores['weighted avg']['precision'])+", "+str(scores['weighted avg']['recall'])+", "+str(scores['weighted avg']['f1-score']))
nb = MultinomialNB()
nb.fit(X_train, y_train)

scores = classification_report(y_test, nb.predict(X_test),output_dict=True)
print(str(scores['A']['precision'])+", "+str(scores['A']['recall'])+", "+str(scores['A']['f1-score']))
print(str(scores['B']['precision'])+", "+str(scores['B']['recall'])+", "+str(scores['B']['f1-score']))
print(str(scores['C']['precision'])+", "+str(scores['C']['recall'])+", "+str(scores['C']['f1-score']))
print(str(scores['weighted avg']['precision'])+", "+str(scores['weighted avg']['recall'])+", "+str(scores['weighted avg']['f1-score']))