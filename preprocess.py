import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
import re

def clean_data(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r'\\n'," ",text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r":","",text)
    text = re.sub(r"-","",text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(" n "," ",text)
    text = re.sub(r".com","",text)
    text = re.sub(r"www","",text)


    
    ## Stemming
    text = text.split(' ')
    stemmer = nltk.SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]


    text = " ".join(stemmed_words)
    
    return text



data_train = pd.read_csv('train.csv')
print(data_train.head())
print(data_train.shape)
data_train['Text'] = data_train['Text'].map(lambda x : clean_data(x))
data_train.to_csv('train_clean.csv')

data_test = pd.read_csv('test.csv')
print(data_test.head())
print(data_test.shape)
data_test['Text'] = data_test['Text'].map(lambda x : clean_data(x))
data_test.to_csv('train_clean.csv')