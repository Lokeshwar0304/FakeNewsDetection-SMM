# Load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import nltk #Import NLTK ---> Natural Language Toolkit
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re,ftfy,string
from sklearn.metrics import f1_score

train = pd.read_csv('train.csv')

hashtag_re = re.compile(r"#\w+")
mention_re = re.compile(r"@\w+")
url_re = re.compile(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*")
extras_re = re.compile("[.;:!\'?,\"()\[\]]")
#apos_re = "\'[a-z]*"
#leftover_re = "\S+"

""" Preprocessing the text in the statements"""
def clean(text):
    p_text = hashtag_re.sub("[hashtag]",text)
    p_text = mention_re.sub("[mention]",p_text)
    p_text = extras_re.sub("",p_text)
    p_text = url_re.sub("[url]",p_text)
    p_text = ftfy.fix_text(p_text)
    p_text = p_text.translate(str.maketrans("", "", string.punctuation))
    return p_text.lower()


# create a function to tokenize the data
def preprocess_data(data):
  
  # 1. Tokenization
  tk = RegexpTokenizer('\s+', gaps = True)
  text_data = [] # List for storing the tokenized data
  for values in data.Claim:
    value = clean(values)
    tokenized_data = tk.tokenize(value) # Tokenize the news
    text_data.append(tokenized_data) # append the tokenized data

  # 2. Stopword Removal

  # Extract the stopwords
  sw = stopwords.words('english')
  clean_data = [] # List for storing the clean text
  # Remove the stopwords using stopwords
  for data in text_data:
    clean_text = [words.lower() for words in data if words.lower() not in sw]
    clean_data.append(clean_text) # Appned the clean_text in the clean_data list

    # 3. Stemming

  # Create a stemmer object
  ps = PorterStemmer()
  stemmed_data = [] # List for storing the stemmed data
  for data in clean_data:
    stemmed_text = [ps.stem(words) for words in data] # Stem the words
    stemmed_data.append(stemmed_text) # Append the stemmed text
    
  updated_data = []
  for data in stemmed_data:
    updated_data.append(" ".join(data))

  # TFID Vector object
  tfidf = TfidfVectorizer()
  tfidf_matrix = tfidf.fit_transform(updated_data)

  return tfidf_matrix

train['preproc'] = train['Claim'].apply(preprocess_data)

preprocessed_data = preprocess_data(train)

test = pd.read_csv('test.csv')
preprocessed_data_test = preprocess_data(test)


# Model selection
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, train.Label, test_size=0.1, random_state = 42, stratify = train.Label)

# Metrics
from sklearn.metrics import accuracy_score

# model
def compute_metrics(data, y_true, model_obj, model):

  # Make predictions
  y_pred = model_obj.predict(data)

  # Compute accuracy
  acc = accuracy_score(y_true = y_true, y_pred = y_pred)
  
  f1 = f1_score(y_true, y_pred, average='weighted')

  # Make DataFrame
  metrics = pd.DataFrame(data = {'Accuracy Score': np.array([acc]), 'F1 Score':np.array([f1]) }, index=[model])
  return metrics

# 1. LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
# Model object
lr_reg = LogisticRegressionCV(Cs=20, cv=3, random_state=42)
# fit the model
lr_reg.fit(X_train, y_train)
lr_metrics =  compute_metrics(X_test, y_test, lr_reg, 'LogisticRegression')


#2. Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Model Object
mnb = MultinomialNB(alpha=0.0)
# Fit the object
mnb.fit(X_train, y_train)
mnb_metrics = compute_metrics(X_test, y_test, mnb, 'Naive Bayes')
mnb_metrics

# 3. DecisionTree
from sklearn.tree import DecisionTreeClassifier
# Model Object
dt_clf = DecisionTreeClassifier()
# Fit the object
dt_clf.fit(X_train, y_train)
dt_metrics = compute_metrics(X_test, y_test, dt_clf, "DecisionTree")
dt_metrics


from xgboost import XGBClassifier
# XGB model
xgb_model = XGBClassifier(n_estimators=200)
xgb_model.fit(X_train, y_train)
xgb_metrics = compute_metrics(X_test, y_test, xgb_model, 'XGBClassifier')
xgb_metrics