import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import itertools
import re
import nltk
from nltk.stem import SnowballStemmer
from sklearn.linear_model import SGDClassifier
eng_stemmer = SnowballStemmer('english')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
#import contractions
from itertools import combinations
import logging
import gensim

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

logger = logging.getLogger('preprocessing')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(sh)
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def sent_to_words(sentence):
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


nlp = spacy.load('en_core_web_md')
called = 0
total = 0
    
def remove_url_content(text):
    return re.sub(r"http\S+", "", text)


def remove_email_content(text):
    return re.sub('\S*@\S*\s?', '', text)

def remove_single_quotes(text):
    return re.sub("\'", "", str(text))

def remove_new_line(text):
    return re.sub('\s+', ' ', str(text))
  
    
def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


# def expand_contractions(text):
#     """expand shortened words, e.g. don't to do not"""
#     text = contractions.fix(text)
#     return text


def remove_html_tags(text):
    return re.sub('<[^<]+?>', '', text)


def text_preprocessing(text, accented_chars=True, contractions=True, 
                       convert_num=False, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=True, remove_url=True, remove_email=True, remove_html_content=True,
                       single_quotes=True,new_line = True):
    """preprocess text with default option set to true for all steps"""
    global called
    called+=1
    #print('Processed ' + str(called) + ' of ' + str(total) + ' records', end='\r', flush=True)
    
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    # if contractions == True: #expand contractions
    #     text = expand_contractions(text)
    if lowercase == True: #convert all characters to lowercase
        text = text.lower()
    if remove_url == True:
        text = remove_url_content(text)
    if remove_email == True:
        text = remove_email_content(text)
    if remove_html_content == True:
        text = remove_html_tags(text)
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)
    if single_quotes == True:
        text = remove_single_quotes(text)
    if new_line == True:
        text = remove_new_line(text)

    text = ' '.join(next(sent_to_words(text)))
    
    # # Checking names and removing it
    # fil = [i for i in doc.ents if i.label_.lower() in ["person"]]
    # for chunks in fil:
    #     text = text.replace(str(chunks), '')
        
    doc = nlp(text) #tokenise texts

    clean_text = []
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)  
    
    clean_text = ' '.join(clean_text)  
       
    return [text,clean_text]


def replace(text):
    return text.replace(' ', '_')


# # Replacing N_grams
# def replace_ngrams(text):
#     ngrams = pd.read_csv('./Token Transformations/ngrams.csv')
#     ngrams['mapping_new'] = ngrams['Mapping'].apply(replace)
    
#     for i, ngram in enumerate(ngrams['Word'].tolist()):
#         if ngram in text:
#             text = text.replace(ngram, ngrams.loc[i, 'mapping_new'])
#     return text

# def join_wrds(text):
#     abbr = pd.read_csv('./Token Transformations/abbr.csv')

#     # Abbreviation List
#     abbr_list = abbr['Word'].tolist()

#     # Abbreviation Dict
#     abbr_dict = {}
#     for i, row in abbr.iterrows():
#         abbr_dict[row['Word']] = row['abbr']
#     abbr_dict
    
#     clean_text = []
    
#     for token in text:
#         # Further check for stop words
#         if (token in custom_stopwords) or ('asu.edu' in token) or ('yiv' in token):
#             continue
#         # Replace abbreviations
#         if token in abbr_list:
#             token = abbr_dict[token]
#         if (token == 'pay' or token == 'payer'):
#             token = 'payment'
#         elif token == 'reimburse':
#             token = 'reimbursement'
#         elif token == 'unpaid':
#             token = 'nonpayment'
#         elif token == 'repay':
#             token = 'repayment'
#         elif token == 'tution':
#             token = 'tuition'
#         elif token == 'borrower':
#             token = 'borrow'
#         elif token == 'confidentiality':
#             token = 'confidential'
#         elif (token == 'financing' or token == 'finance'):
#             token = 'financial'
#         elif token == 'bank':
#             token = 'banking'
#         elif token == 'bill':
#             token = 'billing'
#         elif token == 'acct':
#             token = 'account'
#         elif token == 'tuition_fees' or token == 'tuition_feess':
#             token = 'tuition_fee'
#         elif token == 'business_services':
#             token = 'business_service'
#         clean_text.append(token)
    
#     tmp = (" ").join(clean_text)
    
#     # Replacing fasfa with fafsa
#     tmp =  tmp.replace('fasfa', 'fafsa')
#     tmp =  tmp.replace('refunded', 'refund')
#     return tmp


def preprocess(text):

    prep = text_preprocessing(text)
    return prep
    #print(prep)
    

#Stemming
def stem_tokens(tokens, stemmer = SnowballStemmer('english')):
    stemmed = []
    for token in tokens.split():
        stemmed.append(stemmer.stem(token))
    return  ' '.join(stemmed)

#process the data
def process_data(data,exclude_stopword=True,stem=True):
    # tokens = [w.lower() for w in data.split()]
    # tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(data, eng_stemmer)
    #tokens_stemmed = [w for w in tokens_stemmed.split() if w not in stopwords ]
    return tokens_stemmed


df = pd.read_csv('train.csv')
df = pd.read_csv('test.csv')

df['preproc_claims'] = df['Claim'].apply(preprocess)

df['clean_claims'] = df['preproc_claims'].apply(lambda x: x[0])
df['preproc_claims'] = df['preproc_claims'].apply(lambda x: x[1])

df.drop(columns=['preproc_claims'], inplace=True)

df['clean_claims_wsw'] = df['clean_claims'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
df['clean_claims_stems'] = df['clean_claims_wsw'].apply(process_data)

#df.to_csv('preprocessed_claims.csv', index=False)
#df = pd.read_csv('preprocessed_claims.csv')
#df.to_csv('preprocessed_claims_stems.csv', index= False)
#data = pd.read_csv('preprocessed_claims_wsw.csv')

df.dropna(how='any', axis=0, inplace=True)

X_train,X_test,y_train,y_test = train_test_split(df['clean_claims_stems'], df.Label, test_size=0.1, stratify=df.Label)

X_pred = df['clean_claims_stems']

dct = dict()

#############################################################
results = pd.DataFrame()
results.reset_index(inplace=True, drop=False)
results.rename(columns={'index':'id'}, inplace=True)
results['id'] = results['id'].apply(lambda x: x+1)


if os.path.isfile('./results/results.csv'):
    os.remove('./results/results.csv')
    results.to_csv('results/results.csv', index=False)
##############################################################
                                                          # MultinomialNB
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', NB_classifier)])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

dct['Naive Bayes'] = round(accuracy_score(y_test, prediction)*100,2)

cm =confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, [0, 1, 2, 3])


                                                          # LogisticRegression
# Vectorizing and applying TF-IDF
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
dct['Logistic Regression'] = round(accuracy_score(y_test, prediction)*100,2)


                                                        # DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
dct['Decision Tree'] = round(accuracy_score(y_test, prediction)*100,2)

                                                        # RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
dct['Random Forest'] = round(accuracy_score(y_test, prediction)*100,2)

                                                        # svm
from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', clf)])
model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
dct['SVM'] = round(accuracy_score(y_test, prediction)*100,2)


#using SVM Stochastic Gradient Descent on hinge loss
sgd_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
        ])

model = sgd_pipeline.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

# sgd_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
# predicted_sgd = sgd_pipeline.predict(DataPrep.test_news['Statement'])
# np.mean(predicted_sgd == DataPrep.test_news['Label'])


# PLot
cm =confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, [0, 1, 2, 3])


                                                        # Results
results = pd.DataFrame(prediction, columns=["Predicted"])
results.to_csv('results.csv', index=True)



