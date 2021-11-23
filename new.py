# import libraries
import ftfy
import nltk
import json
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import nltk
import seaborn as sb
import warnings
import string

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Binarizer

from collections import Counter
from os import listdir, makedirs
from os.path import isfile, join, splitext, split

from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
#nltk.download('stopwords')


from wordcloud import STOPWORDS, WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.naive_bayes import GaussianNB

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier # need to import xboost calssifer

warnings.filterwarnings('ignore')
np.random.seed(0)


df = pd.read_csv('train.csv')

# specifying features and labels
X= df['Claim']
y=df['Label']

test = pd.read_csv('test.csv')
X_pred = test['Claim']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state = 0, stratify=y)

hashtag_re = re.compile(r"#\w+")
mention_re = re.compile(r"@\w+")
url_re = re.compile(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*")
extras_re = re.compile("[.;:!\'?,\"()\[\]]")
#apos_re = "\'[a-z]*"
#leftover_re = "\S+"

""" Preprocessing the text in the statements"""
def preprocess(text):
    p_text = hashtag_re.sub("[hashtag]",text)
    p_text = mention_re.sub("[mention]",p_text)
    p_text = extras_re.sub("",p_text)
    p_text = url_re.sub("[url]",p_text)
    p_text = ftfy.fix_text(p_text)
    p_text = p_text.translate(str.maketrans("", "", string.punctuation))
    return p_text.lower()

# regular expression for custom tokenisation"
tokenise_re = re.compile(r"(\[[^\]]+\]|[-'\w]+|[^\s\w\[']+)") #([]|words|other non-space)

# defining 3 types of tokenisation

def custom_tokenise(text):
    return tokenise_re.findall(text.lower())

def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words


def nltk_twitter_tokenise(text):
    twtok = nltk.tokenize.TweetTokenizer()
    return twtok.tokenize(text.lower())

# stop words list set to english
stopwords_list = stopwords.words('english') # stop word list


# function for results of cross-validation
def print_cv_scores_summary(name, scores):
    print("{}: mean = {:.2f}%, sd = {:.2f}%, min = {:.2f}, max = {:.2f}".format(name, scores.mean()*100, scores.std()*100, scores.min()*100, scores.max()*100))


# fucntion for results of model fitting
def print_scores():
    print("Accuracy: ", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

# function for displaying confusion matrix
def confusion_matrix_heatmap(cm, index):
    cmdf = pd.DataFrame(cm, index = index, columns=index)
    dims = (10, 8)
    fig, ax = plt.subplots(figsize=dims)
    sns.heatmap(cmdf, annot=True, cmap="BuPu", center=0, fmt='g')
    ax.set_ylabel('Actual')    
    ax.set_xlabel('Predicted')
    
    
#############################################################
results = pd.DataFrame()
results.reset_index(inplace=True, drop=False)
results.rename(columns={'index':'id'}, inplace=True)
results['id'] = results['id'].apply(lambda x: x+1)


results.to_csv('results/results.csv', index=False)
    
##############################################################

model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',preprocessor=preprocess,tokenizer=Tokenizer,stop_words=stopwords_list)),
    ('norm2', TfidfTransformer(norm=None)),
    ('selector', SelectKBest(chi2, k=1000)),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
])


# fitting the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores() # using the predefined function to display results of the classification


model.set_params(vectorizer__max_features=1000)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()


model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',preprocessor=preprocess,tokenizer=Tokenizer,stop_words=stopwords_list)),
    ('norm', Binarizer()),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

confusion_matrix_heatmap(confusion_matrix(y_test,predictions), model.classes_)


model.set_params(clf=MultinomialNB())
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()


# Fitting C-Support Vector Classifier
model.set_params(clf=svm.SVC(C=1.0, kernel='linear', degree=2, gamma='auto',random_state=500))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()


# fitting Decision tree classifier
decision_tree=tree.DecisionTreeClassifier(random_state=1000)
model.set_params(clf=decision_tree)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()


# fitting Random forest classifier
model.set_params(clf=RandomForestClassifier(random_state=1000))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

# fitting X-gradient boost algorithm
model.set_params(clf=XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1, random_state=0))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char_wb',preprocessor=preprocess,tokenizer=Tokenizer,stop_words=stopwords_list, ngram_range=(1,3))),
    ('norm', Binarizer()),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

model.set_params(clf=MultinomialNB())
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

model.set_params(clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',random_state=1000))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

decision_tree=tree.DecisionTreeClassifier(random_state=1000)
model.set_params(clf=decision_tree)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

results['DTC1'] = model.predict(X_pred)

model.set_params(clf=RandomForestClassifier(random_state=1000))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

results['RFC1'] = model.predict(X_pred)

model.set_params(clf=XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1, random_state=0))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_scores()

results['XGB1'] = model.predict(X_pred)


model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word',preprocessor=preprocess,tokenizer=Tokenizer,stop_words=stopwords_list, ngram_range=(1,3))),
    ('norm', Binarizer()),
    ('selector', SelectKBest(score_func = chi2)),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
])

search = GridSearchCV(model, cv=StratifiedKFold(n_splits=5, random_state=0), 
                      return_train_score=False, 
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit = 'f1_weighted',
                      param_grid={
                          'selector__k': [10, 50, 100, 250, 500, 1000],
                          'clf': [MultinomialNB(), XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1, random_state=0)],
                      })

search.fit(X_train, y_train)


# results of the best fit classifier
predictions = search.predict(X_test)

results['LogisticRegression3'] = search.predict(X_pred)

print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

confusion_matrix_heatmap(confusion_matrix(y_test,predictions), search.classes_)


from sklearn.base import BaseEstimator, TransformerMixin
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

df_raw_tfid= df.copy()

df_raw_tfid['TotalWords'] = df_raw_tfid['Claim'].str.split().str.len()

X = df_raw_tfid[['Claim', 'TotalWords']]

Y = df_raw_tfid['Label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, stratify = Y)

test['TotalWords'] = test['Claim'].str.split().str.len() 

X_pred = test[['Claim', 'TotalWords']]

classifier = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('colext', TextSelector('Claim')),
            ('tfidf', TfidfVectorizer(analyzer='word',preprocessor=preprocess, tokenizer=Tokenizer, stop_words=stopwords_list,
                     min_df=.0025, max_df=0.25, ngram_range=(1,3))),
            ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
        ])),
        ('words', Pipeline([
            ('wordext', NumberSelector('TotalWords')),
            ('wscaler', StandardScaler()),
        ])),
    ])),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
    ])

# logistic regression
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print_scores()

results['LogisticRegression4'] = classifier.predict(X_pred)

#becasue multinominal naive baiese deosnt fit
classifier.set_params(clf=GaussianNB())
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print_scores()

results['GNB'] = classifier.predict(X_pred)

classifier.set_params(clf=svm.SVC(C=1.0, kernel='linear', degree=2, gamma='auto',random_state=100))
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print_scores()

results['SVM2'] = classifier.predict(X_pred)


decision_tree=tree.DecisionTreeClassifier(random_state=1000)
classifier.set_params(clf=decision_tree)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print_scores()

results['DTC2'] = classifier.predict(X_pred)


classifier.set_params(clf=RandomForestClassifier(random_state=1000))
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print_scores()

results['RFC2'] = classifier.predict(X_pred)

classifier.set_params(clf=XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1, random_state=0))
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print_scores()

results['XGB2'] = classifier.predict(X_pred)


classifier_char_wb = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('colext', TextSelector('Claim')),
            ('tfidf', TfidfVectorizer(analyzer='char_wb',preprocessor=preprocess, tokenizer=Tokenizer, stop_words=stopwords_list,
                     min_df=.0025, max_df=0.25, ngram_range=(1,3))),
            ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
        ])),
        ('words', Pipeline([
            ('wordext', NumberSelector('TotalWords')),
            ('wscaler', StandardScaler()),
        ])),
    ])),
    ('clf', LogisticRegression(solver='liblinear', random_state=1000)),
    ])

#logistic regression
classifier_char_wb.fit(X_train, y_train)
predictions = classifier_char_wb.predict(X_test)
print_scores()

results['LogisticRegression5'] = classifier_char_wb.predict(X_pred)

classifier_char_wb.set_params(clf=GaussianNB())
classifier_char_wb.fit(X_train, y_train)
predictions = classifier_char_wb.predict(X_test)
print_scores()

results['GNB1'] = classifier_char_wb.predict(X_pred)

#Support vector Machine classifier
classifier_char_wb.set_params(clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))
classifier_char_wb.fit(X_train, y_train)
predictions = classifier_char_wb.predict(X_test)
print_scores()

results['SVM3'] = classifier_char_wb.predict(X_pred)

#decision tree
decision_tree=tree.DecisionTreeClassifier(random_state=1000)
classifier_char_wb.set_params(clf=decision_tree)
classifier_char_wb.fit(X_train, y_train)
predictions = classifier_char_wb.predict(X_test)
print_scores()

results['DTC3'] = classifier_char_wb.predict(X_pred)

classifier_char_wb.set_params(clf=RandomForestClassifier())
classifier_char_wb.fit(X_train, y_train)
predictions = classifier_char_wb.predict(X_test)
print_scores()

results['RFC3'] = classifier_char_wb.predict(X_pred)

classifier_char_wb.set_params(clf=XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1, random_state=0))
classifier_char_wb.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print_scores()
results['XGB3'] = classifier_char_wb.predict(X_pred)

corpus=[]

for state in df_raw_tfid['Claim']:
    
    texts=preprocess(state)
    token=nltk.word_tokenize(texts)
    corpus.append(token)

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(corpus):
    # looping through the entries and saving in the corpus
    Final_words = []
    # fitting WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag will provide the 'tag' i.e if the word is Noun(N) or Verb(V) etc.
    for word, tag in pos_tag(entry):
        # condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The processed words for each 'statement' will be store in column 'lemmatised_words in the dataframe'
    df_raw_tfid.loc[index,'Lemmatised_words'] = str(Final_words)
    
X=df_raw_tfid['Lemmatised_words'].tolist()
y=df_raw_tfid['Label']


# fitting TfidfVectorizer with the lemmatised 'statements'
Encoder = LabelEncoder()
y = Encoder.fit_transform(y)

Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(df_raw_tfid['Lemmatised_words'])
X = Tfidf_vect.transform(X)


# logistic regression classifier
logistic = LogisticRegression(solver='liblinear', random_state=0)

cv_scores = cross_validate(logistic, X, y, cv=StratifiedKFold(n_splits=5, random_state=0), return_train_score=False, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
print_cv_scores_summary("Accuracy", cv_scores['test_accuracy'])
print_cv_scores_summary("Precision", cv_scores['test_precision_weighted'])
print_cv_scores_summary("Recall", cv_scores['test_recall_weighted'])
print_cv_scores_summary("F1", cv_scores['test_f1_weighted'])

SVM_classifier=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', random_state=0)
cv_scores = cross_validate(SVM_classifier, X, y, cv=StratifiedKFold(n_splits=5, random_state=0), return_train_score=False, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
print_cv_scores_summary("Accuracy", cv_scores['test_accuracy'])
print_cv_scores_summary("Precision", cv_scores['test_precision_weighted'])
print_cv_scores_summary("Recall", cv_scores['test_recall_weighted'])
print_cv_scores_summary("F1", cv_scores['test_f1_weighted'])



classifier_biclass = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('colext', TextSelector('Claim')),
            ('tfidf', TfidfVectorizer(analyzer='word',preprocessor=preprocess, tokenizer=Tokenizer, stop_words=stopwords_list,
                     min_df=.0025, max_df=0.25, ngram_range=(1,3))),
            ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
        ])),
        ('words', Pipeline([
            ('wordext', NumberSelector('TotalWords')),
            ('wscaler', StandardScaler()),
        ])),
    ])),
    ('clf', LogisticRegression(solver='liblinear', random_state=0)),
    ])


classifier_biclass.fit(X_train, y_train)
predictions = classifier_biclass.predict(X_test)
print_scores()
results['LogisticRegression6'] = classifier_biclass.predict(X_pred)

classifier_biclass.set_params(clf=GaussianNB())
classifier_biclass.fit(X_train, y_train)
predictions = classifier_biclass.predict(X_test)
print_scores()
results['GNB2'] = classifier_biclass.predict(X_pred)


classifier.set_params(clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))
classifier_biclass.fit(X_train, y_train)
predictions = classifier_biclass.predict(X_test)
print_scores()
results['SVM4'] = classifier_biclass.predict(X_pred)


decision_tree=tree.DecisionTreeClassifier(random_state=1000)
classifier.set_params(clf=decision_tree)
classifier_biclass.fit(X_train, y_train)
predictions = classifier_biclass.predict(X_test)
print_scores()
results['DTC4'] = classifier_biclass.predict(X_pred)


classifier.set_params(clf=RandomForestClassifier())
classifier_biclass.fit(X_train, y_train)
predictions = classifier_biclass.predict(X_test)
print_scores()
results['RFC4'] = classifier_biclass.predict(X_pred)


classifier.set_params(clf=XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1, random_state=0))
classifier_biclass.fit(X_train, y_train)
predictions = classifier_biclass.predict(X_test)
print_scores()
results['XGB4'] = classifier_biclass.predict(X_pred)