import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords


def ClaimClean(Claim):
    Claim = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", Claim)
    Claim = Claim.lower().split()
    stops = set(stopwords.words("english"))
    Claim = [w for w in Claim if not w in stops]
    Claim = " ".join(Claim)
    return (Claim)


def cleanup(Claim):
    Claim = ClaimClean(Claim)
    Claim = Claim.translate(str.maketrans("", "", string.punctuation))
    return Claim


def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Claim' + '_%s' % str(index)]))
    return sentences


def getEmbeddings(path,vector_dimension=100, size = 0.8):
    data = pd.read_csv(path)

    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'Claim'] != data.loc[i, 'Claim']:
            missing_rows.append(i)
    #data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)
    data.dropna(how='any', axis=0, inplace=True)
    
    print(data.shape)

    for i in range(len(data)):
        data.loc[i, 'Claim'] = cleanup(data.loc[i,'Claim'])

    x = constructLabeledSentences(data['Claim'])
    y = data['Label'].values

    Claim_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, workers=7, epochs=5,
                         seed=1)
    Claim_model.build_vocab(x)
    Claim_model.train(x, total_examples=Claim_model.corpus_count, epochs=Claim_model.iter)

    train_size = int(size * len(x))
    test_size = len(x) - train_size

    Claim_train_arrays = np.zeros((train_size, vector_dimension))
    Claim_test_arrays = np.zeros((test_size, vector_dimension))
    train_Labels = np.zeros(train_size)
    test_Labels = np.zeros(test_size)

    for i in range(train_size):
        Claim_train_arrays[i] = Claim_model.docvecs['Claim_' + str(i)]
        train_Labels[i] = y[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        Claim_test_arrays[j] = Claim_model.docvecs['Claim_' + str(i)]
        test_Labels[j] = y[i]
        j = j + 1

    return Claim_train_arrays, Claim_test_arrays, train_Labels, test_Labels



#train_data, eval_data, train_Labels, eval_Labels = getEmbeddings("train.csv")

data = pd.read_csv("test.csv")

missing_rows = []
for i in range(len(data)):
    if data.loc[i, 'Claim'] != data.loc[i, 'Claim']:
        missing_rows.append(i)
    #data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)
data.dropna(how='any', axis=0, inplace=True)

print(data.shape)

for i in range(len(data)):
     data.loc[i, 'Claim'] = cleanup(data.loc[i,'Claim'])
     
x = constructLabeledSentences(data['Claim'])

Claim_model = Doc2Vec(min_count=1, window=5, vector_size=100, sample=1e-4,  workers=7, epochs=10,
                         seed=1)
Claim_model.build_vocab(x)
Claim_model.train(x, total_examples=Claim_model.corpus_count, epochs=Claim_model.iter)


train_size = int(1 * len(x))


Claim_train_arrays = np.zeros((train_size, 100))


for i in range(train_size):
    Claim_train_arrays[i] = Claim_model.docvecs['Claim_' + str(i)]