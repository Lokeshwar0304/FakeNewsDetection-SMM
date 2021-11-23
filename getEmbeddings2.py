import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


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

def clean_data():
    path = 'train.csv'
    vector_dimension=300

    data = pd.read_csv(path)

    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'Claim'] != data.loc[i, 'Claim']:
            missing_rows.append(i)
    data.dropna(how='any', axis=0, inplace=True)

    for i in range(len(data)):
        data.loc[i, 'Claim'] = cleanup(data.loc[i,'Claim'])

    data = data.sample(frac=1).reset_index(drop=True)

    x = data.loc[:,'Claim'].values
    y = data.loc[:,'Label'].values

    # train_size = int(0.8 * len(y))
    # test_size = len(x) - train_size
    
    xtr,xte,ytr,yte = train_test_split(x, y, test_size=0.2, stratify=y)

    # xtr = x[:train_size]
    # xte = x[train_size:]
    # ytr = y[:train_size]
    # yte = y[train_size:]

    np.save('xtr_shuffled.npy',xtr)
    np.save('xte_shuffled.npy',xte)
    np.save('ytr_shuffled.npy',ytr)
    np.save('yte_shuffled.npy',yte)