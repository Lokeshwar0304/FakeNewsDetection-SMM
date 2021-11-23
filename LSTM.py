import numpy as np
from collections import Counter
import os
import getEmbeddings2
import getEmbeddings2_test
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


top_words = 1000
epoch_num = 10
batch_size = 64

# def plot_cmat(yte, ypred):
#     '''Plotting confusion matrix'''
#     skplt.plot_confusion_matrix(yte, ypred)
#     plt.show()

if not os.path.isfile('./xtr_shuffled.npy') or \
    not os.path.isfile('./xte_shuffled.npy') or \
    not os.path.isfile('./ytr_shuffled.npy') or \
    not os.path.isfile('./yte_shuffled.npy'):
    getEmbeddings2.clean_data()


xtr = np.load('./xtr_shuffled.npy', allow_pickle=True)
xte = np.load('./xte_shuffled.npy', allow_pickle=True)
y_train = np.load('./ytr_shuffled.npy', allow_pickle=True)
y_test = np.load('./yte_shuffled.npy', allow_pickle=True)


#########################################################

getEmbeddings2_test.clean_data()
xtr = np.load('./xtr_shuffled_test.npy', allow_pickle=True)

##########################################################

cnt = Counter()
x_train = []
for x in xtr:
    x_train.append(x.split())
    for word in x_train[-1]:
        cnt[word] += 1  

# Storing most common words
most_common = cnt.most_common(top_words + 1)
word_bank = {}
id_num = 1
for word, freq in most_common:
    word_bank[word] = id_num
    id_num += 1

# Encode the sentences
for news in x_train:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]

y_train = list(y_train)
y_test = list(y_test)

# Delete the short news
i = 0
while i < len(x_train):
    if len(x_train[i]) > 10:
        i += 1
    else:
        del x_train[i]
        del y_train[i]

# Generating test data
x_test = []
for x in xte:
    x_test.append(x.split())

# Encode the sentences
for news in x_test:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]


# Truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(x_train)
X_test = sequence.pad_sequences(x_test)

# Convert to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create the model
embedding_vecor_length = 32
model = Sequential()
#model.add(Embedding(top_words+2, embedding_vecor_length, input_length=max_review_length))
model.add(Embedding(top_words+2, embedding_vecor_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_num, batch_size=batch_size)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy= %.2f%%" % (scores[1]*100))

# Draw the confusion matrix
y_pred = model.predict_classes(X_test)
# plot_cmat(y_test, y_pred)

#########################################################

y_pred = model.predict_classes(X_train)

y_pred = [x[0] for x in y_pred]

results = pd.DataFrame(y_pred, columns=["Predicted"])
results.to_csv('results_lstm.csv', index=True)

##########################################################