import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Embedding
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score
import pandas as pd
                            #hyperparameters

#training
epochs = 4
batch_size = 128

#vector-space embedding
n_dim = 64
n_unique_words = 5000
n_words_to_skip = 50
max_review_length = 100
pad_type = trunc_type = 'pre'

#neural network architecture
n_dense = 64
dropout = 0.5

                              #loading the data
(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words,skip_top=n_words_to_skip)
print(x_train[0:6])# 0  reserved for padding, 1 would be starting character, 2 is unknown

for x in x_train[0:6]:
    print(len(x))
print(y_train)
                          #restoring words from index
word_index = keras.datasets.imdb.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index['PAD'] = 0
word_index['START'] = 1
word_index['UNK'] = 2
print(word_index)
index_word = {v: k for k, v in word_index.items()}
print(x_train[0])

print(' '.join(index_word[id] for id in x_train[0]))

x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
x_valid = pad_sequences(x_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modelcheckpoint = ModelCheckpoint(filepath='sent.weights.best.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=[modelcheckpoint])

y_hat = model.predict_proba(x_valid)
print(len(y_hat))
auc = roc_auc_score(y_valid,y_hat) * 100.00
print(auc)