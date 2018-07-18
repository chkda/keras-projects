from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SpatialDropout1D,Dense,Dropout,Embedding,SimpleRNN
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb

#hyperparameters
#pad_sequences
max_len = 100
pad_type = trunc_type ='pre'

#vectorspace
n_dim = 64
n_unique_words = 5000
n_skip_words = 50
drop_embed = 0.2

#rnn
n_cells = 256
drop_rnn = 0.2


#dense
n_dense = 64
dropout = 0.3

#loading the data
(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=n_unique_words, skip_top=n_skip_words)

#preprocess the data
x_train = pad_sequences(x_train, maxlen=max_len, padding=pad_type, truncating=trunc_type, value=0)
x_val = pad_sequences(x_val, maxlen=max_len, padding=pad_type, truncating=trunc_type, value=0)

#neural network architecture
model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_len))
model.add(SpatialDropout1D(drop_embed))
model.add(SimpleRNN(n_cells, dropout=drop_rnn))
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='rnn_sent.weights.best.hdf5', save_best_only=True, verbose=1)
hist = model.fit(x_train, y_train, batch_size=128, epochs=4, verbose=1, callbacks=[checkpoint], validation_data=(x_val, y_val), shuffle=True)

y_hat = model.predict_proba(x_val)
roc = roc_auc_score(y_val, y_hat)
print(roc)

