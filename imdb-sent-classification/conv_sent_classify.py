from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D,GlobalMaxPool1D,Dense,SpatialDropout1D,Embedding,Dropout
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint

#hyperparameters

#pad_sequences
maxlen = 400
pad_type = trunc_type = 'pre'

#Conv1D
filters = 64
k_size = 3

#embedding
n_dim = 64
n_unique_words = 5000
n_skip_words = 50
drop_embed = 0.2

#dense
n_dense = 256
dropout = 0.3


#loading the data
(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=n_unique_words, skip_top=n_skip_words)

#preprocessing the data
x_train = pad_sequences(x_train, maxlen=maxlen, padding=pad_type, truncating=trunc_type, value=0)
x_val = pad_sequences(x_val, maxlen=maxlen, padding=pad_type, truncating=trunc_type, value=0)

#neural network architecture
model = Sequential()
model.add(Embedding(n_unique_words, n_dim,input_length=maxlen))
model.add(SpatialDropout1D(drop_embed))
model.add(Conv1D(filters=filters, kernel_size=k_size, strides=1, padding='same', activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add((Dense(1, activation='sigmoid')))
model.summary()

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
modelcheckpoint = ModelCheckpoint(filepath='cnvsent.weights.best.hdf5', save_best_only=True, verbose=1)
hist = model.fit(x_train, y_train, batch_size=128, epochs=4, validation_data=(x_val, y_val), callbacks=[modelcheckpoint], verbose=1)

y_hat = model.predict_proba(x_val)

roc = roc_auc_score(y_val, y_hat) * 100.00
print(roc)


