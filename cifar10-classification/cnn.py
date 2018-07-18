# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:46:36 2018

@author: ckd08
"""
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,GlobalAveragePooling2D

(X_train,y_train),(X_test,y_test)=cifar10.load_data()
#print(len(X_train))


X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255

num_classes=len(np.unique(y_train))

y_train=np_utils.to_categorical(y_train,num_classes) 
y_test=np_utils.to_categorical(y_test,num_classes)  
(X_train,X_valid)=X_train[10000:],X_train[:10000]
(y_train,y_valid)=y_train[10000:],y_train[:10000]



model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,strides=1,padding='same',activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2,strides=1,padding='valid'))
model.add(Conv2D(filters=32,kernel_size=2,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=1,padding='valid'))
model.add(Conv2D(filters=64,kernel_size=2,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=1,padding='valid'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(.2))
#model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(.2))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#checkpointer=ModelCheckpoint(filepath='cnn.weights.best.hdf5',verbose=1,save_best_only=True)
#hist=model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)
model.load_weights('cnn.weights.best.hdf5')
score=model.evaluate(X_test,y_test)
print(score[1])