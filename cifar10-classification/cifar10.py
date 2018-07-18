# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:08:29 2018

@author: ckd08
"""

from keras.datasets import cifar10
import numpy as np
#import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation,Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
print(len(X_test))
#print('el')
#fig=plt.figure(figsize=(20,5))
#for i in range(36):
    #ax=fig.add_subplot(3,12,i+1,xticks=[],yticks=[])
    #ax.imshow(np.squeeze(X_train[i]))
X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255

num_classes=len(np.unique(y_train))

y_train=np_utils.to_categorical(y_train,num_classes) 
y_test=np_utils.to_categorical(y_test,num_classes)  
(X_train,X_valid)=X_train[5000:],X_train[:5000]
(y_train,y_valid)=y_train[5000:],y_train[:5000]

print(X_train.shape[0])

model=Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer=ModelCheckpoint(filepath='mlp.weights.best.hdf5',verbose=1,save_best_only=True)
hist=model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)
#model.load_weights('mlp.weights.best.hdf5')
#score=model.evaluate(X_test,y_test)
#print(score[1])